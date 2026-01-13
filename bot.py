
# bot.py — умный криптобот с BingX (ccxt), AI, ATR-риском и whitelist-доступом (.env)

from __future__ import annotations

import datetime
import json
import math
import os
from typing import Any, Dict, List, Set, Tuple

import asyncio
from collections import defaultdict

import nest_asyncio
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pycoingecko import CoinGeckoAPI
from telegram import (
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)

# === Биржа через ccxt ===
# pip install ccxt
import ccxt

# === Загрузка переменных окружения ===
load_dotenv()
nest_asyncio.apply()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
BINGX_API_KEY = os.getenv("BINGX_API_KEY")
BINGX_SECRET_KEY = os.getenv("BINGX_SECRET_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
START_DEPOSIT = 1000.0
LOG_FILE = "profit_log.csv"
CFG_FILE = "config.json"

# === Доступ/Whitelist ===
OWNER_ID = int(os.getenv("TELEGRAM_OWNER_ID", "0"))  # свой Telegram user id в .env
WHITELIST_FILE = "whitelist.json"

# === Клиенты ===
exchange = ccxt.bingx({
    "apiKey": BINGX_API_KEY or "",
    "secret": BINGX_SECRET_KEY or "",
    "enableRateLimit": True,
    "options": {"defaultType": "swap"},  # торгуем перпетуалами (swap)
})
client = OpenAI(api_key=OPENAI_API_KEY)

# === Конфиг по умолчанию ===
DEFAULT_CFG = {
    "risk_per_trade": 0.01,   # 1% от депозита на риск
    "atr_mult": 2.0,          # стоп = atr_mult * ATR
    "leverage": 5,            # используемое плечо (для инфо/расчётов)
    "timeframe": "15m"        # таймфрейм ccxt: 1m 3m 5m 15m 30m 1h 4h ...
}

# === Инициализация файлов ===
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["timestamp", "amount", "total"]).to_csv(LOG_FILE, index=False)

if not os.path.exists(CFG_FILE):
    with open(CFG_FILE, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_CFG, f, indent=2, ensure_ascii=False)

# === Антидубль/антифлуд для /setup ===
SETUP_LOCKS: defaultdict[int, asyncio.Lock] = defaultdict(asyncio.Lock)  # по chat_id
LAST_SETUP_AT: dict[int, float] = {}  # chat_id -> loop.time()

# === Whitelist helpers ===
def load_whitelist() -> List[int]:
    if not os.path.exists(WHITELIST_FILE):
        return []
    try:
        with open(WHITELIST_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("approved", []) if isinstance(data, dict) else data
    except Exception:
        return []

def save_whitelist(ids: List[int]) -> None:
    with open(WHITELIST_FILE, "w", encoding="utf-8") as f:
        json.dump({"approved": sorted(list(set(ids)))}, f, ensure_ascii=False, indent=2)

APPROVED: Set[int] = set(load_whitelist())

def is_owner(user_id: int) -> bool:
    return OWNER_ID != 0 and user_id == OWNER_ID

def is_allowed(user_id: int) -> bool:
    return is_owner(user_id) or user_id in APPROVED

def access_required(func):
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        uid = update.effective_user.id if update.effective_user else 0
        if is_allowed(uid):
            return await func(update, context, *args, **kwargs)
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=(
                "❌ Доступ запрещён.\n\n"
                "Отправь команду /request — я уведомлю владельца, и он согласует доступ.\n"
                f"Для отладки: твой user_id: {uid}"
            ),
        )
        return
    return wrapper

# === Утилиты ===
def get_cfg() -> dict:
    try:
        with open(CFG_FILE, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        for k, v in DEFAULT_CFG.items():
            cfg.setdefault(k, v)
        return cfg
    except Exception:
        return DEFAULT_CFG.copy()

def save_cfg(cfg: dict) -> None:
    with open(CFG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

def get_current_deposit() -> float:
    df = pd.read_csv(LOG_FILE)
    return df["total"].iloc[-1] if not df.empty else START_DEPOSIT

# === Маркеты BingX (USDT Perpetual / swap) ===
_cached_symbols: Set[str] = set()
_cached_markets: dict = {}

def load_markets_if_needed():
    global _cached_markets, _cached_symbols
    if not _cached_markets:
        _cached_markets = exchange.load_markets()
        _cached_symbols = {
            s for s, m in _cached_markets.items()
            if m.get("swap") and m.get("quote") == "USDT"
        }

def get_futures_symbols() -> Set[str]:
    load_markets_if_needed()
    return _cached_symbols

# === Данные и риски ===
def futures_klines(symbol: str, interval: str, limit: int = 200) -> List[List[Any]]:
    """Возвращаем список свечей в стиле: [openTime, open, high, low, close]"""
    load_markets_if_needed()
    limit = max(1, min(1000, limit))
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=interval, limit=limit)
    out: List[List[Any]] = []
    for ts, o, h, l, c, v in ohlcv:
        out.append([ts, float(o), float(h), float(l), float(c)])
    return out

def calc_atr_from_klines(klines: List[List[Any]], period: int = 14) -> float:
    highs = pd.Series([float(k[2]) for k in klines])
    lows = pd.Series([float(k[3]) for k in klines])
    closes = pd.Series([float(k[4]) for k in klines])
    prev_close = closes.shift(1)
    tr = pd.concat(
        [(highs - lows), (highs - prev_close).abs(), (lows - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    return float(atr) if not math.isnan(atr) else 0.0

# === Нормализация количества/цены ===
def get_market(symbol: str) -> dict:
    """Вернёт запись из markets по символу (учитывая defaultType=swap)."""
    load_markets_if_needed()
    mkts = _cached_markets or exchange.load_markets()
    if symbol in mkts:
        return mkts[symbol]
    # попытка найти по базе
    base = symbol.split("/")[0]
    for s, m in mkts.items():
        if m.get("swap") and m.get("quote") == "USDT" and s.split("/")[0] == base:
            return m
    raise ValueError(f"Маркет не найден для {symbol}")

def _round_to_step(value: float, step: float) -> float:
    if not step:
        return value
    return (int(value / step)) * step

def normalize_qty_price(symbol: str, qty: float, price: float | None = None) -> Tuple[float, float | None]:
    """
    Приводим qty/price к precision и limits биржи.
    Возвращаем (qty, price).
    """
    m = get_market(symbol)
    # precision
    qty_prec = (m.get("precision") or {}).get("amount")
    price_prec = (m.get("precision") or {}).get("price")

    # limits
    limits = m.get("limits") or {}
    amount_limits = limits.get("amount") or {}
    price_limits = limits.get("price") or {}

    min_amt = amount_limits.get("min")
    max_amt = amount_limits.get("max")

    # шаг количества (может лежать в info.stepSize в ряде бирж)
    info = m.get("info") or {}
    amount_step = info.get("stepSize")
    if amount_step is not None:
        try:
            amount_step = float(amount_step)
        except Exception:
            amount_step = None

    # округление по precision
    if qty_prec is not None:
        qty = float(f"{qty:.{qty_prec}f}")
    if amount_step:
        qty = _round_to_step(qty, amount_step)

    if min_amt and qty < float(min_amt):
        qty = float(min_amt)
    if max_amt and qty > float(max_amt):
        qty = float(max_amt)

    if price is not None:
        if price_prec is not None:
            price = float(f"{price:.{price_prec}f}")
        min_p = price_limits.get("min")
        max_p = price_limits.get("max")
        if min_p and price < float(min_p):
            price = float(min_p)
        if max_p and price > float(max_p):
            price = float(max_p)

    return qty, price

# === Базовые команды ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("💡 AI-анализ", callback_data="ai_menu")],
        [InlineKeyboardButton("🔥 Горячие монеты", callback_data="hot")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("🦾 Бот активен (BingX). Выбери действие:", reply_markup=reply_markup)

@access_required
async def balance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"💰 Текущий депозит: {get_current_deposit():.2f} USDT")

@access_required
async def addprofit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        amount = float(context.args[0])
        new_total = get_current_deposit() + amount
        df = pd.read_csv(LOG_FILE)
        df.loc[len(df.index)] = [datetime.datetime.now(), amount, new_total]
        df.to_csv(LOG_FILE, index=False)
        await update.message.reply_text(
            f"✅ Добавлено: +{amount:.2f} USDT\n"
            f"💰 Новый депозит: {new_total:.2f} USDT"
        )
    except Exception:
        await update.message.reply_text("❌ Используй: /addprofit <сумма>")

@access_required
async def log(update: Update, context: ContextTypes.DEFAULT_TYPE):
    df = pd.read_csv(LOG_FILE)
    if df.empty:
        await update.message.reply_text("📭 Лог пуст.")
        return
    msg = "🧾 Последние сделки:\n"
    for _, row in df.tail(5).iterrows():
        dt = pd.to_datetime(row["timestamp"]).strftime("%d.%m %H:%M")
        msg += f"• {dt} — +{row['amount']:.2f} → {row['total']:.2f} USDT\n"
    await update.message.reply_text(msg)

# === Горячие монеты по CoinGecko (как подсказка к выбору пары) ===
@access_required
async def hot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cg = CoinGeckoAPI()
    data = cg.get_coins_markets(vs_currency="usd", order="volume_desc", per_page=50, page=1)
    top = sorted(
        [c for c in data if c["total_volume"] > 10_000_000 and c["current_price"] > 0.1],
        key=lambda x: x["price_change_percentage_24h"] or 0,
        reverse=True,
    )[:3]
    msg = "🔥 *Топ трендовых монет за 24ч:*\n\n"
    for coin in top:
        msg += (
            f"🪙 *{coin['symbol'].upper()}* ({coin['name']})\n"
            f"— Цена: ${coin['current_price']:.2f}\n"
            f"— Изменение: {coin['price_change_percentage_24h']:.2f}%\n"
            f"— Объём: ${coin['total_volume'] / 1_000_000:.1f}M\n\n"
        )
    await update.message.reply_text(msg, parse_mode="Markdown")

@access_required
async def setup(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    lock = SETUP_LOCKS[chat_id]

    async with lock:
        # антифлуд: проверяем ТОЛЬКО внутри lock, чтобы второй вызов не прошёл после ожидания
        now = asyncio.get_running_loop().time()
        last = LAST_SETUP_AT.get(chat_id, 0.0)
        if now - last < 3.0:
            return
        LAST_SETUP_AT[chat_id] = now

        cg = CoinGeckoAPI()
        data = cg.get_coins_markets(vs_currency="usd", order="volume_desc", per_page=50, page=1)
        top = sorted(
            [c for c in data if c["total_volume"] > 10_000_000 and c["current_price"] > 0.1],
            key=lambda x: x["price_change_percentage_24h"] or 0,
            reverse=True,
        )

        futures_symbols = get_futures_symbols()

        def to_ccxt_pair(sym: str) -> str:
            base = sym.upper()
            for s in futures_symbols:
                if s.split("/")[0] == base:
                    return s
            return f"{base}/USDT:USDT"

        top_futures = []
        for c in top:
            pair = to_ccxt_pair(c["symbol"])
            if pair in futures_symbols:
                c["_pair"] = pair
                top_futures.append(c)

        deposit = get_current_deposit()
        cfg = get_cfg()
        risk_amount = deposit * cfg["risk_per_trade"]

        msg = "🎯 *Торговые возможности (ATR-оценка):*\n\n"

        for coin in top_futures[:3]:
            symbol = coin["_pair"]
            kl = futures_klines(symbol, cfg["timeframe"], limit=100)
            if not kl:
                continue
            last_close = float(kl[-1][4])
            atr = calc_atr_from_klines(kl)
            stop = max(last_close - cfg["atr_mult"] * atr, 0.0001)
            stop_pct = 100 * (last_close - stop) / last_close
            take = last_close * 1.05
            position_size_base = (risk_amount / (last_close - stop)) if last_close > stop else 0
            notional = position_size_base * last_close

            msg += (
                f"🪙 *{symbol}*\n"
                f"ATR(14): {atr:.4f}\n"
                f"📈 Сетап: *LONG*\n"
                f"— Вход: {last_close:.4f} USDT\n"
                f"— Стоп: {stop:.4f} (-{stop_pct:.2f}%)\n"
                f"— Тейк: {take:.4f} (+5%)\n\n"
                f"📏 Позиция: *{position_size_base:.3f} {symbol.split('/')[0]}* (~{notional:.2f} USDT)\n\n"
            )
	

        msg += (
            f"💰 Депозит: {deposit:.2f} USDT\n"
            f"⚠️ Риск на сделку: {cfg['risk_per_trade']*100:.1f}% = {risk_amount:.2f} USDT"
        )

        await update.message.reply_text(msg, parse_mode="Markdown")


# === Исполнение ордеров на BingX через ccxt ===
def _ensure_symbol(symbol: str) -> str:
    """Нормализуем ввод типа 'SOL' -> реальный своп-тикер 'SOL/USDT:USDT'."""
    load_markets_if_needed()
    base = symbol.upper()
    for s in _cached_symbols:
        if s.split("/")[0] == base:
            return s
    return f"{base}/USDT:USDT"

def _place_market_entry(symbol: str, side: str, qty_base: float) -> Dict[str, Any]:
    qty_base, _ = normalize_qty_price(symbol, qty_base, None)
    params = {"reduceOnly": False}
    return exchange.create_order(symbol, "market", side.lower(), qty_base, None, params)

def _place_tp_sl_reduce(symbol: str, side: str, qty_base: float, take_profit: float, stop_loss: float) -> Dict[str, Any]:
    """
    Ставим reduce-only TP/SL. На BingX через ccxt может потребоваться своя комбинация параметров,
    поэтому делаем две попытки.
    """
    results: Dict[str, Any] = {}

    # Попытка 1: лимитный TP (половина позиции) с reduceOnly + tpsl params
    try:
        tp_qty, tp_price = normalize_qty_price(symbol, qty_base * 0.5, take_profit)
        params = {
            "reduceOnly": True,
            "takeProfitPrice": float(take_profit),
            "stopLossPrice": float(stop_loss),
        }
        o = exchange.create_order(
            symbol, "limit",
            ("sell" if side.lower() == "buy" else "buy"),
            tp_qty, tp_price, params
        )
        results["tp_half_limit"] = o
    except Exception as e:
        results["tp_half_limit_error"] = str(e)

    # Попытка 2: стоп-рынок reduceOnly для всей позиции
    try:
        params_sl = {
            "reduceOnly": True,
            "stopPrice": float(stop_loss),
            "type": "stop_market",
        }
        o2 = exchange.create_order(
            symbol, "market",
            ("sell" if side.lower() == "buy" else "buy"),
            qty_base, None, params_sl
        )
        results["sl_market"] = o2
    except Exception as e:
        results["sl_market_error"] = str(e)

    return results

async def place_futures_trade(symbol: str, side: str, quantity: float, entry_type: str,
                             take_profit: float, stop_loss: float):
    """side: BUY/SELL, entry_type игнорируется (рынок по умолчанию)"""
    try:
        sym = _ensure_symbol(symbol)
        quantity, _ = normalize_qty_price(sym, float(quantity), None)
        take_profit = float(take_profit)
        stop_loss = float(stop_loss)

        entry = _place_market_entry(sym, "buy" if side.upper() == "BUY" else "sell", quantity)
        tpsl = _place_tp_sl_reduce(sym, "buy" if side.upper() == "BUY" else "sell", quantity, take_profit, stop_loss)
        return True, {"entry": entry, "tpsl": tpsl}
    except Exception as e:
        return False, str(e)

# === /trade ---
@access_required
async def trade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/trade <SYMBOL> — рассчитает ATR-стоп, размер и спросит подтверждение на исполнение."""
    if not context.args:
        await update.message.reply_text("❌ Используй: /trade <SYMBOL>, например /trade SOL")
        return
    cfg = get_cfg()
    deposit = get_current_deposit()
    risk_usd = deposit * cfg["risk_per_trade"]

    base = context.args[0].upper()
    symbol = _ensure_symbol(base)

    if symbol not in get_futures_symbols():
        await update.message.reply_text("⛔ Пара не торгуется на BingX как USDT-перпетуал (swap).")
        return

    kl = futures_klines(symbol, cfg["timeframe"], limit=100)
    last_close = float(kl[-1][4])
    atr = calc_atr_from_klines(kl)

    stop = max(last_close - cfg["atr_mult"] * atr, 0.0001)
    take = last_close * 1.05
    stop_dist = last_close - stop
    if stop_dist <= 0:
        await update.message.reply_text("❌ Стоп получился некорректный. Попробуй другую пару/TF.")
        return
    qty_base = risk_usd / stop_dist  # количество базового актива
    qty_base, _ = normalize_qty_price(symbol, qty_base, None)

    text = (
        f"🎛 Настройки сделки для *{symbol}*\n\n"
        f"Цена: {last_close:.4f}\n"
        f"ATR(14): {atr:.4f} | k={cfg['atr_mult']}\n"
        f"Стоп: {stop:.4f} | Тейк: {take:.4f}\n"
        f"Риск: {cfg['risk_per_trade']*100:.1f}% = {risk_usd:.2f} USDT\n"
        f"Размер: ~{qty_base:.3f} {base}\n\n"
        f"Плечо (расчётное): x{cfg['leverage']}\n"
    )
    # В callback передаём только BASE (без слеша и двоеточия), чтобы не ломать split(":")
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ Открыть LONG",  callback_data=f"open_long:{base}:{qty_base}:{take}:{stop}")],
        [InlineKeyboardButton("⬇️ Открыть SHORT", callback_data=f"open_short:{base}:{qty_base}:{take}:{stop}")],
        [InlineKeyboardButton("❌ Отмена",        callback_data="cancel_trade")]
    ])
    await update.message.reply_text(text, parse_mode="Markdown", reply_markup=kb)

# === AI анализ (данные со свечей BingX) ===
@access_required
async def ai(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if not context.args:
            await update.message.reply_text("❌ Укажи тикер монеты. Пример: /ai BTC")
            return
        base = context.args[0].upper()
        pair = _ensure_symbol(base)
        candles = futures_klines(pair, interval="15m", limit=50)
        if not candles:
            await update.message.reply_text(f"❌ Нет данных по {pair}.")
            return
        closes = [float(c[4]) for c in candles]
        if not closes:
            await update.message.reply_text("❌ Не удалось получить цены закрытия.")
            return
        closes_np = np.array(closes)
        sma = pd.Series(closes_np).rolling(window=14).mean().tolist()
        rsi_delta = np.diff(closes_np)
        up = rsi_delta.clip(min=0)
        down = -rsi_delta.clip(max=0)
        avg_gain = pd.Series(up).rolling(window=14).mean()
        avg_loss = pd.Series(down).rolling(window=14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        macd_short = pd.Series(closes_np).ewm(span=12, adjust=False).mean()
        macd_long = pd.Series(closes_np).ewm(span=26, adjust=False).mean()
        macd_line = macd_short - macd_long
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - signal_line
        trend_text = (
            f"Данные по {pair} за последние 50 свечей:\n\n"
            f"Цены закрытия: {closes[-20:]}\n"
            f"SMA (14): {sma[-5:]}\n"
            f"RSI (14): {pd.Series(rsi).dropna().tolist()[-5:]}\n"
            f"MACD: {macd_line.tolist()[-5:]}\n"
            f"MACD-гистограмма: {macd_hist.tolist()[-5:]}\n"
        )
		response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": "Ты опытный криптоаналитик. Проанализируй данные: цены, RSI, SMA, MACD. Дай краткий вывод для трейдера (входы/выходы/уровни/риск)."},
                {"role": "user", "content": trend_text}
            ]
        )
        insight = response.output_text
        await update.message.reply_text(f"📊 AI-анализ {pair}:\n\n{insight}")
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка AI-анализа: {e}")

# === Callback-и ===
async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    uid = update.effective_user.id if update.effective_user else 0
    await query.answer()

    # админ approve/revoke
    if query.data.startswith("approve:") and is_owner(uid):
        target = int(query.data.split(":")[1])
        APPROVED.add(target)
        save_whitelist(list(APPROVED))
        await query.edit_message_text(f"✅ Одобрено: {target}")
        try:
            await context.bot.send_message(chat_id=target, text="✅ Тебе выдали доступ к боту.")
        except Exception:
            pass
        return
    if query.data.startswith("revoke:") and is_owner(uid):
        target = int(query.data.split(":")[1])
        if target in APPROVED:
            APPROVED.remove(target)
            save_whitelist(list(APPROVED))
        await query.edit_message_text(f"⛔ Отозвано: {target}")
        try:
            await context.bot.send_message(chat_id=target, text="⛔ Твой доступ к боту отозван.")
        except Exception:
            pass
        return

    # блокируем неавторизованных
    if not is_allowed(uid):
        await query.edit_message_text("❌ Доступ запрещён. Используй /request для запроса доступа.")
        return

    # кнопки торговли
    if query.data.startswith("open_long:"):
        # формат: open_long:BASE:qty:tp:sl
        _, base, qty, tp, sl = query.data.split(":")
        symbol = _ensure_symbol(base)  # восстанавливаем полный символ свопа
        ok, info_or_err = await place_futures_trade(symbol, "BUY", float(qty), "MARKET", float(tp), float(sl))
        if ok:
            await query.edit_message_text(
                f"✅ LONG ордер(а) выставлены по {symbol}.\nTP={float(tp):.4f}, SL={float(sl):.4f}"
            )
        else:
            await query.edit_message_text(f"❌ Ошибка при выставлении ордеров: {info_or_err}")
        return

    if query.data.startswith("open_short:"):
        # формат: open_short:BASE:qty:tp:sl
        _, base, qty, tp, sl = query.data.split(":")
        symbol = _ensure_symbol(base)
        ok, info_or_err = await place_futures_trade(symbol, "SELL", float(qty), "MARKET", float(tp), float(sl))
        if ok:
            await query.edit_message_text(
                f"✅ SHORT ордер(а) выставлены по {symbol}.\nTP={float(tp):.4f}, SL={float(sl):.4f}"
            )
        else:
            await query.edit_message_text(f"❌ Ошибка при выставлении ордеров: {info_or_err}")
        return

    if query.data == "hot":
        await hot(update, context)
    elif query.data == "ai_menu":
        coins = ["BTC", "ETH", "SOL", "BNB", "AVAX", "OP"]
        buttons = [[InlineKeyboardButton(symbol, callback_data=f"ai_{symbol}")] for symbol in coins]
        buttons.append([InlineKeyboardButton("🔙 Назад", callback_data="back_to_main")])
        await query.edit_message_text("Выбери монету для AI-анализа:", reply_markup=InlineKeyboardMarkup(buttons))
    elif query.data == "back_to_main":
        fake_update = Update(update.update_id, message=update.effective_message)
        await start(fake_update, context)
    elif query.data.startswith("ai_"):
        symbol = query.data.split("_")[1]
        class DummyArgs:
            args = [symbol]
        update.message = query.message
        context.args = DummyArgs.args
        await ai(update, context)

# === Админ/Whitelist команды ===
async def whoami(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id if update.effective_user else 0
    await update.message.reply_text(f"🆔 Твой user_id: {uid}")

async def request_access(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id if update.effective_user else 0
    if is_allowed(uid):
        await update.message.reply_text("✅ У тебя уже есть доступ.")
        return
    if OWNER_ID == 0:
        await update.message.reply_text("❌ Владелец не настроен. Укажи TELEGRAM_OWNER_ID в .env")
        return
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ Одобрить", callback_data=f"approve:{uid}"),
         InlineKeyboardButton("⛔ Отклонить", callback_data=f"revoke:{uid}")]
    ])
    await context.bot.send_message(chat_id=OWNER_ID, text=f"🔔 Запрос доступа от user_id={uid}", reply_markup=kb)
    await update.message.reply_text("📨 Запрос отправлен владельцу. Ожидай одобрения.")

async def whitelist_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id if update.effective_user else 0
    if not is_owner(uid):
        await update.message.reply_text("❌ Только владелец может смотреть whitelist.")
        return
    lst = sorted(list(APPROVED))
    body = "\n".join(map(str, lst)) if lst else "<пусто>"
    await update.message.reply_text(f"✅ Whitelist:\n{body}")

async def approve_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id if update.effective_user else 0
    if not is_owner(uid):
        await update.message.reply_text("❌ Только владелец может одобрять.")
        return
    try:
        target = int(context.args[0])
    except Exception:
        await update.message.reply_text("Используй: /approve <user_id>")
        return
    APPROVED.add(target)
    save_whitelist(list(APPROVED))
    await update.message.reply_text(f"✅ Доступ выдан: {target}")
    try:
        await context.bot.send_message(chat_id=target, text="✅ Тебе выдали доступ к боту.")
    except Exception:
        pass

async def revoke_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id if update.effective_user else 0
    if not is_owner(uid):
        await update.message.reply_text("❌ Только владелец может отзывать доступ.")
        return
    try:
        target = int(context.args[0])
    except Exception:
        await update.message.reply_text("Используй: /revoke <user_id>")
        return
    if target in APPROVED:
        APPROVED.remove(target)
        save_whitelist(list(APPROVED))
    await update.message.reply_text(f"🗑️ Доступ отозван: {target}")
    try:
        await context.bot.send_message(chat_id=target, text="⛔ Твой доступ к боту отозван.")
    except Exception:
        pass

# === Конфиг командой /config ===
@access_required
async def config_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cfg = get_cfg()
    if not context.args:
        await update.message.reply_text(
            "Текущий конфиг:\n"
            f"risk_per_trade: {cfg['risk_per_trade']}\n"
            f"atr_mult: {cfg['atr_mult']}\n"
            f"leverage: {cfg['leverage']}\n"
            f"timeframe: {cfg['timeframe']}\n\n"
            "Изменение: /config ключ=значение (например, /config risk_per_trade=0.01 atr_mult=2 timeframe=15m)"
        )
        return
    new_cfg = cfg.copy()
    for part in context.args:
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k in ("risk_per_trade", "atr_mult"):
            try:
                new_cfg[k] = float(v)
            except Exception:
                pass
        elif k in ("leverage",):
            try:
                new_cfg[k] = int(v)
            except Exception:
                pass
        elif k in ("timeframe",):
            new_cfg[k] = v
    save_cfg(new_cfg)
    await update.message.reply_text("✅ Конфиг обновлён.")

# === Инфо о процессе (помогает ловить двойной запуск) ===
@access_required
async def botinfo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    import platform
    me = await context.bot.get_me()
    pid = os.getpid()
    await update.message.reply_text(
        f"🤖 @{me.username}\n"
        f"PID: {pid}\n"
        f"Host: {platform.node()}\n"
        f"Python: {platform.python_version()}"
    )

# === Регистрация команд в Telegram ===
async def post_init(app):
    commands = [
        BotCommand("start", "Запуск бота"),
        BotCommand("whoami", "Показать мой user_id"),
        BotCommand("request", "Запросить доступ у владельца"),
        BotCommand("whitelist", "(Owner) Показать whitelist"),
        BotCommand("approve", "(Owner) Одобрить <user_id>"),
        BotCommand("revoke", "(Owner) Отозвать <user_id>"),
        BotCommand("balance", "Показать текущий депозит"),
        BotCommand("addprofit", "Добавить прибыль вручную"),
        BotCommand("log", "Показать историю сделок"),
        BotCommand("hot", "Трендовые монеты по CoinGecko"),
        BotCommand("setup", "Авторасчёт сделки (ATR)"),
        BotCommand("trade", "Рассчитать и открыть сделку (ATR)"),
        BotCommand("config", "Показать/изменить конфиг"),
        BotCommand("ai", "AI-анализ тренда монеты"),
        BotCommand("setleverage", "Выставить плечо для символа"),
        BotCommand("botinfo", "Инфо о процессе бота"),
    ]
    await app.bot.set_my_commands(commands)

# === Доп. команда: установка плеча ===
@access_required
async def setleverage(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # /setleverage <SYMBOL> <x>
    try:
        base = context.args[0].upper()
        lev = int(context.args[1])
    except Exception:
        await update.message.reply_text("Используй: /setleverage <SYMBOL> <x>  (например, /setleverage BTC 5)")
        return
    symbol = _ensure_symbol(base)
    try:
        if hasattr(exchange, "set_leverage"):
            res = exchange.set_leverage(lev, symbol, params={})
            await update.message.reply_text(f"✅ Плечо выставлено: {symbol} → x{lev}\n{res}")
        else:
            await update.message.reply_text("⚠️ Биржа через ccxt не поддерживает set_leverage; нужен нативный REST BingX.")
    except Exception as e:
        await update.message.reply_text(f"❌ Не удалось выставить плечо: {e}")

# === Запуск ===
def main():
    if not TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не найден. Проверь .env и переменные окружения.")
    if not (BINGX_API_KEY and BINGX_SECRET_KEY):
        print("⚠️ Запуск без ключей BingX: торговые вызовы будут падать. Укажи BINGX_API_KEY/BINGX_SECRET_KEY в .env")

    try:
        load_markets_if_needed()
    except Exception as e:
        print(f"⚠️ Ошибка загрузки рынков BingX: {e}")

    app = ApplicationBuilder().token(TOKEN).post_init(post_init).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("whoami", whoami))
    app.add_handler(CommandHandler("request", request_access))
    app.add_handler(CommandHandler("whitelist", whitelist_cmd))
    app.add_handler(CommandHandler("approve", approve_cmd))
    app.add_handler(CommandHandler("revoke", revoke_cmd))

    app.add_handler(CommandHandler("balance", balance))
    app.add_handler(CommandHandler("addprofit", addprofit))
    app.add_handler(CommandHandler("log", log))
    app.add_handler(CommandHandler("hot", hot))
    app.add_handler(CommandHandler("setup", setup))
    app.add_handler(CommandHandler("trade", trade))
    app.add_handler(CommandHandler("ai", ai))
    app.add_handler(CommandHandler("config", config_cmd))
    app.add_handler(CommandHandler("setleverage", setleverage))
    app.add_handler(CommandHandler("botinfo", botinfo))

    app.add_handler(CallbackQueryHandler(handle_callback))

    print("✅ Бот запущен (BingX). Ждёт команды…")
    app.run_polling()

if __name__ == "__main__":
    import logging
    logging.basicConfig(
        filename="bot.log",
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
