# bot.py — умный криптобот с BingX (ccxt), ATR-риском и whitelist-доступом (.env)

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
import ccxt

# === Загрузка переменных окружения ===
load_dotenv()
nest_asyncio.apply()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
BINGX_API_KEY = os.getenv("BINGX_API_KEY")
BINGX_SECRET_KEY = os.getenv("BINGX_SECRET_KEY")
START_DEPOSIT = 1000.0
LOG_FILE = "profit_log.csv"
CFG_FILE = "config.json"

# === Доступ/Whitelist ===
OWNER_ID = int(os.getenv("TELEGRAM_OWNER_ID", "0"))
WHITELIST_FILE = "whitelist.json"

# === Клиенты ===
exchange = ccxt.bingx({
    "apiKey": BINGX_API_KEY or "",
    "secret": BINGX_SECRET_KEY or "",
    "enableRateLimit": True,
    "options": {"defaultType": "swap"},
})

# === Конфиг по умолчанию ===
DEFAULT_CFG = {
    "risk_per_trade": 0.01,
    "atr_mult": 2.0,
    "leverage": 5,
    "timeframe": "15m"
}

# === Инициализация файлов ===
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["timestamp", "amount", "total"]).to_csv(LOG_FILE, index=False)

if not os.path.exists(CFG_FILE):
    with open(CFG_FILE, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_CFG, f, indent=2, ensure_ascii=False)

# === Антидубль/антифлуд для /setup ===
SETUP_LOCKS: defaultdict[int, asyncio.Lock] = defaultdict(asyncio.Lock)
LAST_SETUP_AT: dict[int, float] = {}

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
                "Отправь команду /request — владелец выдаст доступ.\n"
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

# === Маркеты BingX ===
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
    load_markets_if_needed()
    limit = max(1, min(1000, limit))
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=interval, limit=limit)
    return [[ts, float(o), float(h), float(l), float(c)] for ts, o, h, l, c, v in ohlcv]

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
    load_markets_if_needed()
    mkts = _cached_markets or exchange.load_markets()
    if symbol in mkts:
        return mkts[symbol]
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
    m = get_market(symbol)
    qty_prec = (m.get("precision") or {}).get("amount")
    price_prec = (m.get("precision") or {}).get("price")
    limits = m.get("limits") or {}
    min_amt = limits.get("amount", {}).get("min")
    max_amt = limits.get("amount", {}).get("max")
    amount_step = (m.get("info") or {}).get("stepSize")
    if amount_step is not None:
        try:
            amount_step = float(amount_step)
        except Exception:
            amount_step = None
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
        min_p = limits.get("price", {}).get("min")
        max_p = limits.get("price", {}).get("max")
        if min_p and price < float(min_p):
            price = float(min_p)
        if max_p and price > float(max_p):
            price = float(max_p)
    return qty, price

# === Простейшие команды ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("🔥 Горячие монеты", callback_data="hot")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("🦾 Бот активен (BingX). Выбери действие:", reply_markup=reply_markup)

@access_required
async def balance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"💰 Текущий депозит: {get_current_deposit():.2f} USDT")

# === Горячие монеты по CoinGecko ===
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

# === Заглушка AI ===
@access_required
async def ai(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📊 AI-анализ временно отключён.")

# === Callback заглушки ===
async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    uid = update.effective_user.id if update.effective_user else 0
    await query.answer()
    if not is_allowed(uid):
        await query.edit_message_text("❌ Доступ запрещён. Используй /request для запроса доступа.")
        return
    if query.data == "hot":
        await hot(update, context)
    elif query.data.startswith("ai_"):
        symbol = query.data.split("_")[1]
        await query.edit_message_text(f"📊 AI-анализ {symbol} временно отключён.")

# === Запуск ===
def main():
    if not TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не найден. Проверь .env и переменные окружения.")
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("balance", balance))
    app.add_handler(CommandHandler("hot", hot))
    app.add_handler(CommandHandler("ai", ai))
    app.add_handler(CallbackQueryHandler(handle_callback))
    print("✅ Бот запущен. Ждёт команды…")
    app.run_polling()

if __name__ == "__main__":
    import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

def start_fake_web():
    port = int(os.environ.get("PORT", 10000))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Telegram bot is running")

    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"🌐 Fake web server listening on port {port}")
    server.serve_forever()


def main():
    if not TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не найден. Проверь переменные окружения.")

    # запускаем HTTP-порт для Render
    threading.Thread(target=start_fake_web, daemon=True).start()

    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("balance", balance))
    app.add_handler(CommandHandler("hot", hot))
    app.add_handler(CommandHandler("ai", ai))
    app.add_handler(CallbackQueryHandler(handle_callback))

    print("✅ Telegram бот запущен (Render Web Service режим)")
    app.run_polling()


if __name__ == "__main__":
    main()
