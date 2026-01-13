# bot.py — умный криптобот с BingX (ccxt), ATR-риском и whitelist-доступом (.env)

from __future__ import annotations

import os
import json
import math
import threading
import asyncio
from collections import defaultdict
from typing import Any, List, Set, Tuple

import nest_asyncio
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from pycoingecko import CoinGeckoAPI
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

import ccxt
from http.server import BaseHTTPRequestHandler, HTTPServer

# === Настройки ===
load_dotenv()
nest_asyncio.apply()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
BINGX_API_KEY = os.getenv("BINGX_API_KEY")
BINGX_SECRET_KEY = os.getenv("BINGX_SECRET_KEY")
OWNER_ID = int(os.getenv("TELEGRAM_OWNER_ID", "0"))

START_DEPOSIT = 1000.0
LOG_FILE = "profit_log.csv"
CFG_FILE = "config.json"
WHITELIST_FILE = "whitelist.json"

# === Инициализация файлов ===
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["timestamp", "amount", "total"]).to_csv(LOG_FILE, index=False)

if not os.path.exists(CFG_FILE):
    DEFAULT_CFG = {
        "risk_per_trade": 0.01,
        "atr_mult": 2.0,
        "leverage": 5,
        "timeframe": "15m"
    }
    with open(CFG_FILE, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_CFG, f, indent=2, ensure_ascii=False)

# === Клиент биржи ===
exchange = ccxt.bingx({
    "apiKey": BINGX_API_KEY or "",
    "secret": BINGX_SECRET_KEY or "",
    "enableRateLimit": True,
    "options": {"defaultType": "swap"},
})

# === Whitelist ===
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
        target = update.effective_chat.id if update.effective_chat else None
        if target:
            await context.bot.send_message(
                chat_id=target,
                text=(
                    "❌ Доступ запрещён.\n\n"
                    "Отправь команду /request — владелец выдаст доступ.\n"
                    f"Твой user_id: {uid}"
                ),
            )
    return wrapper

# === Утилиты ===
def get_current_deposit() -> float:
    df = pd.read_csv(LOG_FILE)
    return df["total"].iloc[-1] if not df.empty else START_DEPOSIT

# === Telegram команды ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("🔥 Горячие монеты", callback_data="hot")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    target = update.message or update.callback_query.message
    await target.reply_text("🦾 Бот активен (BingX). Выбери действие:", reply_markup=reply_markup)

@access_required
async def balance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target = update.message or update.callback_query.message
    await target.reply_text(f"💰 Текущий депозит: {get_current_deposit():.2f} USDT")

# === Горячие монеты ===
@access_required
async def hot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target = update.message or update.callback_query.message
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
    await target.reply_text(msg, parse_mode="Markdown")

# === Callback handler ===
async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    uid = update.effective_user.id if update.effective_user else 0
    await query.answer()
    if not is_allowed(uid):
        await query.edit_message_text("❌ Доступ запрещён. Используй /request для запроса доступа.")
        return
    if query.data == "hot":
        await hot(update, context)

# === Fake HTTP сервер для Render ===
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

# === Запуск бота ===
def main():
    if not TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не найден. Проверь переменные окружения.")

    # Запуск HTTP сервера
    threading.Thread(target=start_fake_web, daemon=True).start()

    app = ApplicationBuilder().token(TOKEN).build()

    # Handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("balance", balance))
    app.add_handler(CommandHandler("hot", hot))
    app.add_handler(CallbackQueryHandler(handle_callback))

    print("✅ Telegram бот запущен (Render Web Service режим)")
    app.run_polling()

if __name__ == "__main__":
    main()
