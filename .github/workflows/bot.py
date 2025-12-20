from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import os
import re
import io
import json
import time
import uuid
import hashlib
import logging
import sqlite3
import tempfile
import threading
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Literal, Optional

import requests
import telebot

try:
    from apscheduler.schedulers.background import BackgroundScheduler
except Exception:
    BackgroundScheduler = None  # type: ignore

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


# =========================
# Types
# =========================
Role = Literal["user", "assistant", "system"]
Mode = Literal["chat", "auto", "search", "improve"]
Provider = Literal["openrouter", "openai"]


# =========================
# Helpers (env)
# =========================
def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}

def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        return int(raw) if raw is not None and raw.strip() else default
    except Exception:
        return default

def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    try:
        return float(raw) if raw is not None and raw.strip() else default
    except Exception:
        return default

def _get_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return raw.strip() if raw is not None and raw.strip() else default


# =========================
# Config
# =========================
@dataclass(frozen=True, slots=True)
class Config:
    telegram_token: str
    admin_chat_id: int | None

    cheap_provider: Provider
    openrouter_api_key: str | None
    openai_api_key: str | None
    elevenlabs_api_key: str | None

    # Models
    openai_heavy_model: str
    openai_web_model: str
    openai_vision_model: str
    openai_stt_model: str
    openai_chat_cheap_model: str

    # Limits
    memory_turns: int
    min_interval_s: float
    http_timeout_s: float
    max_file_chars: int

    # Web search cache
    search_cache_ttl_sec: int

    # TTS
    tts_enabled: bool
    voice_id: str
    tts_model: str

    # Scheduled improve
    daily_improve: bool


def load_config() -> Config:
    tg = (os.getenv("TELEGRAM_TOKEN") or "").strip()
    if not tg:
        raise RuntimeError("Missing TELEGRAM_TOKEN in .env")

    admin_raw = (os.getenv("ADMIN_CHAT_ID") or "").strip()
    admin_chat_id = int(admin_raw) if admin_raw else None

    cheap_provider = _get_str("CHEAP_PROVIDER", "openrouter").lower()
    if cheap_provider not in {"openrouter", "openai"}:
        cheap_provider = "openrouter"

    return Config(
        telegram_token=tg,
        admin_chat_id=admin_chat_id,

        cheap_provider=cheap_provider,  # type: ignore
        openrouter_api_key=(os.getenv("OPENROUTER_API_KEY") or "").strip() or None,
        openai_api_key=(os.getenv("OPENAI_API_KEY") or "").strip() or None,
        elevenlabs_api_key=(os.getenv("ELEVENLABS_API_KEY") or "").strip() or None,

        openai_heavy_model=_get_str("OPENAI_HEAVY_MODEL", "gpt-5.1-codex-max"),
        openai_web_model=_get_str("OPENAI_WEB_MODEL", "gpt-5"),
        openai_vision_model=_get_str("OPENAI_VISION_MODEL", "gpt-4o-mini"),
        openai_stt_model=_get_str("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe"),
        openai_chat_cheap_model=_get_str("OPENAI_CHAT_CHEAP_MODEL", "gpt-4o-mini"),

        memory_turns=_get_int("MEMORY_TURNS", 12),
        min_interval_s=_get_float("MIN_INTERVAL", 1.5),
        http_timeout_s=_get_float("HTTP_TIMEOUT", 25.0),
        max_file_chars=_get_int("MAX_FILE_CHARS", 120_000),

        search_cache_ttl_sec=_get_int("SEARCH_CACHE_TTL_SEC", 86400),

        tts_enabled=_get_bool("TTS_ENABLED", True),
        voice_id=_get_str("VOICE_ID", "EXAVITQu4vr4xnSDxMaL"),
        tts_model=_get_str("TTS_MODEL", "eleven_multilingual_v2"),

        daily_improve=_get_bool("DAILY_IMPROVE", True),
    )


cfg = load_config()


# =========================
# Logging
# =========================
def setup_logging() -> None:
    Path("logs").mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        "logs/bot.log", maxBytes=3_000_000, backupCount=3, encoding="utf-8"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[handler, logging.StreamHandler()],
    )

setup_logging()
log = logging.getLogger("bot")


# =========================
# Prompts
# =========================
def today_ymd() -> str:
    return time.strftime("%Y-%m-%d", time.localtime())

SYSTEM_PROMPT = (
    "Ты — профессиональный радиоведущий.\n"
    "Стиль: спокойный, уверенный, чистый, естественный.\n"
    "Короткие фразы. Ровный ритм. Без канцелярита.\n"
    "Если фраза звучит не по-человечески — перепиши.\n"
    "спокойный, уверенный, чистый, естественный ,графический, сексуальнй.\n"
    "Если вопрос про факты/версии/релизы/установку/совместимость/ codes/ python — попроси веб‑проверку или используй web_search.\n"
    "Если даёшь инструкции по установке/версии — обязательно указывай дату актуальности.\n"
)

IMPROVE_PROMPT = (
    "Ты — senior Python engineer и аудитор качества.\n"
    "Главная задача: улучшать этого Telegram-бота (Python 3.12, Windows 11).\n"
    "Ты МОЖЕШЬ: искать в интернете, анализировать, предлагать изменения.\n"
    "Ты НЕ МОЖЕШЬ: выполнять команды, менять файлы, раскрывать секреты.\n"
    "Выход: (1) P0/P1/P2 улучшения, (2) риски, (3) конкретные патчи в формате git diff.\n"
)


# =========================
# Telegram
# =========================
bot = telebot.TeleBot(cfg.telegram_token, parse_mode=None)


# =========================
# Constants / Utils
# =========================
TELEGRAM_LIMIT = 20000

def split_for_telegram(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return ["..."]
    if len(text) <= TELEGRAM_LIMIT:
        return [text]

    parts: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for line in text.splitlines(keepends=True):
        if buf_len + len(line) > TELEGRAM_LIMIT:
            chunk = "".join(buf).strip()
            if chunk:
                parts.append(chunk)
            buf = [line]
            buf_len = len(line)
        else:
            buf.append(line)
            buf_len += len(line)

    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)

    final: list[str] = []
    for p in parts:
        p = p.strip()
        while len(p) > TELEGRAM_LIMIT:
            final.append(p[:TELEGRAM_LIMIT])
            p = p[TELEGRAM_LIMIT:].strip()
        if p:
            final.append(p)
    return final

def send_long(chat_id: int, text: str) -> None:
    for part in split_for_telegram(text):
        bot.send_message(chat_id, part)

def clamp(text: str, max_len: int) -> str:
    t = (text or "").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 1].rstrip() + "…"

def clean_for_voice(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"[*_`#]", "", t)
    t = t.replace("…", "…\n").replace(". ", ".\n")
    return t.strip()

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# =========================
# SQLite (memory + settings + cache)
# =========================
DB_PATH = Path("data/bot.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

_db = sqlite3.connect(DB_PATH.as_posix(), check_same_thread=False)
_db.execute("PRAGMA journal_mode=WAL;")
_db_lock = threading.Lock()

def db_exec(sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
    with _db_lock:
        cur = _db.execute(sql, params)
        _db.commit()
        return cur

def db_query(sql: str, params: tuple[Any, ...] = ()) -> list[tuple]:
    with _db_lock:
        cur = _db.execute(sql, params)
        return cur.fetchall()

db_exec("""
CREATE TABLE IF NOT EXISTS messages (
  chat_id INTEGER NOT NULL,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  ts INTEGER NOT NULL
)
""")
db_exec("CREATE INDEX IF NOT EXISTS idx_messages_chat_ts ON messages(chat_id, ts)")

db_exec("""
CREATE TABLE IF NOT EXISTS chat_settings (
  chat_id INTEGER PRIMARY KEY,
  mode TEXT NOT NULL DEFAULT 'auto',
  voice_on INTEGER NOT NULL DEFAULT 1,
  web_on INTEGER NOT NULL DEFAULT 1,
  heavy_on INTEGER NOT NULL DEFAULT 1
)
""")

db_exec("""
CREATE TABLE IF NOT EXISTS search_cache (
  key TEXT PRIMARY KEY,
  response_text TEXT NOT NULL,
  sources_json TEXT NOT NULL,
  ts INTEGER NOT NULL
)
""")

def mem_add(chat_id: int, role: Role, content: str) -> None:
    db_exec(
        "INSERT INTO messages(chat_id, role, content, ts) VALUES(?, ?, ?, ?)",
        (chat_id, role, content, int(time.time())),
    )

def mem_recent(chat_id: int, limit: int) -> list[dict[str, str]]:
    rows = db_query(
        "SELECT role, content FROM messages WHERE chat_id=? ORDER BY ts DESC LIMIT ?",
        (chat_id, limit),
    )
    rows.reverse()
    return [{"role": r, "content": c} for (r, c) in rows if r in ("user", "assistant")]

def mem_clear(chat_id: int) -> None:
    db_exec("DELETE FROM messages WHERE chat_id=?", (chat_id,))

@dataclass(frozen=True, slots=True)
class ChatSettings:
    mode: Mode
    voice_on: bool
    web_on: bool
    heavy_on: bool

def get_settings(chat_id: int) -> ChatSettings:
    rows = db_query("SELECT mode, voice_on, web_on, heavy_on FROM chat_settings WHERE chat_id=?", (chat_id,))
    if not rows:
        db_exec("INSERT OR IGNORE INTO chat_settings(chat_id) VALUES(?)", (chat_id,))
        mode, voice_on, web_on, heavy_on = ("auto", 1, 1, 1)
    else:
        mode, voice_on, web_on, heavy_on = rows[0]
    mode_norm: Mode = mode if mode in {"chat", "auto", "search", "improve"} else "auto"  # type: ignore
    return ChatSettings(mode=mode_norm, voice_on=bool(int(voice_on)), web_on=bool(int(web_on)), heavy_on=bool(int(heavy_on)))

def set_settings(chat_id: int, *, mode: Mode | None = None, voice_on: bool | None = None, web_on: bool | None = None, heavy_on: bool | None = None) -> ChatSettings:
    cur = get_settings(chat_id)
    m = mode or cur.mode
    v = int(cur.voice_on if voice_on is None else bool(voice_on))
    w = int(cur.web_on if web_on is None else bool(web_on))
    h = int(cur.heavy_on if heavy_on is None else bool(heavy_on))
    db_exec(
        "INSERT INTO chat_settings(chat_id, mode, voice_on, web_on, heavy_on) VALUES(?, ?, ?, ?, ?) "
        "ON CONFLICT(chat_id) DO UPDATE SET mode=excluded.mode, voice_on=excluded.voice_on, web_on=excluded.web_on, heavy_on=excluded.heavy_on",
        (chat_id, m, v, w, h),
    )
    return get_settings(chat_id)


# =========================
# Rate limit
# =========================
_last_request_time: dict[int, float] = {}
def rate_limited(chat_id: int) -> bool:
    now = time.time()
    if now - _last_request_time.get(chat_id, 0.0) < cfg.min_interval_s:
        return True
    _last_request_time[chat_id] = now
    return False


# =========================
# Inline keyboard
# =========================
def kb_main(chat_id: int) -> telebot.types.InlineKeyboardMarkup:
    s = get_settings(chat_id)
    kb = telebot.types.InlineKeyboardMarkup(row_width=2)
    kb.add(
        telebot.types.InlineKeyboardButton(text=f"Mode: {s.mode}", callback_data="mode_cycle"),
        telebot.types.InlineKeyboardButton(text=f"Web: {'ON' if s.web_on else 'OFF'}", callback_data="toggle_web"),
    )
    kb.add(
        telebot.types.InlineKeyboardButton(text=f"Voice: {'ON' if (s.voice_on and cfg.tts_enabled and cfg.elevenlabs_api_key) else 'OFF'}", callback_data="toggle_voice"),
        telebot.types.InlineKeyboardButton(text=f"Heavy: {'ON' if s.heavy_on else 'OFF'}", callback_data="toggle_heavy"),
    )
    kb.add(
        telebot.types.InlineKeyboardButton(text="Help", callback_data="help"),
        telebot.types.InlineKeyboardButton(text="Improve now", callback_data="run_improve"),
    )
    return kb

@bot.callback_query_handler(func=lambda c: True)
def on_callback(call: telebot.types.CallbackQuery) -> None:
    chat_id = call.message.chat.id if call.message else call.from_user.id
    data = call.data or ""
    try:
        if data == "mode_cycle":
            s = get_settings(chat_id)
            modes: list[Mode] = ["auto", "chat", "search", "improve"]
            nxt = modes[(modes.index(s.mode) + 1) % len(modes)]
            set_settings(chat_id, mode=nxt)
            bot.edit_message_reply_markup(chat_id, call.message.message_id, reply_markup=kb_main(chat_id))
            bot.answer_callback_query(call.id, f"Mode -> {nxt}")
        elif data == "toggle_web":
            s = get_settings(chat_id)
            set_settings(chat_id, web_on=not s.web_on)
            bot.edit_message_reply_markup(chat_id, call.message.message_id, reply_markup=kb_main(chat_id))
            bot.answer_callback_query(call.id, "Web toggled")
        elif data == "toggle_voice":
            s = get_settings(chat_id)
            set_settings(chat_id, voice_on=not s.voice_on)
            bot.edit_message_reply_markup(chat_id, call.message.message_id, reply_markup=kb_main(chat_id))
            bot.answer_callback_query(call.id, "Voice toggled")
        elif data == "toggle_heavy":
            s = get_settings(chat_id)
            set_settings(chat_id, heavy_on=not s.heavy_on)
            bot.edit_message_reply_markup(chat_id, call.message.message_id, reply_markup=kb_main(chat_id))
            bot.answer_callback_query(call.id, "Heavy toggled")
        elif data == "help":
            bot.answer_callback_query(call.id)
            cmd_help_like(chat_id)
        elif data == "run_improve":
            bot.answer_callback_query(call.id, "Running improve…")
            run_improve(chat_id)
        else:
            bot.answer_callback_query(call.id, "Unknown action")
    except Exception as e:
        log.exception("callback error: %s", e)
        try:
            bot.answer_callback_query(call.id, "Error")
        except Exception:
            pass


# =========================
# OpenAI client
# =========================
openai_client: Optional[Any] = None
if cfg.openai_api_key and OpenAI is not None:
    openai_client = OpenAI(api_key=cfg.openai_api_key)


# =========================
# OpenRouter (CHEAP coding model)
# =========================
OPENROUTER_MODELS_CHEAP: list[str] = [
    "nvidia/nemotron-nano-12b-v2-vl:free",
]

def openrouter_chat(messages: list[dict[str, str]], timeout_s: float) -> str:
    if not cfg.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.openrouter_api_key}",
        "Content-Type": "application/json",
        "X-Title": "Telegram Super Bot",
    }
    last_err: Exception | None = None
    for model in OPENROUTER_MODELS_CHEAP:
        try:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.5,
                "max_tokens": 1200,
            }
            r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
            r.raise_for_status()
            data = r.json()
            text = (data["choices"][0]["message"]["content"] or "").strip()
            if text:
                return text
        except Exception as e:
            last_err = e
            time.sleep(0.5)
    raise RuntimeError(f"OpenRouter failed: {last_err}")

def openai_chat(messages: list[dict[str, str]], model: str) -> str:
    if not openai_client:
        raise RuntimeError("OPENAI_API_KEY/openai SDK not available")
    cc = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.4,
        max_tokens=1600,
    )
    return (cc.choices[0].message.content or "").strip()


# =========================
# Web search (unrestricted domains) + cache bypass for updates
# =========================
@dataclass(frozen=True, slots=True)
class WebResult:
    text: str
    sources: list[str]
    cached: bool

def _extract_sources(resp: Any) -> list[str]:
    out = getattr(resp, "output", None)
    if not out:
        return []
    sources: list[str] = []
    for item in out:
        item_type = getattr(item, "type", None) if not isinstance(item, dict) else item.get("type")
        if item_type != "web_search_call":
            continue
        action = getattr(item, "action", None) if not isinstance(item, dict) else item.get("action")
        if not action:
            continue
        srcs = getattr(action, "sources", None) if not isinstance(action, dict) else action.get("sources")
        if not srcs:
            continue
        for s in srcs:
            url = s.get("url") if isinstance(s, dict) else getattr(s, "url", None)
            if url and url not in sources:
                sources.append(url)
    return sources

def web_search(prompt: str, *, bypass_cache: bool) -> WebResult:
    if not openai_client:
        raise RuntimeError("OPENAI_API_KEY required for web_search")

    # include date in cache key so it naturally refreshes daily
    key_payload = {"prompt": prompt, "model": cfg.openai_web_model, "date": today_ymd()}
    cache_key = sha256(json.dumps(key_payload, ensure_ascii=False, sort_keys=True))
    now = int(time.time())

    if not bypass_cache:
        rows = db_query("SELECT response_text, sources_json, ts FROM search_cache WHERE key=?", (cache_key,))
        if rows:
            response_text, sources_json, ts = rows[0]
            if now - int(ts) <= cfg.search_cache_ttl_sec:
                try:
                    sources = json.loads(sources_json) if sources_json else []
                except Exception:
                    sources = []
                return WebResult(text=response_text, sources=sources, cached=True)

    # unrestricted domains: no filters
    tool: dict[str, Any] = {"type": "web_search"}

    resp = openai_client.responses.create(
        model=cfg.openai_web_model,
        reasoning={"effort": "low"},
        tools=[tool],
        tool_choice="auto",
        input=prompt,
        include=["web_search_call.action.sources"],
    )
    text = (resp.output_text or "").strip()
    sources = _extract_sources(resp)[:10]

    db_exec(
        "INSERT OR REPLACE INTO search_cache(key, response_text, sources_json, ts) VALUES(?, ?, ?, ?)",
        (cache_key, text, json.dumps(sources, ensure_ascii=False), now),
    )
    return WebResult(text=text, sources=sources, cached=False)


# =========================
# TTS (ElevenLabs)
# =========================
class ElevenTTS:
    def __init__(self, api_key: str, voice_id: str, model_id: str, timeout_s: float) -> None:
        self.api_key = api_key
        self.voice_id = voice_id
        self.model_id = model_id
        self.timeout_s = timeout_s

    def synthesize(self, text: str) -> Path | None:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
        headers = {"xi-api-key": self.api_key, "Content-Type": "application/json"}
        payload = {
            "text": text,
            "model_id": self.model_id,
            "voice_settings": {"stability": 0.6, "similarity_boost": 0.9},
        }
        r = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
        if r.status_code >= 400:
            return None
        if "audio" not in (r.headers.get("content-type") or "").lower():
            return None
        out_path = Path(tempfile.gettempdir()) / f"tg_tts_{uuid.uuid4().hex}.mp3"
        out_path.write_bytes(r.content)
        return out_path

tts: ElevenTTS | None = None
if cfg.tts_enabled and cfg.elevenlabs_api_key:
    tts = ElevenTTS(cfg.elevenlabs_api_key, cfg.voice_id, cfg.tts_model, cfg.http_timeout_s)


# =========================
# STT (voice -> text)
# =========================
def transcribe_voice(ogg_bytes: bytes) -> str:
    if not openai_client:
        raise RuntimeError("OPENAI_API_KEY required for STT")
    f = io.BytesIO(ogg_bytes)
    f.name = "voice.ogg"
    tr = openai_client.audio.transcriptions.create(model=cfg.openai_stt_model, file=f)
    return (getattr(tr, "text", "") or "").strip()


# =========================
# Routing heuristics
# =========================
def looks_heavy(text: str) -> bool:
    t = text.strip()
    if len(t) > 900:
        return True
    heavy_markers = [
        "архитектур", "рефактор", "оптимиз", "security", "уязвим",
        "сделай проект", "полный код", "docker", "ci/cd",
        "asyncio", "нагруз", "pytest", "coverage", "benchmark",
        "traceback", "stack trace", "Exception",
    ]
    code_markers = ["```", "class ", "def ", "SELECT ", "CREATE TABLE", "TypeScript", "tsconfig"]
    tl = t.lower()
    if any(m in tl for m in heavy_markers):
        return True
    if any(m in t for m in code_markers):
        return True
    return False

def is_update_query(text: str) -> bool:
    tl = text.lower()
    markers = [
        "последняя версия", "какая версия", "версия", "релиз", "release", "changelog",
        "обновлен", "update", "патч", "pip install", "requirements", "совместим",
        "windows 11", "python 3.12", "python 3.13", "telebot", "aiogram", "openai sdk",
        "deprecated", "устарел", "security advisory", "cve"
    ]
    if any(m in tl for m in markers):
        return True
    if re.search(r"\b(2024|2025|2026)\b", tl):
        return True
    return False

def looks_need_search(text: str) -> bool:
    tl = text.lower()
    triggers = [
        "сегодня", "вчера", "новости", "курс", "цена", "ссылка", "источник", "пруф",
        "что нового", "какие изменения", "актуально", "официально", "документац",
    ]
    if any(w in tl for w in triggers):
        return True
    if is_update_query(text):
        return True
    if "http://" in tl or "https://" in tl:
        return True
    return False

def build_messages(chat_id: int, user_text: str) -> list[dict[str, str]]:
    hist = mem_recent(chat_id, limit=cfg.memory_turns * 2)
    msgs: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    msgs.extend(hist)
    msgs.append({"role": "user", "content": user_text})
    return msgs


# =========================
# Answer engine
# =========================
def answer_text(chat_id: int, user_text: str, *, force_heavy: bool = False, force_search: bool = False) -> str:
    s = get_settings(chat_id)
    mode = s.mode
    messages = build_messages(chat_id, user_text)
    today = today_ymd()

    want_search = (
        force_search
        or (mode in {"search", "improve"})
        or (mode == "auto" and s.web_on and looks_need_search(user_text))
    )

    # Always bypass cache for update/version queries
    bypass_cache = is_update_query(user_text) or mode == "improve"

    if mode == "improve":
        if not openai_client:
            return "Для improve нужен OPENAI_API_KEY (web_search)."
        prompt = (
            f"Сегодня: {today}.\n"
            "Сделай веб‑поиск и подготовь улучшения.\n\n"
            + IMPROVE_PROMPT
            + "\n\nЗапрос пользователя:\n"
            + user_text
        )
        res = web_search(prompt, bypass_cache=True)
        out = res.text
        if res.sources:
            out += "\n\nИсточники:\n" + "\n".join(f"- {u}" for u in res.sources)
        out += f"\n\nАктуально на: {today} | Кэш: {'HIT' if res.cached else 'MISS'}"
        return out

    if want_search:
        if not openai_client:
            return "Поиск требует OPENAI_API_KEY. Добавь его в .env."
        prompt = (
            f"Сегодня: {today}.\n"
            "Сделай веб‑поиск и ответь кратко, точно, в радио‑стиле.\n"
            "Если это про версии/установку/обновления — проверь актуальность на сегодня.\n"
            "Если приводишь факты/версии/команды — укажи источники.\n\n"
            f"Вопрос: {user_text}"
        )
        res = web_search(prompt, bypass_cache=bypass_cache)
        out = res.text
        if res.sources:
            out += "\n\nИсточники:\n" + "\n".join(f"- {u}" for u in res.sources)
        out += f"\n\nАктуально на: {today} | Кэш: {'HIT' if res.cached else 'MISS'}"
        return out

    # Heavy routing
    want_heavy = force_heavy or (mode == "auto" and s.heavy_on and looks_heavy(user_text))
    if want_heavy:
        if not openai_client:
            return "Тяжёлый режим требует OPENAI_API_KEY."
        return openai_chat(messages, model=cfg.openai_heavy_model)

    # Cheap routing
    if cfg.cheap_provider == "openrouter":
        if cfg.openrouter_api_key:
            return openrouter_chat(messages, timeout_s=cfg.http_timeout_s)
        if openai_client:
            return openai_chat(messages, model=cfg.openai_chat_cheap_model)
        return "Нет провайдера LLM. Настрой OPENROUTER_API_KEY или OPENAI_API_KEY."

    # cheap via OpenAI
    if openai_client:
        return openai_chat(messages, model=cfg.openai_chat_cheap_model)
    if cfg.openrouter_api_key:
        return openrouter_chat(messages, timeout_s=cfg.http_timeout_s)
    return "Нет провайдера LLM. Настрой OPENROUTER_API_KEY или OPENAI_API_KEY."


# =========================
# Commands / Help
# =========================
def cmd_help_like(chat_id: int) -> None:
    s = get_settings(chat_id)
    send_long(
        chat_id,
        "Команды:\n"
        "/settings — панель кнопок\n"
        "/mode auto|chat|search|improve\n"
        "/web on|off — web_search в auto\n"
        "/heavy on|off — heavy routing в auto\n"
        "/voice on|off — озвучка (ElevenLabs)\n"
        "/search <запрос> — принудительный web_search\n"
        "/heavy <сообщение> — принудительно gpt-5.1-codex-max\n"
        "/improve — отчёт улучшений (web_search)\n"
        "/reset — очистить память\n"
        "/export — выгрузить память\n"
        "\n"
        f"Текущие настройки: mode={s.mode}, web={'ON' if s.web_on else 'OFF'}, heavy={'ON' if s.heavy_on else 'OFF'}, voice={'ON' if s.voice_on else 'OFF'}\n"
        "Поиск: без ограничений доменов.\n"
    )

@bot.message_handler(commands=["start", "help"])
def cmd_help(message: telebot.types.Message) -> None:
    chat_id = message.chat.id
    cmd_help_like(chat_id)
    bot.send_message(chat_id, "Панель:", reply_markup=kb_main(chat_id))

@bot.message_handler(commands=["settings"])
def cmd_settings(message: telebot.types.Message) -> None:
    chat_id = message.chat.id
    bot.send_message(chat_id, "Настройки:", reply_markup=kb_main(chat_id))

@bot.message_handler(commands=["reset"])
def cmd_reset(message: telebot.types.Message) -> None:
    mem_clear(message.chat.id)
    bot.send_message(message.chat.id, "Память очищена.", reply_markup=kb_main(message.chat.id))

@bot.message_handler(commands=["mode"])
def cmd_mode(message: telebot.types.Message) -> None:
    chat_id = message.chat.id
    parts = (message.text or "").split(maxsplit=1)
    if len(parts) < 2:
        bot.send_message(chat_id, f"mode={get_settings(chat_id).mode}", reply_markup=kb_main(chat_id))
        return
    m = parts[1].strip().lower()
    if m not in {"auto", "chat", "search", "improve"}:
        bot.send_message(chat_id, "Доступно: auto | chat | search | improve")
        return
    set_settings(chat_id, mode=m)  # type: ignore[arg-type]
    bot.send_message(chat_id, f"Mode -> {m}", reply_markup=kb_main(chat_id))

@bot.message_handler(commands=["web"])
def cmd_web(message: telebot.types.Message) -> None:
    chat_id = message.chat.id
    parts = (message.text or "").split(maxsplit=1)
    if len(parts) < 2:
        s = get_settings(chat_id)
        bot.send_message(chat_id, f"Web: {'ON' if s.web_on else 'OFF'}", reply_markup=kb_main(chat_id))
        return
    v = parts[1].strip().lower()
    if v not in {"on", "off"}:
        bot.send_message(chat_id, "Используй: /web on|off")
        return
    set_settings(chat_id, web_on=(v == "on"))
    bot.send_message(chat_id, f"Web -> {v}", reply_markup=kb_main(chat_id))

@bot.message_handler(commands=["heavy"])
def cmd_heavy(message: telebot.types.Message) -> None:
    chat_id = message.chat.id
    parts = (message.text or "").split(maxsplit=1)
    if len(parts) == 1:
        s = get_settings(chat_id)
        bot.send_message(chat_id, f"Heavy: {'ON' if s.heavy_on else 'OFF'}", reply_markup=kb_main(chat_id))
        return

    arg = parts[1].strip()
    if arg.lower() in {"on", "off"}:
        set_settings(chat_id, heavy_on=(arg.lower() == "on"))
        bot.send_message(chat_id, f"Heavy -> {arg.lower()}", reply_markup=kb_main(chat_id))
        return

    if rate_limited(chat_id):
        return
    bot.send_chat_action(chat_id, "typing")
    try:
        ans = answer_text(chat_id, arg, force_heavy=True)
        mem_add(chat_id, "user", "/heavy " + arg)
        mem_add(chat_id, "assistant", ans)
        send_long(chat_id, ans)
    except Exception as e:
        log.exception("heavy failed: %s", e)
        bot.send_message(chat_id, "Ошибка heavy режима.")

@bot.message_handler(commands=["voice"])
def cmd_voice(message: telebot.types.Message) -> None:
    chat_id = message.chat.id
    if not tts:
        bot.send_message(chat_id, "TTS не настроен (ELEVENLABS_API_KEY или TTS_ENABLED).", reply_markup=kb_main(chat_id))
        return
    parts = (message.text or "").split(maxsplit=1)
    if len(parts) < 2:
        s = get_settings(chat_id)
        bot.send_message(chat_id, f"Voice: {'ON' if s.voice_on else 'OFF'}", reply_markup=kb_main(chat_id))
        return
    v = parts[1].strip().lower()
    if v not in {"on", "off"}:
        bot.send_message(chat_id, "Используй: /voice on|off")
        return
    set_settings(chat_id, voice_on=(v == "on"))
    bot.send_message(chat_id, f"Voice -> {v}", reply_markup=kb_main(chat_id))

@bot.message_handler(commands=["search"])
def cmd_search(message: telebot.types.Message) -> None:
    chat_id = message.chat.id
    if rate_limited(chat_id):
        return
    parts = (message.text or "").split(maxsplit=1)
    if len(parts) < 2:
        bot.send_message(chat_id, "Пример: /search что нового в Python 3.12.10?")
        return
    q = parts[1].strip()
    bot.send_chat_action(chat_id, "typing")
    try:
        ans = answer_text(chat_id, q, force_search=True)
        mem_add(chat_id, "user", "/search " + q)
        mem_add(chat_id, "assistant", ans)
        send_long(chat_id, ans)
    except Exception as e:
        log.exception("search cmd failed: %s", e)
        bot.send_message(chat_id, "Поиск недоступен.")

@bot.message_handler(commands=["improve"])
def cmd_improve(message: telebot.types.Message) -> None:
    run_improve(message.chat.id)

def run_improve(chat_id: int) -> None:
    if rate_limited(chat_id):
        return
    bot.send_chat_action(chat_id, "typing")
    try:
        set_settings(chat_id, mode="improve")
        ans = answer_text(chat_id, "Сделай отчёт улучшений и патчи для текущей версии.", force_search=True)
        mem_add(chat_id, "user", "/improve")
        mem_add(chat_id, "assistant", ans)
        send_long(chat_id, ans)
    except Exception as e:
        log.exception("improve failed: %s", e)
        bot.send_message(chat_id, "Improve недоступен (нужен OPENAI_API_KEY).")

@bot.message_handler(commands=["export"])
def cmd_export(message: telebot.types.Message) -> None:
    chat_id = message.chat.id
    rows = db_query("SELECT role, content, ts FROM messages WHERE chat_id=? ORDER BY ts ASC", (chat_id,))
    if not rows:
        bot.send_message(chat_id, "Память пуста.")
        return
    lines = [f"[{ts}] {role}: {content}" for (role, content, ts) in rows[-400:]]
    text = "\n".join(lines)
    out_path = Path(tempfile.gettempdir()) / f"chat_{chat_id}_export.txt"
    out_path.write_text(text, encoding="utf-8")
    with out_path.open("rb") as f:
        bot.send_document(chat_id, f)
    try:
        out_path.unlink(missing_ok=True)
    except Exception:
        pass


# =========================
# Main handler (text + voice)
# =========================
def maybe_tts_send(chat_id: int, answer: str) -> None:
    s = get_settings(chat_id)
    if not (tts and s.voice_on):
        return
    tts_text = clamp(clean_for_voice(answer), 1200)
    audio_path = tts.synthesize(tts_text)
    if not audio_path or not audio_path.exists():
        return
    try:
        with audio_path.open("rb") as f:
            bot.send_voice(chat_id, f)
    finally:
        try:
            audio_path.unlink(missing_ok=True)
        except Exception:
            pass

@bot.message_handler(content_types=["text"])
def on_text(message: telebot.types.Message) -> None:
    chat_id = message.chat.id
    text = (message.text or "").strip()
    if not text:
        return
    if rate_limited(chat_id):
        return

    bot.send_chat_action(chat_id, "typing")
    try:
        if text.lower() in {"settings", "настройки"}:
            bot.send_message(chat_id, "Настройки:", reply_markup=kb_main(chat_id))
            return

        force_search = bool(re.match(r"^(search:|поиск:)", text.lower()))
        force_heavy = bool(re.match(r"^(heavy:|кодекс:|codex:)", text.lower()))

        ans = answer_text(chat_id, text, force_heavy=force_heavy, force_search=force_search)

        mem_add(chat_id, "user", text)
        mem_add(chat_id, "assistant", ans)

        send_long(chat_id, ans)
        bot.send_message(chat_id, "Панель:", reply_markup=kb_main(chat_id))
        maybe_tts_send(chat_id, ans)

    except Exception as e:
        log.exception("text handler failed: %s", e)
        bot.send_message(chat_id, "Сейчас не могу ответить. Попробуй позже.")

@bot.message_handler(content_types=["voice"])
def on_voice(message: telebot.types.Message) -> None:
    chat_id = message.chat.id
    if rate_limited(chat_id):
        return
    if not openai_client:
        bot.send_message(chat_id, "Для распознавания голосовых нужен OPENAI_API_KEY.")
        return

    bot.send_chat_action(chat_id, "typing")
    try:
        file_info = bot.get_file(message.voice.file_id)
        ogg_bytes = bot.download_file(file_info.file_path)
        text = transcribe_voice(ogg_bytes)
        if not text:
            bot.send_message(chat_id, "Не разобрал голос. Попробуй ещё раз.")
            return
        bot.send_message(chat_id, f"Я услышал: {text}")
        ans = answer_text(chat_id, text)
        mem_add(chat_id, "user", f"[voice->text] {text}")
        mem_add(chat_id, "assistant", ans)
        send_long(chat_id, ans)
        maybe_tts_send(chat_id, ans)
    except Exception as e:
        log.exception("voice failed: %s", e)
        bot.send_message(chat_id, "Не получилось обработать голосовое.")


# =========================
# Scheduled daily improve report to admin
# =========================
def schedule_daily_improve() -> None:
    if not (cfg.daily_improve and cfg.admin_chat_id):
        return
    if not openai_client:
        log.warning("Daily improve disabled: OPENAI_API_KEY missing.")
        return
    if BackgroundScheduler is None:
        log.warning("Daily improve disabled: APScheduler not installed.")
        return

    def job() -> None:
        try:
            today = today_ymd()
            prompt = (
                f"Сегодня: {today}.\n"
                + IMPROVE_PROMPT
                + "\n\nСделай короткий ежедневный отчёт (5–12 пунктов) улучшений для этого бота.\n"
                "Фокус: дешёвый OpenRouter (nemotron free), авто‑роутинг на gpt-5.1-codex-max, web_search актуальность.\n"
            )
            res = web_search(prompt, bypass_cache=True)
            text = "[Daily Improve Report]\n\n" + res.text
            if res.sources:
                text += "\n\nИсточники:\n" + "\n".join(f"- {u}" for u in res.sources)
            text += f"\n\nАктуально на: {today} | Кэш: {'HIT' if res.cached else 'MISS'}"
            send_long(cfg.admin_chat_id, text)
        except Exception as e:
            log.exception("daily improve failed: %s", e)

    sched = BackgroundScheduler()
    sched.add_job(job, "cron", hour=10, minute=0)
    sched.start()
    log.info("Daily improve scheduled for 10:00 local time")


# =========================
# Run
# =========================
if __name__ == "__main__":
    Path("data").mkdir(parents=True, exist_ok=True)
    schedule_daily_improve()
    log.info(
        "Bot started. cheap_provider=%s cheap_model=%s heavy_model=%s web_model=%s",
        cfg.cheap_provider,
        OPENROUTER_MODELS_CHEAP[0],
        cfg.openai_heavy_model,
        cfg.openai_web_model,
    )
    bot.infinity_polling(skip_pending=True)
