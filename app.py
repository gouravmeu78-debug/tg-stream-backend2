"""
app.py — FINAL PRODUCTION VERSION (v3.0)
=========================================
Bulletproof multi-chat Telegram streaming backend.

Features:
  ✓ HTTP Range / 206 Partial Content (proper video seeking)
  ✓ Parallel block prefetching (4-8 workers = fast streaming)
  ✓ Auto-reconnect on Telegram disconnect (3 retries)
  ✓ Block-level retry (per chunk reconnect)
  ✓ Zero-padding fallback (NEVER crashes Content-Length mismatch)
  ✓ Background keepalive every 60s (prevents idle disconnect)
  ✓ Chat-id auto-normalisation (-100 prefix variants)
  ✓ Memory-safe streaming (no buffering whole file)
  ✓ Graceful client disconnect handling
  ✓ Dialog cache pre-warming

URL pattern: /watch/{chat_id}/{msg_id}, /pdf/{chat_id}/{msg_id}, /file/{chat_id}/{msg_id}
Legacy:      /watch/{msg_id}  (uses TG_CHAT_ID env)

Env vars required:
  TG_API_ID, TG_API_HASH, TG_SESSION_STRING
Env vars optional:
  TG_CHAT_ID         (for legacy URLs)
  ALLOWED_CHATS      (comma-separated, locks backend to specific chats)
  TG_PARALLEL        (1-8, default 4 — bump to 6/8 on faster hosts)
  TG_REQUEST_KB      (max 1024, default 1024)
  PORT               (host injects)

Author: Gourav Rajput
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Optional, Dict, Tuple, Union, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, Response

from telethon import TelegramClient
from telethon.sessions import StringSession
from telethon.errors import FloodWaitError, AuthKeyDuplicatedError
from telethon.tl.types import (
    Document,
    DocumentAttributeFilename,
    InputDocumentFileLocation,
    MessageMediaDocument,
    MessageMediaPhoto,
    InputPhotoFileLocation,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger("tg-stream")
# Quiet down chatty Telethon logs (keep WARN+ only)
logging.getLogger("telethon").setLevel(logging.WARNING)

API_ID        = int(os.environ.get("TG_API_ID", "0"))
API_HASH      = os.environ.get("TG_API_HASH", "")
DEFAULT_CHAT  = os.environ.get("TG_CHAT_ID", "")
SESSION       = os.environ.get("TG_SESSION_STRING", "")
PORT          = int(os.environ.get("PORT", "8080"))
TG_REQUEST_KB = int(os.environ.get("TG_REQUEST_KB", "1024"))
TG_PARALLEL   = int(os.environ.get("TG_PARALLEL", "4"))
CACHE_TTL     = int(os.environ.get("CACHE_TTL", "3600"))
ALLOWED       = {x.strip() for x in os.environ.get("ALLOWED_CHATS", "").split(",") if x.strip()}

TG_BLOCK    = max(64, min(TG_REQUEST_KB, 1024)) * 1024   # 1 MB max
TG_PARALLEL = max(1, min(TG_PARALLEL, 8))                # cap at 8 to avoid floods

if not API_ID or not API_HASH:
    log.warning("Missing TG_API_ID / TG_API_HASH — set them as env vars.")

# ---------------------------------------------------------------------------
# Telethon client (singleton, shared across requests)
# ---------------------------------------------------------------------------

client = TelegramClient(
    StringSession(SESSION) if SESSION else "session",
    API_ID, API_HASH,
    device_model="TG-Stream Backend",
    system_version="3.0", app_version="3.0",
    connection_retries=10, retry_delay=2, request_retries=5,
    flood_sleep_threshold=120,
    auto_reconnect=True,
)

_chat_cache: Dict[str, object] = {}
_msg_cache:  Dict[Tuple[int, int], Tuple[float, object]] = {}
_dialogs_loaded = False
_start_lock = asyncio.Lock()
_keepalive_task: Optional[asyncio.Task] = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm_chat(chat: Union[str, int]) -> str:
    return str(chat).strip()


def _candidate_ids(chat: str) -> List[Union[int, str]]:
    """Generate possible Telegram ID forms (bare, -100 prefixed, etc.)."""
    chat = chat.strip()
    if not chat:
        return []
    if chat.startswith("@") or not chat.lstrip("-").isdigit():
        return [chat]
    n = int(chat)
    candidates: List[Union[int, str]] = []
    if n < 0:
        candidates.append(n)
        s = str(abs(n))
        if not s.startswith("100"):
            candidates.append(int("-100" + s))
        else:
            candidates.append(int(s[3:]))
    else:
        s = str(n)
        candidates.append(int("-100" + s))
        candidates.append(-n)
        candidates.append(n)
    seen = set(); out = []
    for c in candidates:
        if c not in seen:
            seen.add(c); out.append(c)
    return out


def _check_allowed(chat: str):
    if not ALLOWED:
        return
    cands = {str(c) for c in _candidate_ids(chat)} | {_norm_chat(chat)}
    if not (cands & ALLOWED):
        raise HTTPException(403, f"Chat {chat} not in ALLOWED_CHATS")


# ---------------------------------------------------------------------------
# Connection management — auto-reconnect with retries
# ---------------------------------------------------------------------------

async def ensure_started():
    """
    Make sure Telethon is connected & authorized.
    Reconnects on any disconnect. Safe to call before every request.
    """
    async with _start_lock:
        for attempt in range(3):
            try:
                if not client.is_connected():
                    log.info("Telethon reconnecting (attempt %d/3)...", attempt + 1)
                    await client.connect()
                if not await client.is_user_authorized():
                    raise RuntimeError(
                        "Telethon session not authorized. "
                        "Check TG_SESSION_STRING — may have been killed by "
                        "AuthKeyDuplicated (don't run same session on 2 backends)."
                    )
                return
            except (ConnectionError, OSError, asyncio.TimeoutError) as e:
                log.warning("Connection attempt %d failed: %s", attempt + 1, e)
                try:
                    await client.disconnect()
                except Exception:
                    pass
                if attempt == 2:
                    raise
                await asyncio.sleep(1.5 * (attempt + 1))
            except AuthKeyDuplicatedError:
                log.error("FATAL: session key killed — TG_SESSION_STRING used elsewhere.")
                raise RuntimeError("Telegram session killed (AuthKeyDuplicated).")


async def _prime_dialogs():
    """One-time cache of all dialogs (so entity lookups work fast)."""
    global _dialogs_loaded
    if _dialogs_loaded:
        return
    try:
        await ensure_started()
        log.info("Priming dialogs cache...")
        async for _ in client.iter_dialogs(limit=None):
            pass
        _dialogs_loaded = True
        log.info("✔ Dialogs cache primed.")
    except Exception as e:
        log.warning("Dialog priming failed: %s", e)


async def _keepalive():
    """
    Background pinger — keeps Telethon connection warm during idle periods
    (Railway/Render often kill idle connections, causing 'disconnected' errors).
    """
    log.info("Keepalive task started (60s interval)")
    while True:
        try:
            await asyncio.sleep(60)
            if not client.is_connected():
                log.info("Keepalive: reconnecting stale client")
                await ensure_started()
                continue
            try:
                await client.get_me()  # cheap, validates session
            except (ConnectionError, OSError) as e:
                log.warning("Keepalive ping error (will reconnect): %s", e)
                try:
                    await client.disconnect()
                except Exception:
                    pass
        except asyncio.CancelledError:
            log.info("Keepalive stopped")
            return
        except Exception as e:
            log.warning("Keepalive unexpected error: %s", e)


# ---------------------------------------------------------------------------
# Entity / message lookup
# ---------------------------------------------------------------------------

async def get_entity(chat: str):
    key = _norm_chat(chat)
    if key in _chat_cache:
        return _chat_cache[key]
    await ensure_started()

    last_err: Optional[Exception] = None
    for cand in _candidate_ids(key):
        try:
            ent = await client.get_entity(cand)
            _chat_cache[key] = ent
            log.info("Resolved chat %s -> %s", key, getattr(ent, "title", ent))
            return ent
        except Exception as e:
            last_err = e
            continue

    # Fallback: prime dialogs (loads all entities into session cache) and retry
    await _prime_dialogs()
    for cand in _candidate_ids(key):
        try:
            ent = await client.get_entity(cand)
            _chat_cache[key] = ent
            log.info("Resolved chat %s (after prime) -> %s", key, getattr(ent, "title", ent))
            return ent
        except Exception as e:
            last_err = e
            continue

    raise HTTPException(
        404,
        f"Could not resolve chat '{chat}'. Is your account a member? "
        f"Try the full -100xxxxxxxxxx form. Last error: {last_err}",
    )


async def get_message(chat: str, msg_id: int):
    ent = await get_entity(chat)
    cache_key = (ent.id, msg_id)
    now = time.time()
    cached = _msg_cache.get(cache_key)
    if cached and cached[0] > now:
        return cached[1]
    msg = await client.get_messages(ent, ids=msg_id)
    if not msg or not msg.media:
        raise HTTPException(404, "Message not found or has no media")
    _msg_cache[cache_key] = (now + CACHE_TTL, msg)
    return msg


def media_info(msg) -> Tuple[object, int, str, str]:
    if isinstance(msg.media, MessageMediaDocument) and msg.media.document:
        doc: Document = msg.media.document
        loc = InputDocumentFileLocation(
            id=doc.id, access_hash=doc.access_hash,
            file_reference=doc.file_reference, thumb_size="",
        )
        fname = ""
        for a in doc.attributes:
            if isinstance(a, DocumentAttributeFilename):
                fname = a.file_name
        return (
            loc,
            doc.size,
            doc.mime_type or "application/octet-stream",
            fname or f"file_{msg.id}",
        )

    if isinstance(msg.media, MessageMediaPhoto) and msg.media.photo:
        photo = msg.media.photo
        largest = max(
            (s for s in photo.sizes if hasattr(s, "size")),
            key=lambda s: getattr(s, "size", 0),
            default=photo.sizes[-1],
        )
        loc = InputPhotoFileLocation(
            id=photo.id, access_hash=photo.access_hash,
            file_reference=photo.file_reference,
            thumb_size=getattr(largest, "type", "y"),
        )
        size = getattr(largest, "size", 0) or 0
        return loc, size, "image/jpeg", f"photo_{msg.id}.jpg"

    raise HTTPException(415, "Unsupported media type")


# ---------------------------------------------------------------------------
# Range parsing
# ---------------------------------------------------------------------------

def parse_range(header: str, size: int) -> Tuple[int, int]:
    try:
        units, rng = header.split("=", 1)
        if units.strip().lower() != "bytes":
            raise ValueError
        start_s, end_s = rng.split("-", 1)
        if start_s == "":
            n = int(end_s)
            start = max(0, size - n)
            end = size - 1
        else:
            start = int(start_s)
            end = int(end_s) if end_s else size - 1
        if start < 0 or end >= size or start > end:
            raise ValueError
        return start, end
    except Exception:
        raise HTTPException(416, "Invalid Range header")


# ---------------------------------------------------------------------------
# Parallel block streaming (the fast bit)
# ---------------------------------------------------------------------------

async def _fetch_block(loc, offset: int, size: int) -> bytes:
    """
    Fetch one TG_BLOCK at `offset`. On disconnect, reconnect once and retry.
    Returns empty bytes if completely failed (so caller can pad).
    """
    for attempt in range(2):
        try:
            buf = bytearray()
            async for chunk in client.iter_download(
                loc, offset=offset, request_size=TG_BLOCK,
                file_size=size, stride=TG_BLOCK,
            ):
                if not chunk:
                    break
                buf.extend(bytes(chunk))
                if len(buf) >= TG_BLOCK:
                    break
            return bytes(buf[:TG_BLOCK])
        except asyncio.CancelledError:
            raise
        except FloodWaitError as e:
            log.warning("FloodWait %ds at offset=%d, sleeping...", e.seconds, offset)
            await asyncio.sleep(min(e.seconds, 30))
            return b""
        except (ConnectionError, OSError) as e:
            log.warning("Block fetch failed (offset=%d, attempt %d): %s",
                        offset, attempt + 1, e)
            if attempt == 0:
                try:
                    await ensure_started()
                except Exception as ee:
                    log.warning("Reconnect failed: %s", ee)
                    return b""
            else:
                return b""
        except Exception as e:
            log.warning("Block fetch error (offset=%d): %s", offset, e)
            return b""
    return b""


async def iter_telegram(loc, size: int, start: int, end: int):
    """
    Stream bytes [start..end] from Telegram.

    GUARANTEES exactly `end - start + 1` bytes — pads with zeros if needed
    so Starlette's Content-Length matches and the player doesn't crash.

    Cancels prefetch tasks on client disconnect (seek/pause) → no wasted RAM.
    """
    aligned_start = (start // TG_BLOCK) * TG_BLOCK
    skip_head = start - aligned_start
    bytes_needed = end - start + 1
    offsets = list(range(aligned_start, end + 1, TG_BLOCK))

    sent = 0
    in_flight: "list[asyncio.Task[bytes]]" = []

    try:
        # Pre-flight connection check (cheap if already connected)
        try:
            await ensure_started()
        except Exception as e:
            log.warning("Pre-flight reconnect failed: %s — will try to stream anyway", e)

        # Prime the pipeline
        i = 0
        while i < len(offsets) and len(in_flight) < TG_PARALLEL:
            in_flight.append(asyncio.create_task(_fetch_block(loc, offsets[i], size)))
            i += 1

        first = True
        while in_flight and sent < bytes_needed:
            block = await in_flight.pop(0)
            # Schedule next prefetch
            if i < len(offsets):
                in_flight.append(asyncio.create_task(_fetch_block(loc, offsets[i], size)))
                i += 1

            if first:
                block = block[skip_head:] if block else b""
                first = False

            if not block:
                # Block failed → pad with zeros
                remaining = bytes_needed - sent
                pad_len = min(TG_BLOCK, remaining)
                log.warning("Padding %d bytes (block fetch failed)", pad_len)
                yield b"\x00" * pad_len
                sent += pad_len
                continue

            remaining = bytes_needed - sent
            if len(block) >= remaining:
                yield block[:remaining]
                sent = bytes_needed
                break
            yield block
            sent += len(block)

        # Final safety pad
        if sent < bytes_needed:
            pad_len = bytes_needed - sent
            log.warning("Final pad: %d bytes", pad_len)
            yield b"\x00" * pad_len

    except (asyncio.CancelledError, GeneratorExit):
        # Client closed / seeked — normal, silent
        pass
    except Exception as e:
        log.warning("iter_telegram error: %s", e)
        # Still pad whatever's left so Starlette is happy
        if sent < bytes_needed:
            try:
                yield b"\x00" * (bytes_needed - sent)
            except Exception:
                pass
    finally:
        for t in in_flight:
            if not t.done():
                t.cancel()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="TG-Stream",
    version="3.0.0",
    description="Production-ready multi-chat Telegram streaming backend.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "HEAD", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Range", "Accept-Ranges", "Content-Length"],
)


@app.on_event("startup")
async def _startup():
    global _keepalive_task
    if API_ID and API_HASH:
        try:
            await ensure_started()
            asyncio.create_task(_prime_dialogs())
            _keepalive_task = asyncio.create_task(_keepalive())
        except Exception as e:
            log.warning("Startup deferred: %s — will retry on first request", e)


@app.on_event("shutdown")
async def _shutdown():
    if _keepalive_task and not _keepalive_task.done():
        _keepalive_task.cancel()
    if client.is_connected():
        try:
            await client.disconnect()
        except Exception:
            pass


@app.get("/")
async def root():
    return JSONResponse({
        "service": "tg-stream backend",
        "version": "3.0.0",
        "status": "ok",
        "developer": "Gourav Rajput",
        "endpoints": [
            "/watch/{chat}/{msg_id}",
            "/pdf/{chat}/{msg_id}",
            "/file/{chat}/{msg_id}",
            "/info/{chat}/{msg_id}",
            "/watch/{msg_id} (legacy, uses TG_CHAT_ID)",
        ],
        "default_chat":  DEFAULT_CHAT or None,
        "allowed_chats": sorted(ALLOWED) if ALLOWED else "all",
        "tg_block_kb":   TG_BLOCK // 1024,
        "tg_parallel":   TG_PARALLEL,
        "dialogs_primed": _dialogs_loaded,
        "telethon_connected": client.is_connected(),
    })


@app.get("/health")
async def health():
    """Simple health check (use with UptimeRobot)."""
    try:
        await ensure_started()
        return {"status": "healthy", "telethon": "connected"}
    except Exception as e:
        return JSONResponse(
            {"status": "unhealthy", "error": str(e)},
            status_code=503,
        )


# ---- Shared serve handler -------------------------------------------------

async def _serve(chat: str, msg_id: int, request: Request, *,
                 force_inline: Optional[str] = None,
                 force_mime: Optional[str] = None):
    _check_allowed(chat)
    msg = await get_message(chat, msg_id)
    loc, size, mime, fname = media_info(msg)
    if force_mime:
        mime = force_mime

    range_header = request.headers.get("range") or request.headers.get("Range")
    headers = {
        "Accept-Ranges":   "bytes",
        "Content-Type":    mime,
        "Cache-Control":   "public, max-age=3600",
        "Content-Disposition": f'{force_inline or "inline"}; filename="{fname}"',
        "Access-Control-Allow-Origin": "*",
        "X-Accel-Buffering": "no",
    }

    if request.method == "HEAD":
        headers["Content-Length"] = str(size)
        return Response(status_code=200, headers=headers)

    if range_header:
        start, end = parse_range(range_header, size)
        length = end - start + 1
        headers["Content-Range"]  = f"bytes {start}-{end}/{size}"
        headers["Content-Length"] = str(length)
        log.info("RANGE chat=%s id=%s [%s-%s] (%.1f MB)",
                 chat, msg_id, start, end, length / 1048576)
        return StreamingResponse(
            iter_telegram(loc, size, start, end),
            status_code=206, headers=headers, media_type=mime,
        )

    headers["Content-Length"] = str(size)
    log.info("FULL chat=%s id=%s (%.1f MB)", chat, msg_id, size / 1048576)
    return StreamingResponse(
        iter_telegram(loc, size, 0, size - 1),
        status_code=200, headers=headers, media_type=mime,
    )


# ---- Multi-chat routes ----------------------------------------------------

@app.get("/info/{chat}/{msg_id}")
async def info_multi(chat: str, msg_id: int):
    _check_allowed(chat)
    msg = await get_message(chat, msg_id)
    loc, size, mime, fname = media_info(msg)
    return {"chat": chat, "id": msg_id, "filename": fname, "size": size, "mime": mime}


@app.api_route("/watch/{chat}/{msg_id}", methods=["GET", "HEAD"])
async def watch_multi(chat: str, msg_id: int, request: Request):
    return await _serve(chat, msg_id, request)


@app.api_route("/pdf/{chat}/{msg_id}", methods=["GET", "HEAD"])
async def pdf_multi(chat: str, msg_id: int, request: Request):
    return await _serve(chat, msg_id, request, force_mime="application/pdf")


@app.api_route("/file/{chat}/{msg_id}", methods=["GET", "HEAD"])
async def file_multi(chat: str, msg_id: int, request: Request):
    return await _serve(chat, msg_id, request)


# ---- Legacy single-chat routes (use TG_CHAT_ID env) ----------------------

def _require_default_chat():
    if not DEFAULT_CHAT:
        raise HTTPException(
            400,
            "Legacy single-id URL used but TG_CHAT_ID env-var is not set. "
            "Use /watch/{chat}/{msg_id} instead, or set TG_CHAT_ID on the backend."
        )
    return DEFAULT_CHAT


@app.get("/info/{msg_id:int}")
async def info_legacy(msg_id: int):
    return await info_multi(_require_default_chat(), msg_id)


@app.api_route("/watch/{msg_id:int}", methods=["GET", "HEAD"])
async def watch_legacy(msg_id: int, request: Request):
    return await _serve(_require_default_chat(), msg_id, request)


@app.api_route("/pdf/{msg_id:int}", methods=["GET", "HEAD"])
async def pdf_legacy(msg_id: int, request: Request):
    return await _serve(_require_default_chat(), msg_id, request, force_mime="application/pdf")


@app.api_route("/file/{msg_id:int}", methods=["GET", "HEAD"])
async def file_legacy(msg_id: int, request: Request):
    return await _serve(_require_default_chat(), msg_id, request)


# ---------------------------------------------------------------------------
# Local entrypoint — also prints StringSession if not yet set
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not SESSION and API_ID and API_HASH:
        async def _print_string():
            await client.connect()
            if await client.is_user_authorized():
                ss = StringSession.save(client.session)
                print("\n=== Copy this to TG_SESSION_STRING on your host ===")
                print(ss)
                print("===================================================\n")
            await client.disconnect()
        try:
            asyncio.run(_print_string())
        except Exception as e:
            log.warning("Couldn't derive StringSession: %s", e)

    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, log_level="info")
