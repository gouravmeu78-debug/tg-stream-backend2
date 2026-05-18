"""
app.py
======
Multi-chat FastAPI backend that streams Telegram media with HTTP Range / 206
Partial-Content responses. ONE deployment serves unlimited generated sites.

URL pattern: /watch/{chat_id}/{msg_id}, /pdf/{chat_id}/{msg_id}, etc.
Chat IDs are auto-normalised – you can pass 3315036053, -1003315036053,
-3315036053 or @username; backend figures out the right form.

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

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("tg-stream")

API_ID        = int(os.environ.get("TG_API_ID", "0"))
API_HASH      = os.environ.get("TG_API_HASH", "")
DEFAULT_CHAT  = os.environ.get("TG_CHAT_ID", "")
SESSION       = os.environ.get("TG_SESSION_STRING", "")
PORT          = int(os.environ.get("PORT", "8080"))
TG_REQUEST_KB = int(os.environ.get("TG_REQUEST_KB", "1024"))
TG_PARALLEL   = int(os.environ.get("TG_PARALLEL", "4"))
CACHE_TTL     = int(os.environ.get("CACHE_TTL", "3600"))
ALLOWED       = {x.strip() for x in os.environ.get("ALLOWED_CHATS", "").split(",") if x.strip()}

TG_BLOCK    = max(64, min(TG_REQUEST_KB, 1024)) * 1024
TG_PARALLEL = max(1, min(TG_PARALLEL, 16))

if not API_ID or not API_HASH:
    log.warning("Missing TG_API_ID / TG_API_HASH – set them as env vars.")

# ---------------------------------------------------------------------------
# Telethon client
# ---------------------------------------------------------------------------

client = TelegramClient(
    StringSession(SESSION) if SESSION else "session",
    API_ID, API_HASH,
    device_model="TG-Stream Backend",
    system_version="1.0", app_version="1.0",
    connection_retries=10, retry_delay=2, request_retries=5,
    flood_sleep_threshold=60,
)

_chat_cache: Dict[str, object] = {}            # any chat ref -> entity
_msg_cache: Dict[Tuple[int, int], Tuple[float, object]] = {}
_dialogs_loaded = False
_start_lock = asyncio.Lock()


def _norm_chat(chat: Union[str, int]) -> str:
    return str(chat).strip()


def _candidate_ids(chat: str) -> List[Union[int, str]]:
    """
    Telegram has 3 different ways to refer to the same group/channel:
      - bare ID:                 3315036053
      - signed channel ID:       -1003315036053   (most common in URLs)
      - alternate negative ID:   -3315036053       (legacy basic-group form)
    Plus @username for public ones.

    We try them all and use whichever Telethon resolves first.
    """
    chat = chat.strip()
    if not chat:
        return []
    if chat.startswith("@") or not chat.lstrip("-").isdigit():
        return [chat]

    n = int(chat)
    candidates: List[Union[int, str]] = []
    if n < 0:
        # already negative – try as-is plus the "fixed" -100 channel form
        candidates.append(n)
        s = str(abs(n))
        if not s.startswith("100"):
            candidates.append(int("-100" + s))
        else:
            # already -100xxx form, also try the bare positive
            candidates.append(int(s[3:]))
    else:
        # positive – generate negative variants too
        s = str(n)
        candidates.append(int("-100" + s))   # most common
        candidates.append(-n)                # legacy
        candidates.append(n)                 # bare positive (rare)
    # de-dup preserving order
    seen = set(); out = []
    for c in candidates:
        if c not in seen:
            seen.add(c); out.append(c)
    return out


def _check_allowed(chat: str):
    if not ALLOWED:
        return
    # accept either the requested form OR its candidates being in the allow-list
    cands = {str(c) for c in _candidate_ids(chat)} | {_norm_chat(chat)}
    if not (cands & ALLOWED):
        raise HTTPException(403, f"Chat {chat} not in ALLOWED_CHATS")


async def ensure_started():
    """
    Make sure Telethon is connected & authorized. Reconnects on disconnect.
    Safe to call before EVERY request.
    """
    async with _start_lock:
        # Retry a few times in case of transient network issues
        for attempt in range(3):
            try:
                if not client.is_connected():
                    log.info("Telethon disconnected, reconnecting... (attempt %d)", attempt + 1)
                    await client.connect()
                if not await client.is_user_authorized():
                    raise RuntimeError("Telethon session not authorized. Set TG_SESSION_STRING.")
                return
            except (ConnectionError, OSError) as e:
                log.warning("Reconnect attempt %d failed: %s", attempt + 1, e)
                try:
                    await client.disconnect()
                except Exception:
                    pass
                if attempt == 2:
                    raise
                await asyncio.sleep(1.5)


async def _prime_dialogs():
    """
    Fetch all dialogs once so Telethon caches the entities in its session.
    After this, get_entity(channel_id) works even if we never called the API
    with that exact peer before.
    """
    global _dialogs_loaded
    if _dialogs_loaded:
        return
    try:
        await ensure_started()
        log.info("Priming dialogs cache (one-time)…")
        async for _ in client.iter_dialogs(limit=None):
            pass
        _dialogs_loaded = True
        log.info("Dialogs cache primed.")
    except Exception as e:
        log.warning("Dialog priming failed: %s", e)


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
            log.info("Resolved chat %s (via %s) -> %s",
                     key, cand, getattr(ent, "title", ent))
            return ent
        except (ValueError, Exception) as e:
            last_err = e
            continue

    # Fallback: prime dialogs and try again
    await _prime_dialogs()
    for cand in _candidate_ids(key):
        try:
            ent = await client.get_entity(cand)
            _chat_cache[key] = ent
            log.info("Resolved chat %s (after dialog prime, via %s)",
                     key, cand)
            return ent
        except Exception as e:
            last_err = e
            continue

    raise HTTPException(
        404,
        f"Could not resolve chat '{chat}'. Is your account a member of it? "
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
        return loc, doc.size, doc.mime_type or "application/octet-stream", fname or f"file_{msg.id}"

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

    raise HTTPException(415, "Unsupported media")


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
            start = max(0, size - n); end = size - 1
        else:
            start = int(start_s)
            end = int(end_s) if end_s else size - 1
        if start < 0 or end >= size or start > end:
            raise ValueError
        return start, end
    except Exception:
        raise HTTPException(416, "Invalid Range")


# ---------------------------------------------------------------------------
# Parallel streaming
# ---------------------------------------------------------------------------

async def _fetch_block(loc, offset: int, size: int) -> bytes:
    """
    Fetch a single TG_BLOCK starting at `offset`. On disconnect, reconnect
    once and retry. Always returns bytes (possibly empty if fully failed).
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
        except (ConnectionError, OSError, asyncio.CancelledError) as e:
            if isinstance(e, asyncio.CancelledError):
                raise
            log.warning("Block fetch at offset=%d failed (attempt %d): %s", offset, attempt + 1, e)
            if attempt == 0:
                try:
                    await ensure_started()
                except Exception as ee:
                    log.warning("Reconnect failed: %s", ee)
                    return b""
            else:
                return b""
        except Exception as e:
            log.warning("Block fetch error at offset=%d: %s", offset, e)
            return b""
    return b""


async def iter_telegram(loc, size: int, start: int, end: int):
    """
    Stream bytes [start..end] from Telegram.

    Guarantees yielding EXACTLY `end - start + 1` bytes — if Telegram
    fails mid-stream we pad with zeros (the player will just skip) so
    Content-Length matches and Starlette doesn't crash.
    """
    aligned_start = (start // TG_BLOCK) * TG_BLOCK
    skip_head = start - aligned_start
    bytes_needed = end - start + 1
    offsets = list(range(aligned_start, end + 1, TG_BLOCK))

    sent = 0
    in_flight: "list[asyncio.Task[bytes]]" = []
    aborted = False
    try:
        # Ensure connection is alive before starting
        try:
            await ensure_started()
        except Exception as e:
            log.warning("ensure_started failed in iter_telegram: %s", e)

        i = 0
        while i < len(offsets) and len(in_flight) < TG_PARALLEL:
            in_flight.append(asyncio.create_task(_fetch_block(loc, offsets[i], size)))
            i += 1

        first = True
        while in_flight and sent < bytes_needed:
            try:
                block = await in_flight.pop(0)
            except (asyncio.CancelledError, GeneratorExit):
                aborted = True
                raise
            if i < len(offsets):
                in_flight.append(asyncio.create_task(_fetch_block(loc, offsets[i], size)))
                i += 1

            if first:
                block = block[skip_head:] if block else b""
                first = False

            if not block:
                # Telegram failed for this block — pad with zeros so client
                # gets the expected content-length and no crash.
                remaining = bytes_needed - sent
                pad_len = min(TG_BLOCK, remaining)
                log.warning("Padding %d empty bytes (offset failed)", pad_len)
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

        # Final safety pad — if somehow we still didn't hit the full count
        # (e.g. early break) make up the difference so Content-Length matches.
        if sent < bytes_needed and not aborted:
            pad_len = bytes_needed - sent
            log.warning("Final pad %d bytes to match Content-Length", pad_len)
            yield b"\x00" * pad_len
            sent = bytes_needed

    except (asyncio.CancelledError, GeneratorExit):
        # Client disconnected (seek / pause / close tab) — silent
        pass
    except Exception as e:
        log.warning("iter_telegram error: %s", e)
        # Still pad whatever's left so Starlette doesn't blow up
        if sent < bytes_needed:
            try:
                yield b"\x00" * (bytes_needed - sent)
            except Exception:
                pass
    finally:
        for t in in_flight:
            t.cancel()


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(title="TG-Stream", version="2.1.0",
              description="Multi-chat Telegram streaming backend with chat-id auto-normalisation.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "HEAD", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Range", "Accept-Ranges", "Content-Length"],
)


async def _keepalive():
    """Ping Telegram every 60 seconds to keep the connection warm."""
    while True:
        try:
            await asyncio.sleep(60)
            if client.is_connected() and await client.is_user_authorized():
                # cheap noop call
                await client.get_me()
        except asyncio.CancelledError:
            return
        except Exception as e:
            log.warning("Keepalive ping failed: %s — will reconnect on next request", e)
            try:
                await client.disconnect()
            except Exception:
                pass


@app.on_event("startup")
async def _startup():
    if API_ID and API_HASH:
        try:
            await ensure_started()
            # Pre-warm the dialog cache so the first /watch is fast
            asyncio.create_task(_prime_dialogs())
            # Start keepalive task
            asyncio.create_task(_keepalive())
        except Exception as e:
            log.warning("Startup auth deferred: %s", e)


@app.on_event("shutdown")
async def _shutdown():
    if client.is_connected():
        await client.disconnect()


@app.get("/")
async def root():
    return JSONResponse({
        "service": "tg-stream backend",
        "version": "2.1.0",
        "developer": "Gourav Rajput",
        "endpoints": [
            "/watch/{chat}/{msg_id}", "/pdf/{chat}/{msg_id}",
            "/file/{chat}/{msg_id}", "/info/{chat}/{msg_id}",
            "/watch/{msg_id} (legacy, uses TG_CHAT_ID)",
        ],
        "default_chat": DEFAULT_CHAT or None,
        "allowed_chats": sorted(ALLOWED) if ALLOWED else "all",
        "tg_block_kb": TG_BLOCK // 1024,
        "tg_parallel": TG_PARALLEL,
        "dialogs_primed": _dialogs_loaded,
    })


# ---- Multi-chat handlers -------------------------------------------------

async def _serve(chat: str, msg_id: int, request: Request, *,
                 force_inline: Optional[str] = None, force_mime: Optional[str] = None):
    _check_allowed(chat)
    msg = await get_message(chat, msg_id)
    loc, size, mime, fname = media_info(msg)
    if force_mime:
        mime = force_mime

    range_header = request.headers.get("range") or request.headers.get("Range")
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Type": mime,
        "Cache-Control": "public, max-age=3600",
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
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
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


# ---- Legacy single-chat handlers (use TG_CHAT_ID env) --------------------

def _require_default_chat():
    if not DEFAULT_CHAT:
        raise HTTPException(400,
            "Legacy single-id URL used but TG_CHAT_ID env-var is not set on backend. "
            "Use /watch/{chat}/{id} instead.")
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

if __name__ == "__main__":
    if not SESSION and API_ID and API_HASH:
        async def _print():
            await client.connect()
            if await client.is_user_authorized():
                ss = StringSession.save(client.session)
                print("\n=== Copy this to TG_SESSION_STRING on your host ===")
                print(ss)
                print("===================================================\n")
            await client.disconnect()
        try:
            asyncio.run(_print())
        except Exception as e:
            log.warning("Could not derive StringSession: %s", e)

    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, log_level="info")
