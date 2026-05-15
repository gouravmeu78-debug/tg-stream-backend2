"""
app.py
======
Multi-chat FastAPI backend that streams Telegram media with HTTP Range / 206
Partial-Content responses. ONE deployment serves unlimited generated sites.

URL pattern: /watch/{chat_id}/{msg_id}, /pdf/{chat_id}/{msg_id}, etc.

Backwards-compatible legacy URLs (/watch/{msg_id}) still work if you set
TG_CHAT_ID env var (uses that as the default chat).

Endpoints
---------
GET  /                            -> health
GET  /watch/{chat}/{msg_id}       -> stream video / audio (Range supported)
GET  /pdf/{chat}/{msg_id}         -> inline PDF
GET  /file/{chat}/{msg_id}        -> generic file
GET  /info/{chat}/{msg_id}        -> metadata
GET  /watch/{msg_id}              -> legacy, uses TG_CHAT_ID
GET  /pdf/{msg_id}                -> legacy
GET  /file/{msg_id}               -> legacy
GET  /info/{msg_id}               -> legacy

Environment variables
---------------------
TG_API_ID         (required)
TG_API_HASH       (required)
TG_SESSION_STRING (required)
TG_CHAT_ID        (optional)  Default chat for legacy URLs
PORT              (optional)  Default 8080
TG_REQUEST_KB     (optional)  TG block size, default 1024 (max 1024)
TG_PARALLEL       (optional)  Parallel TG workers, default 4
ALLOWED_CHATS     (optional)  Comma-separated chat ids; if set, only these are served

Author: Gourav Rajput
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Optional, Dict, Tuple, Union

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

_chat_cache: Dict[str, object] = {}            # chat_ref_str -> entity
_msg_cache: Dict[Tuple[str, int], Tuple[float, object]] = {}
_start_lock = asyncio.Lock()
_started = False


def _norm_chat(chat: Union[str, int]) -> str:
    """Normalise any chat reference to a comparable string."""
    return str(chat).strip()


def _parse_chat_ref(chat: str):
    chat = chat.strip()
    if chat.lstrip("-").isdigit():
        return int(chat)
    return chat  # @username


def _check_allowed(chat: str):
    if ALLOWED and _norm_chat(chat) not in ALLOWED:
        raise HTTPException(403, f"Chat {chat} not in ALLOWED_CHATS")


async def ensure_started():
    global _started
    async with _start_lock:
        if not client.is_connected():
            await client.connect()
        if not await client.is_user_authorized():
            raise RuntimeError("Telethon session not authorized. Set TG_SESSION_STRING.")
        _started = True


async def get_entity(chat: str):
    key = _norm_chat(chat)
    if key in _chat_cache:
        return _chat_cache[key]
    await ensure_started()
    ent = await client.get_entity(_parse_chat_ref(key))
    _chat_cache[key] = ent
    log.info("Resolved chat %s -> %s", key, getattr(ent, "title", ent))
    return ent


async def get_message(chat: str, msg_id: int):
    key = (_norm_chat(chat), msg_id)
    now = time.time()
    cached = _msg_cache.get(key)
    if cached and cached[0] > now:
        return cached[1]
    ent = await get_entity(chat)
    msg = await client.get_messages(ent, ids=msg_id)
    if not msg or not msg.media:
        raise HTTPException(404, "Message not found or has no media")
    _msg_cache[key] = (now + CACHE_TTL, msg)
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


async def iter_telegram(loc, size: int, start: int, end: int):
    aligned_start = (start // TG_BLOCK) * TG_BLOCK
    skip_head = start - aligned_start
    bytes_needed = end - start + 1
    offsets = list(range(aligned_start, end + 1, TG_BLOCK))

    sent = 0
    in_flight: "list[asyncio.Task[bytes]]" = []
    try:
        i = 0
        while i < len(offsets) and len(in_flight) < TG_PARALLEL:
            in_flight.append(asyncio.create_task(_fetch_block(loc, offsets[i], size)))
            i += 1

        first = True
        while in_flight:
            block = await in_flight.pop(0)
            if i < len(offsets):
                in_flight.append(asyncio.create_task(_fetch_block(loc, offsets[i], size)))
                i += 1
            if first:
                block = block[skip_head:]
                first = False
            if not block:
                continue
            remaining = bytes_needed - sent
            if len(block) >= remaining:
                yield block[:remaining]
                sent = bytes_needed
                break
            yield block
            sent += len(block)
    except (asyncio.CancelledError, GeneratorExit):
        pass
    except Exception as e:
        log.warning("iter_telegram error: %s", e)
    finally:
        for t in in_flight:
            t.cancel()


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(title="TG-Stream", version="2.0.0",
              description="Multi-chat Telegram streaming backend.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "HEAD", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Range", "Accept-Ranges", "Content-Length"],
)


@app.on_event("startup")
async def _startup():
    if API_ID and API_HASH:
        try:
            await ensure_started()
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
        "version": "2.0.0",
        "developer": "Gourav Rajput",
        "endpoints": [
            "/watch/{chat}/{msg_id}", "/pdf/{chat}/{msg_id}",
            "/file/{chat}/{msg_id}", "/info/{chat}/{msg_id}",
            "/watch/{msg_id} (legacy)",
        ],
        "default_chat": DEFAULT_CHAT or None,
        "allowed_chats": sorted(ALLOWED) if ALLOWED else "all",
        "tg_block_kb": TG_BLOCK // 1024,
        "tg_parallel": TG_PARALLEL,
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
