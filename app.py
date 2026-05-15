"""
app.py
======
FastAPI backend that streams Telegram media with proper HTTP Range / 206
Partial-Content responses (YouTube / Netflix-style seeking, low RAM).

Endpoints
---------
GET  /                  -> health
GET  /watch/{msg_id}    -> stream video / audio (Range supported)
GET  /pdf/{msg_id}      -> inline PDF (Range supported)
GET  /file/{msg_id}     -> generic file (Range supported)

Environment variables
---------------------
TG_API_ID         37720402  Telegram API id
TG_API_HASH       (required)  Telegram API hash
TG_CHAT_ID        (required)  Numeric id (e.g. -1001234567890) or @username
TG_SESSION_STRING (required)  Telethon StringSession (see "First run" below)
PORT              (optional)  Default 8080
CHUNK_KB          (optional)  Chunk size in KB, default 512
CACHE_TTL         (optional)  Seconds to cache message lookups, default 3600

First run – generate a StringSession (do this ONCE locally):

    python -c "from telethon.sync import TelegramClient; \
        from telethon.sessions import StringSession; \
        api_id=int(input('id: ')); api_hash=input('hash: '); \
        print(TelegramClient(StringSession(), api_id, api_hash).start().session.save())"

Paste the printed string as TG_SESSION_STRING on the host.
(You can also reuse the local session.session by running this file once locally
with TG_SESSION_STRING unset – it will print the StringSession to stdout.)

Author: Gourav Rajput
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Optional, Dict, Tuple

from fastapi import FastAPI, Header, HTTPException, Request
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

API_ID    = int(os.environ.get("TG_API_ID", "0"))
API_HASH  = os.environ.get("TG_API_HASH", "")
CHAT_ID   = os.environ.get("TG_CHAT_ID", "")
SESSION   = os.environ.get("TG_SESSION_STRING", "")
PORT      = int(os.environ.get("PORT", "8080"))
CHUNK_KB  = int(os.environ.get("CHUNK_KB", "512"))
CACHE_TTL = int(os.environ.get("CACHE_TTL", "3600"))

# Telegram requires offset to be a multiple of 4 KB and limit a power of 2 ≤ 1MB
TG_BLOCK = 1024 * 1024            # 1 MB internal Telegram block
HTTP_CHUNK = max(64, CHUNK_KB) * 1024

if not API_ID or not API_HASH or not CHAT_ID:
    log.warning("Missing TG_API_ID / TG_API_HASH / TG_CHAT_ID – set them as env vars.")

try:
    CHAT_REF: object = int(CHAT_ID)
except ValueError:
    CHAT_REF = CHAT_ID  # @username

# ---------------------------------------------------------------------------
# Telethon client (single, shared)
# ---------------------------------------------------------------------------

client = TelegramClient(
    StringSession(SESSION) if SESSION else "session",
    API_ID, API_HASH,
    device_model="TG-Stream Backend",
    system_version="1.0", app_version="1.0",
    connection_retries=10, retry_delay=2, request_retries=5,
)

# Per-request locking to avoid hammering Telegram with many concurrent
# downloads of the same chunk (helps a lot with seek-spam).
_chat_entity = None
_msg_cache: Dict[int, Tuple[float, object]] = {}
_lock = asyncio.Lock()


async def ensure_started():
    if not client.is_connected():
        await client.connect()
    if not await client.is_user_authorized():
        raise RuntimeError("Telethon session not authorized. Set TG_SESSION_STRING.")
    global _chat_entity
    if _chat_entity is None:
        _chat_entity = await client.get_entity(CHAT_REF)
        log.info("Resolved chat: %s", getattr(_chat_entity, "title", _chat_entity))


async def get_message(msg_id: int):
    now = time.time()
    cached = _msg_cache.get(msg_id)
    if cached and cached[0] > now:
        return cached[1]
    await ensure_started()
    msg = await client.get_messages(_chat_entity, ids=msg_id)
    if not msg or not msg.media:
        raise HTTPException(404, "Message not found or no media")
    _msg_cache[msg_id] = (now + CACHE_TTL, msg)
    return msg


def media_info(msg) -> Tuple[object, int, str, str]:
    """
    Returns (input_location, size, mime, filename) for the message media.
    """
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
        # pick the largest size
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
# Range-aware streaming
# ---------------------------------------------------------------------------

def parse_range(header: str, size: int) -> Tuple[int, int]:
    """Parse 'bytes=START-END' returning inclusive (start,end)."""
    try:
        units, rng = header.split("=", 1)
        if units.strip().lower() != "bytes":
            raise ValueError
        start_s, end_s = rng.split("-", 1)
        if start_s == "":
            # suffix range  bytes=-N  -> last N bytes
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


async def iter_telegram(loc, size: int, start: int, end: int):
    """
    Yield bytes from `start` to `end` (inclusive) using Telegram's
    iter_download.  We align to 1 MB Telegram blocks then trim.
    """
    aligned_offset = (start // TG_BLOCK) * TG_BLOCK
    skip_head = start - aligned_offset
    bytes_needed = end - start + 1

    sent = 0
    # iter_download streams in TG_BLOCK chunks starting from offset
    async for chunk in client.iter_download(
        loc, offset=aligned_offset, request_size=TG_BLOCK,
        file_size=size, stride=TG_BLOCK,
    ):
        if not chunk:
            break
        if skip_head:
            if skip_head >= len(chunk):
                skip_head -= len(chunk)
                continue
            chunk = chunk[skip_head:]
            skip_head = 0
        if sent + len(chunk) >= bytes_needed:
            yield chunk[: bytes_needed - sent]
            sent = bytes_needed
            break
        yield chunk
        sent += len(chunk)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="TG-Stream", version="1.0.0",
              description="Stream Telegram media with HTTP Range support.")

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
        "developer": "Gourav Rajput",
        "endpoints": ["/watch/{id}", "/pdf/{id}", "/file/{id}", "/info/{id}"],
        "chat": str(CHAT_REF),
    })


@app.get("/info/{msg_id}")
async def info(msg_id: int):
    msg = await get_message(msg_id)
    loc, size, mime, fname = media_info(msg)
    return {"id": msg_id, "filename": fname, "size": size, "mime": mime}


async def _serve(msg_id: int, request: Request, *, force_inline: Optional[str] = None,
                 force_mime: Optional[str] = None):
    msg = await get_message(msg_id)
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
    }

    if request.method == "HEAD":
        headers["Content-Length"] = str(size)
        return Response(status_code=200, headers=headers)

    if range_header:
        start, end = parse_range(range_header, size)
        length = end - start + 1
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        headers["Content-Length"] = str(length)
        return StreamingResponse(
            iter_telegram(loc, size, start, end),
            status_code=206, headers=headers, media_type=mime,
        )

    # No Range -> still stream but in 1MB chunks (don't load to RAM)
    headers["Content-Length"] = str(size)
    return StreamingResponse(
        iter_telegram(loc, size, 0, size - 1),
        status_code=200, headers=headers, media_type=mime,
    )


@app.api_route("/watch/{msg_id}", methods=["GET", "HEAD"])
async def watch(msg_id: int, request: Request):
    return await _serve(msg_id, request, force_inline="inline")


@app.api_route("/pdf/{msg_id}", methods=["GET", "HEAD"])
async def pdf(msg_id: int, request: Request):
    return await _serve(msg_id, request, force_inline="inline",
                        force_mime="application/pdf")


@app.api_route("/file/{msg_id}", methods=["GET", "HEAD"])
async def file(msg_id: int, request: Request):
    return await _serve(msg_id, request, force_inline="inline")


# ---------------------------------------------------------------------------
# Entrypoint  (Railway / Render / PythonAnywhere all run "python app.py" or
# use the Procfile.)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # If no SESSION env-var, print the current StringSession so the user can
    # copy it to the host's environment.
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
