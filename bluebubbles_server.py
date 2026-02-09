#!/usr/bin/env python3
"""BlueBubbles MCP Server - Full-featured interface for the BlueBubbles iMessage API.

A comprehensive MCP server providing 55 tools for interacting with BlueBubbles,
covering messaging, chats, contacts, handles, attachments, FindMy, scheduled
messages, server management, and Mac control.

Configuration via environment variables:
    BLUEBUBBLES_URL      - Server URL (default: http://localhost:1234)
    BLUEBUBBLES_PASSWORD - API password (required)
"""

import os
import re
import sys
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import quote, urlencode
from contextlib import asynccontextmanager

import httpx
from mcp.server.fastmcp import FastMCP, Image

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("bluebubbles-mcp")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BLUEBUBBLES_URL = os.environ.get("BLUEBUBBLES_URL", "http://localhost:1234").rstrip("/")
BLUEBUBBLES_PASSWORD = os.environ.get("BLUEBUBBLES_PASSWORD", "")

if not BLUEBUBBLES_PASSWORD:
    logger.warning("BLUEBUBBLES_PASSWORD not set - API calls will fail")

# ---------------------------------------------------------------------------
# Module-level HTTP client (set during lifespan)
# ---------------------------------------------------------------------------
_http_client: Optional[httpx.AsyncClient] = None


@asynccontextmanager
async def lifespan(server: FastMCP):
    """Manage the httpx.AsyncClient lifecycle and preload contact cache."""
    global _http_client
    async with httpx.AsyncClient(
        base_url=BLUEBUBBLES_URL,
        timeout=httpx.Timeout(30.0, connect=10.0),
    ) as client:
        _http_client = client
        logger.info("BlueBubbles MCP server started (base_url=%s)", BLUEBUBBLES_URL)
        await _ensure_contact_cache()
        yield
    _http_client = None
    logger.info("BlueBubbles MCP server stopped")


mcp = FastMCP("bluebubbles", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Contact name cache
# ---------------------------------------------------------------------------

_contact_cache: dict[str, str] = {}  # normalized address -> display name
_contact_cache_loaded: bool = False


def _normalize_phone(raw: str) -> str:
    """Normalize a phone number to E.164-ish format for cache lookups.

    Strips everything except digits and a leading '+'.  For 10-digit US
    numbers, prepends '+1'.  For 11-digit numbers starting with '1', prepends
    '+'.  Returns lowercase for email addresses.
    """
    if not raw:
        return ""
    # Email addresses -- just lowercase
    if "@" in raw:
        return raw.strip().lower()
    digits = re.sub(r"[^\d]", "", raw)
    if len(digits) == 10:
        return f"+1{digits}"
    if len(digits) == 11 and digits.startswith("1"):
        return f"+{digits}"
    if raw.startswith("+"):
        return f"+{digits}"
    return f"+{digits}" if digits else raw.strip()


async def _ensure_contact_cache() -> None:
    """Populate the contact name cache on first use."""
    global _contact_cache_loaded
    if _contact_cache_loaded:
        return
    try:
        resp = await api_request("contact")
        contacts = resp.get("data") or []
        for c in contacts:
            name = (
                c.get("displayName")
                or f"{c.get('firstName', '')} {c.get('lastName', '')}".strip()
            )
            if not name:
                continue
            # Clean up backslash-escaped commas from macOS Contacts
            name = name.replace("\\,", ",").replace("\\", "")
            for phone in c.get("phoneNumbers") or []:
                addr = phone.get("address", "")
                if addr:
                    _contact_cache[_normalize_phone(addr)] = name
            for email in c.get("emails") or []:
                addr = email.get("address", "")
                if addr:
                    _contact_cache[_normalize_phone(addr)] = name
        _contact_cache_loaded = True
        logger.info("Contact cache loaded: %d address->name mappings", len(_contact_cache))
    except Exception as exc:
        logger.warning("Failed to load contact cache: %s", exc)
        _contact_cache_loaded = True  # Don't retry on every call


def _resolve_name(address: str) -> str:
    """Look up a display name for an address.  Returns the original address
    if no contact match is found."""
    if not address:
        return "Unknown"
    normalized = _normalize_phone(address)
    name = _contact_cache.get(normalized)
    if name:
        return name
    return address


# ---------------------------------------------------------------------------
# Core API helpers
# ---------------------------------------------------------------------------


def _auth_params() -> dict:
    """Return the password query parameter dict."""
    return {"password": BLUEBUBBLES_PASSWORD}


async def api_request(
    endpoint: str,
    method: str = "GET",
    data: Optional[dict] = None,
    params: Optional[dict] = None,
) -> dict:
    """Execute an API request against BlueBubbles.

    Args:
        endpoint: Path relative to /api/v1/ (no leading slash needed).
        method: HTTP method (GET, POST, PUT, DELETE).
        data: JSON body for POST/PUT requests.
        params: Additional query parameters.

    Returns:
        Parsed JSON response dict.

    Raises:
        RuntimeError: If the HTTP client is not initialised.
        httpx.HTTPStatusError: On 4xx/5xx responses.
    """
    if _http_client is None:
        raise RuntimeError("HTTP client not initialised - server may not be running")

    url = f"/api/v1/{endpoint.lstrip('/')}"
    query = _auth_params()
    if params:
        query.update(params)

    response = await _http_client.request(
        method=method.upper(),
        url=url,
        params=query,
        json=data if method.upper() in ("POST", "PUT", "DELETE") and data is not None else None,
    )
    response.raise_for_status()
    return response.json()


def _extract_data(response: dict):
    """Pull the 'data' key from a standard BlueBubbles API response."""
    return response.get("data")


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _ts_to_str(timestamp) -> str:
    """Convert a BlueBubbles millisecond timestamp to a human-readable string.

    Handles None, 0, and already-string values gracefully.
    """
    if timestamp is None or timestamp == 0:
        return ""
    try:
        if isinstance(timestamp, (int, float)):
            dt = datetime.fromtimestamp(timestamp / 1000.0, tz=timezone.utc)
            return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        return str(timestamp)
    except (OSError, ValueError, OverflowError):
        return str(timestamp)


_REACTION_EMOJI: dict = {
    # String names (newer BlueBubbles)
    "love": "❤️", "like": "👍", "dislike": "👎",
    "laugh": "😂", "emphasize": "‼️", "question": "❓",
    # Numeric types
    2000: "❤️", 2001: "👍", 2002: "👎",
    2003: "😂", 2004: "‼️", 2005: "❓",
    # Remove reactions (string)
    "-love": "-❤️", "-like": "-👍", "-dislike": "-👎",
    "-laugh": "-😂", "-emphasize": "-‼️", "-question": "-❓",
    # Remove reactions (numeric)
    3000: "-❤️", 3001: "-👍", 3002: "-👎",
    3003: "-😂", 3004: "-‼️", 3005: "-❓",
    # Sticker
    4000: "🏷️",
    # 2006 = iOS 17+ any-emoji reaction (actual emoji extracted from text field)
}

# Regex to extract the emoji from iOS 17+ reaction text like "Reacted 🔥 to ..."
_IOS17_REACTION_RE = re.compile(r"Reacted\s+(.+?)\s+to\s+")


def _resolve_reaction_emoji(assoc_type, msg_text: str) -> str:
    """Resolve a reaction type to its display emoji."""
    # Check static map first
    static = _REACTION_EMOJI.get(assoc_type)
    if static:
        return static
    # iOS 17+ any-emoji reactions (type "2006" or 2006): extract from text
    if str(assoc_type) == "2006" and msg_text:
        match = _IOS17_REACTION_RE.search(msg_text)
        if match:
            return match.group(1).strip()
    return str(assoc_type)


def _strip_part_prefix(guid: str) -> str:
    """Strip 'p:N/' prefix from associated message GUIDs."""
    return re.sub(r"^p:\d+/", "", guid)


def _sender_name(msg: dict) -> str:
    """Extract resolved sender name from a message."""
    if msg.get("isFromMe", False):
        return "Me"
    handle = msg.get("handle") or {}
    return _resolve_name(handle.get("address") or "Unknown")


def format_message(msg: dict) -> str:
    """Format a single message object into a readable string."""
    ts = _ts_to_str(msg.get("dateCreated"))
    sender = _sender_name(msg)

    assoc_type = msg.get("associatedMessageType")
    if assoc_type:
        emoji = _resolve_reaction_emoji(assoc_type, (msg.get("text") or ""))
        target_guid = msg.get("associatedMessageGuid", "")
        return f"[{ts}] {sender} reacted {emoji} to message {target_guid}"

    text = (msg.get("text") or "").strip()
    subject = (msg.get("subject") or "").strip()
    if not text:
        attributed = msg.get("attributedBody")
        if attributed and isinstance(attributed, list):
            parts = []
            for part in attributed:
                content = part.get("content") or part.get("string") or ""
                if content:
                    parts.append(content.strip())
            text = " ".join(parts) if parts else "[No text content]"
        else:
            text = "[No text content]"

    prefix = f"[{ts}] {sender}"
    if subject:
        line = f"{prefix} (subject: {subject}): {text}"
    else:
        line = f"{prefix}: {text}"

    attachments = msg.get("attachments") or []
    if attachments:
        for att in attachments:
            mime = att.get("mimeType", "unknown")
            fname = att.get("transferName", "")
            att_guid = att.get("guid", "")
            size = att.get("totalBytes", 0)
            size_str = f" {size:,}B" if size else ""
            line += f"\n    📎 {fname} ({mime}{size_str}) [guid: {att_guid}]"

    return line


def _format_messages_grouped(messages: list) -> list[str]:
    """Format messages with reactions grouped onto their parent messages.

    Instead of showing reactions as separate messages, attaches them as
    annotations on the original message they reference.  Orphan reactions
    (whose parent isn't in the current batch) are shown with the quoted
    text from the reaction message itself.
    """
    # Index all messages by GUID and separate reactions from regular messages
    msg_by_guid: dict[str, dict] = {}
    # Store (sender, emoji, reaction_text) so orphans can show context
    reactions_by_target: dict[str, list[tuple[str, str, str]]] = {}
    regular: list[dict] = []

    for msg in messages:
        guid = msg.get("guid", "")
        if guid:
            msg_by_guid[guid] = msg

        assoc_type = msg.get("associatedMessageType")
        if assoc_type:
            raw_target = msg.get("associatedMessageGuid", "")
            target = _strip_part_prefix(raw_target)
            reaction_text = (msg.get("text") or "").strip()
            emoji = _resolve_reaction_emoji(assoc_type, reaction_text)
            sender = _sender_name(msg)
            reactions_by_target.setdefault(target, []).append(
                (sender, emoji, reaction_text)
            )
        else:
            regular.append(msg)

    def _format_reaction_block(
        reactions: list[tuple[str, str, str]],
    ) -> str:
        by_emoji: dict[str, list[str]] = {}
        for sender, emoji, _ in reactions:
            by_emoji.setdefault(emoji, []).append(sender)
        parts = [
            f"{emoji} {', '.join(senders)}"
            for emoji, senders in by_emoji.items()
        ]
        return f"    [{' | '.join(parts)}]"

    lines: list[str] = []
    for msg in regular:
        line = format_message(msg)

        guid = msg.get("guid", "")
        reactions = reactions_by_target.pop(guid, [])
        if reactions:
            line += "\n" + _format_reaction_block(reactions)

        lines.append(line)

    # Orphan reactions: target message not in this batch (pagination edge case)
    for target_guid, reactions in reactions_by_target.items():
        # Try to extract quoted text from the reaction's own text field
        # e.g. 'Liked "Like this for option 2"' -> "Like this for option 2"
        snippet = ""
        for _, _, rtext in reactions:
            if rtext:
                match = re.search(r'["""](.+?)["""]', rtext)
                if match:
                    snippet = match.group(1)
                    if len(snippet) > 60:
                        snippet = snippet[:57] + "..."
                    break
        context = f' "{snippet}"' if snippet else ""
        lines.append(f"  (reaction to older message{context})")
        lines.append(_format_reaction_block(reactions))

    return lines


def format_message_detail(msg: dict) -> str:
    """Format a message with extended detail fields."""
    lines = [format_message(msg)]
    guid = msg.get("guid", "")
    lines.append(f"  GUID: {guid}")

    delivered = _ts_to_str(msg.get("dateDelivered"))
    read = _ts_to_str(msg.get("dateRead"))
    if delivered:
        lines.append(f"  Delivered: {delivered}")
    if read:
        lines.append(f"  Read: {read}")

    if msg.get("isDelivered"):
        lines.append("  Status: Delivered")
    if msg.get("dateEdited"):
        lines.append(f"  Edited: {_ts_to_str(msg.get('dateEdited'))}")
    if msg.get("dateRetracted"):
        lines.append(f"  Retracted: {_ts_to_str(msg.get('dateRetracted'))}")

    attachments = msg.get("attachments") or []
    if attachments:
        lines.append(f"  Attachments: {len(attachments)}")
        for att in attachments:
            mime = att.get("mimeType", "unknown")
            fname = att.get("transferName", "")
            guid = att.get("guid", "")
            size = att.get("totalBytes", 0)
            size_str = f" {size:,}B" if size else ""
            lines.append(f"    - {fname} ({mime}{size_str}) [guid: {guid}]")

    chats = msg.get("chats") or []
    if chats:
        chat_ids = [c.get("guid", "") for c in chats]
        lines.append(f"  Chats: {', '.join(chat_ids)}")

    if msg.get("threadOriginatorGuid"):
        lines.append(f"  Reply to thread: {msg['threadOriginatorGuid']}")
    if msg.get("expressiveSendStyleId"):
        lines.append(f"  Send effect: {msg['expressiveSendStyleId']}")

    error = msg.get("error", 0)
    if error:
        lines.append(f"  Error code: {error}")

    return "\n".join(lines)


def format_chat(chat: dict) -> str:
    """Format a chat object into a readable string."""
    display = chat.get("displayName") or ""
    identifier = chat.get("chatIdentifier") or ""
    guid = chat.get("guid") or ""
    style = chat.get("style", 0)
    chat_type = "Group" if style == 43 else "Individual" if style == 45 else f"Style {style}"

    participants = chat.get("participants") or []
    participant_count = len(participants)

    # For individual chats, resolve the identifier to a contact name
    name = display
    if not name and style == 45 and identifier:
        name = _resolve_name(identifier)
    if not name:
        name = identifier or guid
    parts = [f"{name} [{chat_type}, {participant_count} participant(s)]"]
    parts.append(f"  GUID: {guid}")

    if chat.get("isArchived"):
        parts.append("  [Archived]")
    if chat.get("isFiltered"):
        parts.append("  [Filtered]")

    if participants:
        resolved = [_resolve_name(p.get("address", "?")) for p in participants]
        parts.append(f"  Participants: {', '.join(resolved)}")

    last_msg = chat.get("lastMessage")
    if last_msg and isinstance(last_msg, dict):
        parts.append(f"  Last message: {format_message(last_msg)}")

    return "\n".join(parts)


def format_contact(contact: dict) -> str:
    """Format a contact object into a readable string."""
    first = contact.get("firstName") or ""
    last = contact.get("lastName") or ""
    display = contact.get("displayName") or ""
    nickname = contact.get("nickname") or ""

    name = display or f"{first} {last}".strip() or "Unknown"
    parts = [name]
    if nickname:
        parts[0] += f' ("{nickname}")'

    phones = contact.get("phoneNumbers") or []
    for p in phones:
        parts.append(f"  Phone: {p.get('address', '?')}")

    emails = contact.get("emails") or []
    for e in emails:
        parts.append(f"  Email: {e.get('address', '?')}")

    birthday = contact.get("birthday")
    if birthday:
        parts.append(f"  Birthday: {birthday}")

    source = contact.get("sourceType")
    if source:
        parts.append(f"  Source: {source}")

    return "\n".join(parts)


def format_handle(handle: dict) -> str:
    """Format a handle object into a readable string."""
    addr = handle.get("address", "?")
    service = handle.get("service", "?")
    country = handle.get("country") or ""
    row_id = handle.get("originalROWID", "")
    name = _resolve_name(addr)
    header = f"{name} ({addr}, {service})" if name != addr else f"{addr} (service: {service})"
    parts = [header]
    if country:
        parts.append(f"  Country: {country}")
    if row_id:
        parts.append(f"  Row ID: {row_id}")

    uncanon = handle.get("uncanonicalizedId")
    if uncanon and uncanon != addr:
        parts.append(f"  Uncanonicalized: {uncanon}")

    return "\n".join(parts)


def _safe_call(func):
    """Decorator to wrap tool functions with consistent error handling."""
    import functools

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            try:
                body = exc.response.json()
                detail = body.get("message", str(exc))
            except Exception:
                detail = str(exc)
            return f"Error: HTTP {status} - {detail}"
        except httpx.RequestError as exc:
            return f"Error: Could not reach BlueBubbles server - {exc}"
        except RuntimeError as exc:
            return f"Error: {exc}"
        except Exception as exc:
            logger.exception("Unexpected error in %s", func.__name__)
            return f"Error: {exc}"

    return wrapper


# ===================================================================
# SERVER TOOLS (1-9)
# ===================================================================


@mcp.tool()
@_safe_call
async def get_server_info() -> str:
    """Get BlueBubbles server information including OS version, server version,
    Private API status, proxy service, iCloud/iMessage detection, helper
    connection status, and local IP addresses."""
    resp = await api_request("server/info")
    info = _extract_data(resp) or {}
    lines = ["BlueBubbles Server Information", "=" * 40]
    lines.append(f"Computer ID:       {info.get('computer_id', 'N/A')}")
    lines.append(f"OS Version:        {info.get('os_version', 'N/A')}")
    lines.append(f"Server Version:    {info.get('server_version', 'N/A')}")
    lines.append(f"Private API:       {'Enabled' if info.get('private_api') else 'Disabled'}")
    lines.append(f"Proxy Service:     {info.get('proxy_service', 'N/A')}")
    lines.append(f"Helper Connected:  {'Yes' if info.get('helper_connected') else 'No'}")
    lines.append(f"Detected iCloud:   {info.get('detected_icloud', 'N/A')}")
    lines.append(f"Detected iMessage: {info.get('detected_imessage', 'N/A')}")
    lines.append(f"macOS Time Sync:   {info.get('macos_time_sync', 'N/A')}")

    ipv4s = info.get("local_ipv4s") or []
    if ipv4s:
        lines.append(f"Local IPv4:        {', '.join(ipv4s)}")
    ipv6s = info.get("local_ipv6s") or []
    if ipv6s:
        lines.append(f"Local IPv6:        {', '.join(str(ip) for ip in ipv6s)}")

    return "\n".join(lines)


@mcp.tool()
@_safe_call
async def get_server_logs() -> str:
    """Retrieve the BlueBubbles server log output. Returns the raw log text
    which can be used for debugging server-side issues."""
    resp = await api_request("server/logs")
    data = _extract_data(resp)
    if isinstance(data, str):
        return f"Server Logs\n{'=' * 40}\n{data}"
    return f"Server Logs\n{'=' * 40}\n{data}"


@mcp.tool()
@_safe_call
async def get_server_alerts() -> str:
    """Get all server alerts/notifications. Each alert has a type, value,
    read status, and creation timestamp."""
    resp = await api_request("server/alert")
    alerts = _extract_data(resp) or []
    if not alerts:
        return "No server alerts."
    lines = [f"Server Alerts ({len(alerts)})", "=" * 40]
    for a in alerts:
        ts = _ts_to_str(a.get("created"))
        read_status = "Read" if a.get("isRead") else "Unread"
        lines.append(f"[{ts}] [{read_status}] {a.get('type', 'unknown')}: {a.get('value', '')}")
    return "\n".join(lines)


@mcp.tool()
@_safe_call
async def mark_alerts_read(alert_ids: list[int]) -> str:
    """Mark specific server alerts as read.

    Args:
        alert_ids: List of alert IDs to mark as read.
    """
    resp = await api_request("server/alert/read", method="POST", data={"ids": alert_ids})
    return f"Marked {len(alert_ids)} alert(s) as read."


@mcp.tool()
@_safe_call
async def get_statistics() -> str:
    """Get total counts of handles, messages, chats, and attachments stored
    on the BlueBubbles server."""
    resp = await api_request("server/statistics/totals")
    stats = _extract_data(resp) or {}
    lines = ["Server Statistics", "=" * 40]
    lines.append(f"Handles:     {stats.get('handles', 0):,}")
    lines.append(f"Messages:    {stats.get('messages', 0):,}")
    lines.append(f"Chats:       {stats.get('chats', 0):,}")
    lines.append(f"Attachments: {stats.get('attachments', 0):,}")
    return "\n".join(lines)


@mcp.tool()
@_safe_call
async def get_media_totals() -> str:
    """Get overall media totals (images, videos, locations) across all chats."""
    resp = await api_request("server/statistics/media")
    media = _extract_data(resp) or {}
    lines = ["Media Totals", "=" * 40]
    lines.append(f"Images:    {media.get('images', 0):,}")
    lines.append(f"Videos:    {media.get('videos', 0):,}")
    lines.append(f"Locations: {media.get('locations', 0):,}")
    return "\n".join(lines)


@mcp.tool()
@_safe_call
async def get_media_totals_by_chat() -> str:
    """Get media totals (images, videos, locations) broken down by chat.
    Useful for finding which conversations have the most media."""
    resp = await api_request("server/statistics/media/chat")
    chats = _extract_data(resp) or []
    if not chats:
        return "No media statistics by chat available."
    lines = [f"Media Totals by Chat ({len(chats)} chats)", "=" * 40]
    for c in chats:
        name = c.get("groupName") or c.get("chatGuid", "Unknown")
        totals = c.get("totals") or {}
        imgs = totals.get("images", 0)
        vids = totals.get("videos", 0)
        locs = totals.get("locations", 0)
        lines.append(f"{name}: {imgs} images, {vids} videos, {locs} locations")
    return "\n".join(lines)


@mcp.tool()
@_safe_call
async def check_for_update() -> str:
    """Check if a BlueBubbles server update is available. Returns the current
    version and whether an update exists."""
    resp = await api_request("server/update/check")
    data = _extract_data(resp) or {}
    available = data.get("available", False)
    current = data.get("current", "unknown")
    lines = ["Update Check", "=" * 40]
    lines.append(f"Current Version: {current}")
    lines.append(f"Update Available: {'Yes' if available else 'No'}")
    metadata = data.get("metadata")
    if metadata and isinstance(metadata, dict):
        for k, v in metadata.items():
            lines.append(f"  {k}: {v}")
    return "\n".join(lines)


@mcp.tool()
@_safe_call
async def ping() -> str:
    """Ping the BlueBubbles server to check connectivity. Returns success
    if the server is reachable and responding."""
    resp = await api_request("ping")
    msg = resp.get("message", "pong")
    return f"Server responded: {msg}"


# ===================================================================
# CHAT TOOLS (10-23)
# ===================================================================


@mcp.tool()
@_safe_call
async def get_chat_count() -> str:
    """Get the total number of chats and a breakdown by service type
    (e.g. iMessage count)."""
    resp = await api_request("chat/count")
    data = _extract_data(resp) or {}
    total = data.get("total", 0)
    breakdown = data.get("breakdown") or {}
    lines = ["Chat Count", "=" * 40]
    lines.append(f"Total: {total:,}")
    for service, count in breakdown.items():
        lines.append(f"  {service}: {count:,}")
    return "\n".join(lines)


@mcp.tool()
@_safe_call
async def get_chat(
    chat_guid: str,
    include: Optional[str] = None,
) -> str:
    """Get detailed information about a specific chat.

    Args:
        chat_guid: The GUID of the chat (e.g. 'iMessage;-;+11234567890').
        include: Comma-separated list of extra data to include.
            Options: 'participants', 'lastMessage'. Example: 'participants,lastMessage'.
    """
    params = {}
    if include:
        params["with"] = include
    resp = await api_request(f"chat/{chat_guid}", params=params)
    chat = _extract_data(resp)
    if not chat:
        return f"Chat not found: {chat_guid}"
    return format_chat(chat)


@mcp.tool()
@_safe_call
async def query_chats(
    limit: int = 25,
    offset: int = 0,
    sort: str = "lastmessage",
    include_participants: bool = False,
    include_last_message: bool = True,
) -> str:
    """Query and list chats with pagination and sorting.

    Args:
        limit: Maximum number of chats to return (default 25).
        offset: Number of chats to skip for pagination (default 0).
        sort: Sort order. Use 'lastmessage' to sort by most recent activity.
        include_participants: Include participant details (default False).
            Use get_chat() for full participant info on a specific chat.
        include_last_message: Whether to include the last message in each chat (default True).
    """
    with_list = []
    if include_participants:
        with_list.append("participants")
    if include_last_message:
        with_list.append("lastMessage")

    # BlueBubbles has a server-side pagination bug where sort=lastmessage
    # with small limits misses the most recently active chats.  Work around
    # this by over-fetching and trimming client-side.
    fetch_limit = max(limit + offset, 500) if sort == "lastmessage" else limit

    body: dict = {
        "limit": fetch_limit,
        "offset": 0 if sort == "lastmessage" else offset,
        "sort": sort,
    }
    if with_list:
        body["with"] = with_list

    resp = await api_request("chat/query", method="POST", data=body)
    chats = _extract_data(resp) or []
    meta = resp.get("metadata") or {}

    # Apply client-side offset + limit to the (correctly sorted) larger result
    if sort == "lastmessage":
        chats = chats[offset: offset + limit]

    total = meta.get("total", "?")
    lines = [
        f"Chats (showing {len(chats)} of {total})",
        "=" * 40,
    ]
    for i, chat in enumerate(chats, start=offset + 1):
        lines.append(f"\n--- Chat {i} ---")
        lines.append(format_chat(chat))
    return "\n".join(lines)


@mcp.tool()
@_safe_call
async def get_chat_messages(
    chat_guid: str,
    limit: int = 25,
    offset: int = 0,
    sort: str = "DESC",
    after: Optional[int] = None,
    before: Optional[int] = None,
    include: Optional[str] = "handle,attachment",
) -> str:
    """Get messages from a specific chat with pagination and time filtering.

    Args:
        chat_guid: The GUID of the chat.
        limit: Maximum number of messages to return (default 25).
        offset: Number of messages to skip (default 0).
        sort: Sort order - 'ASC' for oldest first, 'DESC' for newest first (default 'DESC').
        after: Only return messages after this timestamp (milliseconds since epoch).
        before: Only return messages before this timestamp (milliseconds since epoch).
        include: Comma-separated list of related data to include.
            Default: 'handle,attachment'. Options: handle, chat, attachment.
    """
    with_list = []
    if include:
        with_list = [s.strip() for s in include.split(",") if s.strip()]

    body: dict = {
        "chatGuid": chat_guid,
        "limit": min(limit, 1000),
        "offset": offset,
        "sort": sort,
    }
    if with_list:
        body["with"] = with_list
    if after is not None:
        body["after"] = after
    if before is not None:
        body["before"] = before

    resp = await api_request("message/query", method="POST", data=body)
    messages = _extract_data(resp) or []
    lines = [f"Messages in {chat_guid} ({len(messages)} returned)", "=" * 40]
    lines.extend(_format_messages_grouped(messages))
    return "\n".join(lines)


@mcp.tool()
@_safe_call
async def create_chat(
    addresses: list[str],
    message: Optional[str] = None,
) -> str:
    """Create a new chat with one or more participants, optionally sending an
    initial message.

    Args:
        addresses: List of phone numbers or email addresses for the chat participants.
            Example: ['+11234567890'] or ['+11234567890', 'user@example.com'].
        message: Optional initial message to send when creating the chat.
    """
    body: dict = {"addresses": addresses}
    if message:
        body["message"] = message
    resp = await api_request("chat/new", method="POST", data=body)
    chat = _extract_data(resp)
    if chat and isinstance(chat, dict):
        return f"Chat created successfully.\n{format_chat(chat)}"
    return f"Chat creation response: {resp.get('message', 'OK')}"


@mcp.tool()
@_safe_call
async def mark_chat_read(chat_guid: str) -> str:
    """Mark all messages in a chat as read.

    Args:
        chat_guid: The GUID of the chat to mark as read.
    """
    resp = await api_request(f"chat/{chat_guid}/read", method="POST")
    return f"Chat {chat_guid} marked as read."


@mcp.tool()
@_safe_call
async def mark_chat_unread(chat_guid: str) -> str:
    """Mark a chat as unread.

    Args:
        chat_guid: The GUID of the chat to mark as unread.
    """
    resp = await api_request(f"chat/{chat_guid}/unread", method="POST")
    return f"Chat {chat_guid} marked as unread."


@mcp.tool()
@_safe_call
async def delete_chat(chat_guid: str) -> str:
    """Delete a chat conversation.

    Args:
        chat_guid: The GUID of the chat to delete.
    """
    resp = await api_request(f"chat/{chat_guid}", method="DELETE")
    return f"Chat {chat_guid} deleted."


@mcp.tool()
@_safe_call
async def update_chat(chat_guid: str, display_name: str) -> str:
    """Update a chat's display name (for group chats).

    Args:
        chat_guid: The GUID of the chat to update.
        display_name: The new display name for the chat.
    """
    resp = await api_request(
        f"chat/{chat_guid}",
        method="PUT",
        data={"displayName": display_name},
    )
    return f"Chat {chat_guid} display name updated to '{display_name}'."


@mcp.tool()
@_safe_call
async def add_participant(chat_guid: str, address: str) -> str:
    """Add a participant to a group chat.

    Args:
        chat_guid: The GUID of the group chat.
        address: Phone number or email of the participant to add.
    """
    resp = await api_request(
        f"chat/{chat_guid}/participant",
        method="POST",
        data={"address": address},
    )
    return f"Added {address} to chat {chat_guid}."


@mcp.tool()
@_safe_call
async def remove_participant(chat_guid: str, address: str) -> str:
    """Remove a participant from a group chat.

    Args:
        chat_guid: The GUID of the group chat.
        address: Phone number or email of the participant to remove.
    """
    resp = await api_request(
        f"chat/{chat_guid}/participant",
        method="DELETE",
        data={"address": address},
    )
    return f"Removed {address} from chat {chat_guid}."


@mcp.tool()
@_safe_call
async def leave_chat(chat_guid: str) -> str:
    """Leave a group chat.

    Args:
        chat_guid: The GUID of the group chat to leave.
    """
    resp = await api_request(f"chat/{chat_guid}/leave", method="POST")
    return f"Left chat {chat_guid}."


@mcp.tool()
@_safe_call
async def start_typing(chat_guid: str) -> str:
    """Send a typing indicator to a chat (shows the '...' bubble on the
    recipient's device). Requires Private API.

    Args:
        chat_guid: The GUID of the chat to show typing in.
    """
    resp = await api_request(f"chat/{chat_guid}/typing", method="POST")
    return f"Typing indicator started in {chat_guid}."


@mcp.tool()
@_safe_call
async def stop_typing(chat_guid: str) -> str:
    """Stop the typing indicator in a chat. Requires Private API.

    Args:
        chat_guid: The GUID of the chat to stop typing in.
    """
    resp = await api_request(f"chat/{chat_guid}/typing", method="DELETE")
    return f"Typing indicator stopped in {chat_guid}."


# ===================================================================
# MESSAGE TOOLS (24-34)
# ===================================================================


@mcp.tool()
@_safe_call
async def get_message_count() -> str:
    """Get the total message count with a breakdown by service type
    (iMessage vs SMS)."""
    resp = await api_request("message/count")
    data = _extract_data(resp) or {}
    total = data.get("total", 0)
    breakdown = data.get("breakdown") or {}
    lines = ["Message Count", "=" * 40]
    lines.append(f"Total: {total:,}")
    for service, count in breakdown.items():
        lines.append(f"  {service}: {count:,}")
    return "\n".join(lines)


@mcp.tool()
@_safe_call
async def get_my_message_count() -> str:
    """Get the count of messages sent by you (isFromMe = true)."""
    resp = await api_request("message/count/me")
    data = _extract_data(resp) or {}
    if isinstance(data, (int, float)):
        return f"Your sent message count: {int(data):,}"
    total = data.get("total", data) if isinstance(data, dict) else data
    return f"Your sent message count: {total}"


@mcp.tool()
@_safe_call
async def get_message(
    message_guid: str,
    include: Optional[str] = "chats,handle,attachment",
) -> str:
    """Get a single message by its GUID with full details including handle,
    chat, and attachment information.

    Args:
        message_guid: The GUID of the message to retrieve.
        include: Comma-separated list of related data to include.
            Options: 'chats', 'participants', 'attachment', 'handle'.
            Default: 'chats,handle,attachment'.
    """
    params = {}
    if include:
        params["with"] = include
    resp = await api_request(f"message/{message_guid}", params=params)
    msg = _extract_data(resp)
    if not msg:
        return f"Message not found: {message_guid}"
    return format_message_detail(msg)


@mcp.tool()
@_safe_call
async def query_messages(
    limit: int = 25,
    offset: int = 0,
    chat_guid: Optional[str] = None,
    sort: str = "DESC",
    after: Optional[int] = None,
    before: Optional[int] = None,
    search_text: Optional[str] = None,
    include_chat: bool = True,
    include_handle: bool = True,
    include_attachment: bool = True,
) -> str:
    """Search and query messages with flexible filtering. Use search_text to
    find messages containing specific text. Filter by chat, time range, or
    combine multiple criteria.

    Args:
        limit: Maximum number of messages to return (default 25, max 1000).
        offset: Number of messages to skip for pagination.
        chat_guid: Optional chat GUID to filter messages to a specific conversation.
        sort: Sort order - 'DESC' for newest first, 'ASC' for oldest first.
        after: Only return messages after this timestamp (milliseconds since epoch).
        before: Only return messages before this timestamp (milliseconds since epoch).
        search_text: Search for messages containing this text (case-insensitive LIKE match).
        include_chat: Include chat data with each message (default True).
        include_handle: Include handle/sender data with each message (default True).
        include_attachment: Include attachment data with each message (default True).
    """
    with_list = []
    if include_chat:
        with_list.append("chat")
    if include_handle:
        with_list.append("handle")
    if include_attachment:
        with_list.append("attachment")

    body: dict = {
        "limit": min(limit, 1000),
        "offset": offset,
        "sort": sort,
    }
    if with_list:
        body["with"] = with_list
    if chat_guid:
        body["chatGuid"] = chat_guid
    if after is not None:
        body["after"] = after
    if before is not None:
        body["before"] = before
    if search_text:
        body["where"] = [
            {
                "statement": "message.text LIKE :text",
                "args": {"text": f"%{search_text}%"},
            }
        ]

    resp = await api_request("message/query", method="POST", data=body)
    messages = _extract_data(resp) or []
    meta = resp.get("metadata") or {}

    total = meta.get("total", "?")
    lines = [
        f"Messages (showing {len(messages)} of {total}, offset {offset})",
        "=" * 40,
    ]
    lines.extend(_format_messages_grouped(messages))
    return "\n".join(lines)


@mcp.tool()
@_safe_call
async def find_chat_by_address(
    address: str,
) -> str:
    """Find a chat (DM or group) by phone number or email address.

    Use this to get a chat_guid before sending a message with send_message.
    Much faster than send_message_to_new_chat (which has a 120s timeout).

    Args:
        address: Phone number (e.g. '+12039698060') or email to search for.

    Returns:
        Chat GUID if found, or message indicating no chat exists.
    """
    # Query recent chats
    resp = await api_request("chat/query", method="POST", data={
        "limit": 200,
        "sort": "lastmessage",
    })
    chats = _extract_data(resp) or []

    # Search for matching chat
    normalized_search = _normalize_phone(address)
    for chat in chats:
        style = chat.get("style", 0)
        if style == 45:  # DM
            identifier = chat.get("chatIdentifier", "")
            if identifier == address or _normalize_phone(identifier) == normalized_search:
                guid = chat.get("guid", "")
                display = chat.get("displayName") or _resolve_name(identifier)
                return (
                    f"Found chat with {display}:\n"
                    f"  GUID: {guid}\n"
                    f"  Type: Direct Message\n"
                    f"  Identifier: {identifier}\n\n"
                    f"Use send_message(chat_guid=\"{guid}\", message=\"...\") to send."
                )

    return (
        f"No existing chat found with {address}.\n"
        f"You can use send_message_to_new_chat(addresses=[\"{address}\"], message=\"...\") "
        f"to create one, but be aware it has a 120-second timeout (message will send successfully though)."
    )


@mcp.tool()
@_safe_call
async def send_message(
    chat_guid: str,
    message: str,
    method: str = "apple-script",
    subject: Optional[str] = None,
    effect_id: Optional[str] = None,
    selected_message_guid: Optional[str] = None,
) -> str:
    """Send a text message to an existing chat. FAST (responds in <1 second).

    PREFERRED METHOD for sending messages. Use query_chats to find the chat_guid
    for a contact, then use this tool to send. This is 100x faster than
    send_message_to_new_chat which has a 120-second timeout bug.

    Example workflow:
    1. query_chats() to find the chat with your contact
    2. Copy the chat GUID from results
    3. send_message(chat_guid="...", message="...")

    Args:
        chat_guid: The GUID of the chat to send the message to.
            Find via query_chats or get_chat_messages results.
        message: The text message to send.
        method: Send method - 'apple-script' (default, reliable) or 'private-api' (more features).
        subject: Optional subject line for the message.
        effect_id: Optional iMessage effect/animation ID (e.g. 'com.apple.MobileSMS.expressivesend.impact').
        selected_message_guid: Optional GUID of a message to reply to inline.
    """
    body: dict = {
        "chatGuid": chat_guid,
        "tempGuid": str(uuid.uuid4()),
        "message": message,
        "method": method,
    }
    if subject:
        body["subject"] = subject
    if effect_id:
        body["effectId"] = effect_id
    if selected_message_guid:
        body["selectedMessageGuid"] = selected_message_guid

    resp = await api_request("message/text", method="POST", data=body)
    status = resp.get("status", 0)
    if status == 200:
        msg_data = _extract_data(resp)
        if msg_data and isinstance(msg_data, dict):
            return f"Message sent successfully.\n{format_message(msg_data)}"
        return f"Message sent successfully to {chat_guid}."
    return f"Send may have failed (status {status}): {resp.get('message', 'Unknown')}"


@mcp.tool()
@_safe_call
async def send_message_to_new_chat(
    addresses: list[str],
    message: str,
) -> str:
    """Send a message by phone number(s) or email(s). SLOW (120s timeout) - prefer send_message.

    WARNING: This tool has a 120-second timeout bug in BlueBubbles when creating
    new chats. Messages DO send successfully, but the API waits 2 minutes before
    responding. For better performance:

    RECOMMENDED: Use query_chats to find existing chat, then send_message with that GUID.
    Example:
      1. query_chats() -> find chat with contact
      2. send_message(chat_guid="...", message="...") -> instant response

    This tool automatically searches for existing chats first to avoid the timeout.
    Only creates new chats as a last resort.

    Args:
        addresses: List of phone numbers or emails to send to.
            Example: ['+11234567890'] for single recipient, ['+1111', '+1222'] for group.
        message: The text message to send.
    """
    # Try to find existing chat first (much faster than /chat/new)
    if len(addresses) == 1:
        # Single recipient: search for existing DM
        addr = addresses[0]
        search_resp = await api_request("chat/query", method="POST", data={
            "limit": 50,
            "sort": "lastmessage",
        })
        chats = _extract_data(search_resp) or []
        # Look for chat with matching identifier (DM style = 45)
        for chat in chats:
            if chat.get("style") == 45:
                identifier = chat.get("chatIdentifier", "")
                # Match if identifiers are same (exact or normalized phone)
                if identifier == addr or _normalize_phone(identifier) == _normalize_phone(addr):
                    chat_guid = chat.get("guid", "")
                    if chat_guid:
                        # Found existing chat - use fast send_message path
                        return await send_message(chat_guid, message)

    # No existing chat found - fall back to slow /chat/new endpoint
    # This will take 120 seconds to timeout but the message WILL send successfully
    body = {
        "addresses": addresses,
        "message": message,
    }
    resp = await api_request("chat/new", method="POST", data=body)
    status = resp.get("status", 0)
    if status == 200:
        return f"Message sent to {', '.join(addresses)} (via slow /chat/new endpoint)."
    return f"Send may have failed (status {status}): {resp.get('message', 'Unknown')}"


@mcp.tool()
@_safe_call
async def send_reaction(
    chat_guid: str,
    selected_message_guid: str,
    reaction: str,
    selected_message_text: Optional[str] = None,
    part_index: int = 0,
) -> str:
    """Send a tapback/reaction to a message in a chat.

    Args:
        chat_guid: The GUID of the chat containing the message.
        selected_message_guid: The GUID of the message to react to.
        reaction: The reaction type. One of: 'love', 'like', 'dislike', 'laugh', 'emphasize', 'question'.
        selected_message_text: The text of the message being reacted to (helps with delivery).
        part_index: The part index of the message to react to (default 0, for multi-part messages).
    """
    body: dict = {
        "chatGuid": chat_guid,
        "selectedMessageGuid": selected_message_guid,
        "reaction": reaction,
        "partIndex": part_index,
    }
    if selected_message_text:
        body["selectedMessageText"] = selected_message_text

    resp = await api_request("message/react", method="POST", data=body)
    return f"Reaction '{reaction}' sent to message {selected_message_guid}."


@mcp.tool()
@_safe_call
async def edit_message(
    message_guid: str,
    edited_message: str,
    backwards_compatibility_message: Optional[str] = None,
    part_index: int = 0,
) -> str:
    """Edit a previously sent message. Requires Private API and iOS 16+.

    Args:
        message_guid: The GUID of the message to edit.
        edited_message: The new text for the message.
        backwards_compatibility_message: Fallback text shown to recipients on older OS versions.
        part_index: The part index to edit (default 0).
    """
    body: dict = {
        "editedMessage": edited_message,
        "partIndex": part_index,
    }
    if backwards_compatibility_message:
        body["backwardsCompatibilityMessage"] = backwards_compatibility_message

    resp = await api_request(f"message/{message_guid}/edit", method="POST", data=body)
    return f"Message {message_guid} edited successfully."


@mcp.tool()
@_safe_call
async def unsend_message(
    message_guid: str,
    part_index: int = 0,
) -> str:
    """Unsend/retract a previously sent message. Requires Private API and iOS 16+.
    The message will be removed from the recipient's device.

    Args:
        message_guid: The GUID of the message to unsend.
        part_index: The part index to unsend (default 0).
    """
    resp = await api_request(
        f"message/{message_guid}/unsend",
        method="POST",
        data={"partIndex": part_index},
    )
    return f"Message {message_guid} unsent."


@mcp.tool()
@_safe_call
async def get_embedded_media(message_guid: str) -> str:
    """Get embedded media information for a message (e.g. rich links, shared
    content). Returns metadata about any embedded content in the message.

    Args:
        message_guid: The GUID of the message to get embedded media for.
    """
    resp = await api_request(f"message/{message_guid}/embedded-media")
    data = _extract_data(resp)
    if not data:
        return f"No embedded media found for message {message_guid}."
    if isinstance(data, dict):
        lines = [f"Embedded media for {message_guid}:", "=" * 40]
        for k, v in data.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)
    return f"Embedded media for {message_guid}:\n{data}"


@mcp.tool()
@_safe_call
async def notify_silenced_message(message_guid: str) -> str:
    """Send a notification for a message that was silenced/delivered quietly.
    This triggers the notification on the recipient's device.

    Args:
        message_guid: The GUID of the silenced message to notify about.
    """
    resp = await api_request(f"message/{message_guid}/notify", method="POST")
    return f"Notification sent for silenced message {message_guid}."


# ===================================================================
# SCHEDULED MESSAGE TOOLS (35-39)
# ===================================================================


@mcp.tool()
@_safe_call
async def get_scheduled_messages() -> str:
    """Get all scheduled messages. Returns a list of messages that are
    queued for future delivery."""
    resp = await api_request("message/schedule")
    messages = _extract_data(resp) or []
    if not messages:
        return "No scheduled messages."
    lines = [f"Scheduled Messages ({len(messages)})", "=" * 40]
    for msg in messages if isinstance(messages, list) else [messages]:
        msg_id = msg.get("id", "?")
        sched_for = _ts_to_str(msg.get("scheduledFor"))
        payload = msg.get("payload") or {}
        chat = payload.get("chatGuid", "?")
        text = payload.get("message", "")
        status = msg.get("status", "?")
        lines.append(f"  ID: {msg_id} | Chat: {chat} | Scheduled: {sched_for} | Status: {status}")
        if text:
            lines.append(f"    Message: {text}")
    return "\n".join(lines)


@mcp.tool()
@_safe_call
async def get_scheduled_message(schedule_id: int) -> str:
    """Get details of a specific scheduled message by its ID.

    Args:
        schedule_id: The ID of the scheduled message to retrieve.
    """
    resp = await api_request(f"message/schedule/{schedule_id}")
    msg = _extract_data(resp)
    if not msg:
        return f"Scheduled message {schedule_id} not found."
    msg_id = msg.get("id", "?")
    sched_for = _ts_to_str(msg.get("scheduledFor"))
    payload = msg.get("payload") or {}
    schedule = msg.get("schedule") or {}
    lines = [f"Scheduled Message #{msg_id}", "=" * 40]
    lines.append(f"Type: {msg.get('type', '?')}")
    lines.append(f"Status: {msg.get('status', '?')}")
    lines.append(f"Scheduled For: {sched_for}")
    lines.append(f"Schedule Type: {schedule.get('type', '?')}")
    lines.append(f"Chat: {payload.get('chatGuid', '?')}")
    lines.append(f"Message: {payload.get('message', '')}")
    lines.append(f"Method: {payload.get('method', '?')}")
    return "\n".join(lines)


@mcp.tool()
@_safe_call
async def create_scheduled_message(
    chat_guid: str,
    message: str,
    scheduled_for: int,
    schedule_type: str = "once",
    method: str = "private-api",
) -> str:
    """Create a new scheduled message to be sent at a future time.

    Args:
        chat_guid: The GUID of the chat to send the message to.
        message: The text message to schedule.
        scheduled_for: When to send the message, as a millisecond timestamp (epoch).
        schedule_type: Schedule frequency - 'once' for a single send (default).
        method: Send method - 'private-api' (default) or 'apple-script'.
    """
    body = {
        "type": "send-message",
        "payload": {
            "chatGuid": chat_guid,
            "message": message,
            "method": method,
        },
        "scheduledFor": scheduled_for,
        "schedule": {
            "type": schedule_type,
        },
    }
    resp = await api_request("message/schedule", method="POST", data=body)
    data = _extract_data(resp)
    msg_id = data.get("id", "?") if isinstance(data, dict) else "?"
    ts_str = _ts_to_str(scheduled_for)
    return f"Scheduled message created (ID: {msg_id}). Will send at {ts_str} to {chat_guid}."


@mcp.tool()
@_safe_call
async def update_scheduled_message(
    schedule_id: int,
    chat_guid: Optional[str] = None,
    message: Optional[str] = None,
    scheduled_for: Optional[int] = None,
    schedule_type: Optional[str] = None,
    method: Optional[str] = None,
) -> str:
    """Update an existing scheduled message.

    Args:
        schedule_id: The ID of the scheduled message to update.
        chat_guid: New chat GUID (optional, keeps current if not provided).
        message: New message text (optional).
        scheduled_for: New scheduled time as millisecond timestamp (optional).
        schedule_type: New schedule type, e.g. 'once' (optional).
        method: New send method (optional).
    """
    body: dict = {"type": "send-message"}
    payload: dict = {}
    if chat_guid:
        payload["chatGuid"] = chat_guid
    if message:
        payload["message"] = message
    if method:
        payload["method"] = method
    if payload:
        body["payload"] = payload
    if scheduled_for is not None:
        body["scheduledFor"] = scheduled_for
    if schedule_type:
        body["schedule"] = {"type": schedule_type}

    resp = await api_request(f"message/schedule/{schedule_id}", method="PUT", data=body)
    return f"Scheduled message {schedule_id} updated."


@mcp.tool()
@_safe_call
async def delete_scheduled_message(schedule_id: int) -> str:
    """Delete a scheduled message so it will not be sent.

    Args:
        schedule_id: The ID of the scheduled message to delete.
    """
    resp = await api_request(f"message/schedule/{schedule_id}", method="DELETE")
    return f"Scheduled message {schedule_id} deleted."


# ===================================================================
# HANDLE TOOLS (40-45)
# ===================================================================


@mcp.tool()
@_safe_call
async def get_handle_count() -> str:
    """Get the total number of handles (unique contacts/phone numbers/emails)
    in the Messages database."""
    resp = await api_request("handle/count")
    data = _extract_data(resp) or {}
    if isinstance(data, (int, float)):
        return f"Total handles: {int(data):,}"
    total = data.get("total", data) if isinstance(data, dict) else data
    return f"Total handles: {total}"


@mcp.tool()
@_safe_call
async def get_handle(address: str) -> str:
    """Get information about a specific handle (phone number or email).

    Args:
        address: The phone number or email address to look up.
    """
    resp = await api_request(f"handle/{address}")
    handle = _extract_data(resp)
    if not handle:
        return f"Handle not found: {address}"
    if isinstance(handle, dict):
        return format_handle(handle)
    return f"Handle data: {handle}"


@mcp.tool()
@_safe_call
async def query_handles(
    limit: int = 50,
    offset: int = 0,
    address: Optional[str] = None,
    include_chat: bool = False,
) -> str:
    """Query handles with pagination and optional filtering.

    Args:
        limit: Maximum number of handles to return (default 50).
        offset: Number of handles to skip for pagination.
        address: Optional address filter to search for specific handles.
        include_chat: Whether to include associated chat data (default False).
    """
    body: dict = {
        "limit": limit,
        "offset": offset,
    }
    if include_chat:
        body["with"] = ["chat"]
    if address:
        body["address"] = address

    resp = await api_request("handle/query", method="POST", data=body)
    handles = _extract_data(resp) or []
    meta = resp.get("metadata") or {}

    lines = [
        f"Handles (showing {len(handles)} of {meta.get('total', '?')})",
        "=" * 40,
    ]
    for h in handles:
        lines.append(format_handle(h))
        lines.append("")
    return "\n".join(lines)


@mcp.tool()
@_safe_call
async def get_handle_focus(address: str) -> str:
    """Get the focus/Do Not Disturb status for a specific handle. This tells
    you if the person has notifications silenced.

    Args:
        address: The phone number or email address to check focus status for.
    """
    resp = await api_request(f"handle/{address}/focus")
    data = _extract_data(resp)
    if data is None:
        return f"No focus status available for {address}."
    if isinstance(data, dict):
        lines = [f"Focus status for {address}:", "=" * 40]
        for k, v in data.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)
    return f"Focus status for {address}: {data}"


@mcp.tool()
@_safe_call
async def check_imessage_availability(address: str) -> str:
    """Check if a phone number or email is registered with iMessage.
    Useful for determining if you can send an iMessage vs SMS.

    Args:
        address: The phone number or email to check iMessage availability for.
    """
    resp = await api_request(
        "handle/availability/imessage",
        params={"address": address},
    )
    data = _extract_data(resp)
    if isinstance(data, dict):
        available = data.get("available", False)
        return f"iMessage availability for {address}: {'Available' if available else 'Not available'}"
    return f"iMessage availability for {address}: {data}"


@mcp.tool()
@_safe_call
async def check_facetime_availability(address: str) -> str:
    """Check if a phone number or email is registered with FaceTime.

    Args:
        address: The phone number or email to check FaceTime availability for.
    """
    resp = await api_request(
        "handle/availability/facetime",
        params={"address": address},
    )
    data = _extract_data(resp)
    if isinstance(data, dict):
        available = data.get("available", False)
        return f"FaceTime availability for {address}: {'Available' if available else 'Not available'}"
    return f"FaceTime availability for {address}: {data}"


# ===================================================================
# CONTACT TOOLS (46-48)
# ===================================================================


@mcp.tool()
@_safe_call
async def refresh_contact_cache() -> str:
    """Refresh the contact name cache. Call this after adding or updating
    contacts so that message and chat displays show updated names."""
    global _contact_cache_loaded
    _contact_cache.clear()
    _contact_cache_loaded = False
    await _ensure_contact_cache()
    return f"Contact cache refreshed: {len(_contact_cache)} address-to-name mappings loaded."


@mcp.tool()
@_safe_call
async def get_contacts() -> str:
    """Get all contacts from the BlueBubbles server. Returns names, phone
    numbers, emails, birthdays, and other contact details."""
    resp = await api_request("contact")
    contacts = _extract_data(resp) or []
    if not contacts:
        return "No contacts found."
    lines = [f"Contacts ({len(contacts)})", "=" * 40]
    for c in contacts:
        lines.append(format_contact(c))
        lines.append("")
    return "\n".join(lines)


@mcp.tool()
@_safe_call
async def query_contacts(addresses: list[str]) -> str:
    """Look up contacts by their phone numbers or email addresses.
    Useful for resolving handle addresses to contact names.

    Args:
        addresses: List of phone numbers or email addresses to look up.
            Example: ['+11234567890', 'user@example.com'].
    """
    resp = await api_request(
        "contact/query",
        method="POST",
        data={"addresses": addresses},
    )
    contacts = _extract_data(resp) or []
    if not contacts:
        return f"No contacts found for: {', '.join(addresses)}"
    lines = [f"Contact query results ({len(contacts)} found)", "=" * 40]
    for c in contacts:
        lines.append(format_contact(c))
        lines.append("")
    return "\n".join(lines)


# ===================================================================
# ATTACHMENT TOOLS (48-50)
# ===================================================================


@mcp.tool()
@_safe_call
async def get_attachment_count() -> str:
    """Get the total number of attachments stored in the Messages database."""
    resp = await api_request("attachment/count")
    data = _extract_data(resp) or {}
    if isinstance(data, (int, float)):
        return f"Total attachments: {int(data):,}"
    total = data.get("total", data) if isinstance(data, dict) else data
    return f"Total attachments: {total}"


@mcp.tool()
@_safe_call
async def get_attachment(attachment_guid: str) -> str:
    """Get metadata about a specific attachment by its GUID.

    Args:
        attachment_guid: The GUID of the attachment (usually starts with 'att_' or similar).
    """
    resp = await api_request(f"attachment/{attachment_guid}")
    att = _extract_data(resp)
    if not att:
        return f"Attachment not found: {attachment_guid}"
    if isinstance(att, dict):
        lines = [f"Attachment: {attachment_guid}", "=" * 40]
        lines.append(f"  File Name:     {att.get('transferName', 'N/A')}")
        lines.append(f"  MIME Type:     {att.get('mimeType', 'N/A')}")
        lines.append(f"  File Size:     {att.get('totalBytes', 'N/A')} bytes")
        lines.append(f"  Is Outgoing:   {att.get('isOutgoing', 'N/A')}")
        lines.append(f"  Created:       {_ts_to_str(att.get('dateCreated'))}")
        lines.append(f"  Is Sticker:    {att.get('isSticker', False)}")
        lines.append(f"  Has Live Photo: {att.get('hasLivePhoto', False)}")
        hide = att.get("hideAttachment", False)
        lines.append(f"  Hidden:        {hide}")
        return "\n".join(lines)
    return f"Attachment data: {att}"


@mcp.tool()
@_safe_call
async def download_attachment_url(attachment_guid: str) -> str:
    """Get the download URL for an attachment. Returns a URL that can be
    used to download the file directly from the BlueBubbles server.

    Args:
        attachment_guid: The GUID of the attachment to get the download URL for.
    """
    encoded_password = quote(BLUEBUBBLES_PASSWORD, safe="")
    url = f"{BLUEBUBBLES_URL}/api/v1/attachment/{quote(attachment_guid, safe='')}/download?password={encoded_password}"
    return f"Download URL for attachment {attachment_guid}:\n{url}"


_IMAGE_MIME_PREFIXES = ("image/jpeg", "image/png", "image/gif", "image/webp", "image/heic", "image/heif", "image/tiff")
_MIME_TO_FORMAT = {
    "image/jpeg": "jpeg",
    "image/png": "png",
    "image/gif": "gif",
    "image/webp": "webp",
    "image/heic": "png",  # BB converts HEIC on download with resize
    "image/heif": "png",
    "image/tiff": "png",
}


@mcp.tool()
@_safe_call
async def view_attachment(
    attachment_guid: str,
    height: int = 300,
    quality: str = "good",
) -> list:
    """View an attachment. For images, returns the actual image content that can
    be seen directly. For non-image files, returns metadata and a download URL.

    Use the attachment GUID from message listings (shown in brackets after each
    attachment filename).

    Args:
        attachment_guid: The GUID of the attachment (from message attachment listings).
        height: Resize height in pixels for image previews (default 300). Use 0 for original size.
        quality: Image quality: 'good' (default), 'better', or 'best'. Only applies to resized images.
    """
    # First get attachment metadata
    resp = await api_request(f"attachment/{attachment_guid}")
    att = _extract_data(resp) or {}
    mime = att.get("mimeType", "application/octet-stream")
    fname = att.get("transferName", "unknown")
    total_bytes = att.get("totalBytes", 0)

    is_image = any(mime.startswith(p.split("/")[0] + "/" + p.split("/")[1]) for p in _IMAGE_MIME_PREFIXES) if "/" in mime else False
    # Simpler check:
    is_image = mime.lower().startswith("image/")

    if not is_image:
        # Non-image: return metadata + download URL
        encoded_password = quote(BLUEBUBBLES_PASSWORD, safe="")
        url = f"{BLUEBUBBLES_URL}/api/v1/attachment/{quote(attachment_guid, safe='')}/download?password={encoded_password}"
        return (
            f"Attachment: {fname}\n"
            f"Type: {mime}\n"
            f"Size: {total_bytes:,} bytes\n"
            f"This is not an image and cannot be displayed inline.\n"
            f"Download URL: {url}"
        )

    # Image: download (optionally resized) and return as Image content
    encoded_password = quote(BLUEBUBBLES_PASSWORD, safe="")
    download_url = f"/api/v1/attachment/{quote(attachment_guid, safe='')}/download"
    params = {"password": BLUEBUBBLES_PASSWORD}
    if height > 0:
        params["height"] = str(height)
        params["quality"] = quality

    client = _http_client
    assert client is not None, "HTTP client not initialized"
    download_resp = await client.get(download_url, params=params)
    download_resp.raise_for_status()

    img_bytes = download_resp.content
    content_type = download_resp.headers.get("content-type", mime).lower()
    fmt = _MIME_TO_FORMAT.get(content_type, "png")

    info_text = (
        f"Attachment: {fname}\n"
        f"Type: {mime} | Size: {total_bytes:,} bytes\n"
        f"Preview below ({height}px height):" if height > 0 else
        f"Attachment: {fname}\n"
        f"Type: {mime} | Size: {total_bytes:,} bytes\n"
        f"Full resolution:"
    )

    return [info_text, Image(data=img_bytes, format=fmt)]


# ===================================================================
# ICLOUD / FINDMY TOOLS (51-54)
# ===================================================================


@mcp.tool()
@_safe_call
async def get_findmy_devices() -> str:
    """Get a list of all FindMy devices (your Apple devices tracked in
    the FindMy network). Returns device names, locations, and status."""
    resp = await api_request("icloud/findmy/devices")
    devices = _extract_data(resp) or []
    if not devices:
        return "No FindMy devices found."
    lines = [f"FindMy Devices ({len(devices)})", "=" * 40]
    for d in devices:
        name = d.get("name") or d.get("deviceDisplayName") or "Unknown Device"
        model = d.get("deviceModel") or d.get("rawDeviceModel") or ""
        battery = d.get("batteryLevel")
        location = d.get("location") or {}
        lat = location.get("latitude", "?")
        lon = location.get("longitude", "?")
        ts = _ts_to_str(location.get("timeStamp"))

        lines.append(f"\n{name} ({model})")
        if battery is not None:
            lines.append(f"  Battery: {battery}")
        lines.append(f"  Location: {lat}, {lon}")
        if ts:
            lines.append(f"  Last Updated: {ts}")
        status = d.get("deviceStatus") or d.get("deviceClass") or ""
        if status:
            lines.append(f"  Status: {status}")
    return "\n".join(lines)


@mcp.tool()
@_safe_call
async def get_findmy_friends() -> str:
    """Get a list of friends/family members shared in FindMy.
    Returns their names and last known locations."""
    resp = await api_request("icloud/findmy/friends")
    friends = _extract_data(resp) or []
    if not friends:
        return "No FindMy friends found."
    lines = [f"FindMy Friends ({len(friends)})", "=" * 40]
    for f in friends:
        name = f.get("name") or f.get("handle") or f.get("id") or "Unknown"
        location = f.get("location") or {}
        lat = location.get("latitude", "?")
        lon = location.get("longitude", "?")
        ts = _ts_to_str(location.get("timestamp") or location.get("timeStamp"))
        address = f.get("address") or {}

        lines.append(f"\n{name}")
        lines.append(f"  Location: {lat}, {lon}")
        if ts:
            lines.append(f"  Last Updated: {ts}")
        if address:
            formatted_addr = ", ".join(
                str(v) for v in [
                    address.get("streetAddress"),
                    address.get("locality"),
                    address.get("stateCode"),
                    address.get("countryCode"),
                ] if v
            )
            if formatted_addr:
                lines.append(f"  Address: {formatted_addr}")
    return "\n".join(lines)


@mcp.tool()
@_safe_call
async def refresh_findmy_devices() -> str:
    """Refresh the FindMy device locations. Triggers a fresh location lookup
    from Apple's FindMy network. May take a moment to update."""
    resp = await api_request("icloud/findmy/devices/refresh", method="POST")
    return "FindMy device locations refresh requested."


@mcp.tool()
@_safe_call
async def refresh_findmy_friends() -> str:
    """Refresh the FindMy friend locations. Triggers a fresh location lookup
    from Apple's FindMy network. May take a moment to update."""
    resp = await api_request("icloud/findmy/friends/refresh", method="POST")
    return "FindMy friend locations refresh requested."


# ===================================================================
# MAC CONTROL TOOLS (55)
# ===================================================================


@mcp.tool()
@_safe_call
async def lock_mac() -> str:
    """Lock the Mac that the BlueBubbles server is running on.
    This will immediately lock the screen, requiring a password to unlock."""
    resp = await api_request("mac/lock", method="POST")
    return "Mac lock command sent."


# ===================================================================
# Server entry point
# ===================================================================

if __name__ == "__main__":
    logger.info("Starting BlueBubbles MCP server...")
    if not BLUEBUBBLES_PASSWORD:
        logger.error("BLUEBUBBLES_PASSWORD environment variable is required")
        sys.exit(1)
    mcp.run(transport="stdio")
