"""Read-only Gmail OAuth, message retrieval, and safe text extraction."""
from __future__ import annotations

import base64
import os
import re
from dataclasses import asdict, dataclass
from email.utils import parsedate_to_datetime
from html import unescape
from typing import Any

from fastapi import HTTPException
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

@dataclass
class GmailEmail:
    id: str; thread_id: str; sender: str; recipient: str; subject: str
    timestamp: str; body: str; snippet: str; labels: list[str]

def _settings() -> tuple[str, str, str]:
    client_id, secret = os.getenv("GOOGLE_CLIENT_ID"), os.getenv("GOOGLE_CLIENT_SECRET")
    redirect = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/api/auth/gmail/callback")
    if not client_id or not secret:
        raise HTTPException(503, "Gmail is not configured. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET on the server.")
    return client_id, secret, redirect

def create_flow(state: str | None = None) -> Flow:
    client_id, secret, redirect = _settings()
    return Flow.from_client_config({"web": {"client_id": client_id, "client_secret": secret,
        "auth_uri": "https://accounts.google.com/o/oauth2/auth", "token_uri": "https://oauth2.googleapis.com/token",
        "redirect_uris": [redirect]}}, scopes=SCOPES, state=state, redirect_uri=redirect)

def authorization_url() -> tuple[str, str]:
    flow = create_flow()
    return flow.authorization_url(access_type="offline", include_granted_scopes="true", prompt="consent")

def credentials_from_code(code: str, state: str) -> Credentials:
    import os
    os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"
    flow = create_flow(state=state)
    flow.fetch_token(code=code)
    return flow.credentials

def gmail_client(credentials: Credentials):
    return build("gmail", "v1", credentials=credentials, cache_discovery=False)

def profile(credentials: Credentials) -> dict[str, str]:
    data = gmail_client(credentials).users().getProfile(userId="me").execute()
    return {"email": data.get("emailAddress", "")}

def _header(headers: list[dict[str, str]], name: str) -> str:
    return next((x.get("value", "") for x in headers if x.get("name", "").lower() == name.lower()), "")

def _decode(data: str) -> str:
    try: return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4)).decode("utf-8", errors="replace")
    except Exception: return ""

def _plain_part(payload: dict[str, Any]) -> str:
    mime = payload.get("mimeType", "")
    body = payload.get("body", {}).get("data", "")
    if mime == "text/plain" and body: return _decode(body)
    for part in payload.get("parts", []):
        text = _plain_part(part)
        if text: return text
    if mime == "text/html" and body:
        return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", _decode(body)))).strip()
    return ""

def fetch_messages(credentials: Credentials, limit: int = 25, page_token: str | None = None, query: str | None = None) -> tuple[list[GmailEmail], str | None]:
    api = gmail_client(credentials)
    listing = api.users().messages().list(userId="me", maxResults=min(max(limit, 1), 50), pageToken=page_token, q=query or None).execute()
    emails: list[GmailEmail] = []
    for item in listing.get("messages", []):
        raw = api.users().messages().get(userId="me", id=item["id"], format="full").execute()
        payload, headers = raw.get("payload", {}), raw.get("payload", {}).get("headers", [])
        date = _header(headers, "Date")
        try: timestamp = parsedate_to_datetime(date).isoformat()
        except Exception: timestamp = raw.get("internalDate", "")
        emails.append(GmailEmail(id=raw["id"], thread_id=raw.get("threadId", ""), sender=_header(headers, "From"), recipient=_header(headers, "To"), subject=_header(headers, "Subject") or "(No subject)", timestamp=timestamp, body=_plain_part(payload), snippet=raw.get("snippet", ""), labels=raw.get("labelIds", [])))
    return emails, listing.get("nextPageToken")

def serialize(email: GmailEmail) -> dict: return asdict(email)
