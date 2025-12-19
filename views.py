from __future__ import annotations

import base64
import re
import os
import json
import pandas as pd
import mimetypes
import email.utils
from typing import Dict, List, Optional, Tuple
import datetime as dt
from zoneinfo import ZoneInfo
import joblib

from django.conf import settings
from django.http import HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_POST

# Google / Gmail
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request as GoogleRequest

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
]

REDIRECT_URI = getattr(settings, "GMAIL_REDIRECT_URI", "http://localhost:8000/gmail/callback/")
CREDENTIALS_JSON = getattr(settings, "GMAIL_CREDENTIALS_FILE", "credentials/credentials.json")
MAX_RESULTS_DEFAULT = 25

MODEL_PATH = os.path.join(settings.BASE_DIR, "static", "spam_classifier_model.pkl")
VEC_PATH   = os.path.join(settings.BASE_DIR, "static", "vectorizer.pkl")

spam_model = joblib.load(MODEL_PATH)
spam_vectorizer = joblib.load(VEC_PATH)
# --------------------------------------------------------------------------------------
# Helpers: session creds & service
# --------------------------------------------------------------------------------------

def _save_creds_to_session(request, creds: Credentials) -> None:
    request.session["token"] = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": creds.scopes,
    }

def _ensure_service(request):
    """
    Returns (gmail_service, redirect_response_if_needed)
    """
    token = request.session.get("token")
    if not token:
        return None, redirect("gmail_auth")

    creds = Credentials(**token)
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(GoogleRequest())
            _save_creds_to_session(request, creds)
        except Exception:
            return None, redirect("gmail_auth")

    service = build("gmail", "v1", credentials=creds)
    return service, None

# --------------------------------------------------------------------------------------
# Basic pages (no DB)
# --------------------------------------------------------------------------------------

def openLogin(request):
    return render(request, "admin/admin_login.html", {})

def webIndex(request):
    return render(request, "web/index.html", {"web_email": request.session.get("web_email")})

def about(request):
    return render(request, "web/about.html", {"web_email": request.session.get("web_email")})

def contact(request):
    return render(request, "web/contact.html", {"web_email": request.session.get("web_email")})

def service(request):
    return render(request, "web/service.html", {"web_email": request.session.get("web_email")})

def feature(request):
    return render(request, "web/feature.html", {"web_email": request.session.get("web_email")})

def pricing(request):
    return render(request, "web/pricing.html", {"web_email": request.session.get("web_email")})

def loginUser(request):
    # No local DB login. Use Google OAuth below.
    return render(request, "web/login.html", {})

def register(request):
    # No local DB register. Keep page if you show info.
    return render(request, "web/register.html", {})

def profile(request):
    return render(request, "web/profile.html", {"web_email": request.session.get("web_email")})

def email(request):
    return render(request, "web/email.html", {})

def webLogout(request):
    for k in ("web_email", "web_name", "token"):
        request.session.pop(k, None)
    return render(request, "web/index.html")

# --------------------------------------------------------------------------------------
# Google OAuth (no DB)
# --------------------------------------------------------------------------------------

def gmail_auth(request):
    flow = Flow.from_client_secrets_file(
        CREDENTIALS_JSON,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI,
    )
    auth_url, state = flow.authorization_url(
        prompt="consent",
        access_type="offline",
        include_granted_scopes="true",
    )
    request.session["state"] = state
    return redirect(auth_url)

def gmail_callback(request):
    state = request.session.get("state")
    if not state:
        return HttpResponse("Session state missing. Try authenticating again.", status=400)

    flow = Flow.from_client_secrets_file(
        CREDENTIALS_JSON,
        scopes=SCOPES,
        state=state,
        redirect_uri=REDIRECT_URI,
    )
    flow.fetch_token(authorization_response=request.build_absolute_uri())
    creds = flow.credentials
    _save_creds_to_session(request, creds)

    # Set display info in session (email)
    try:
        svc = build("gmail", "v1", credentials=creds)
        prof = svc.users().getProfile(userId="me").execute()
        email_addr = prof.get("emailAddress", "")
        request.session["web_email"] = email_addr
        request.session["web_name"] = (email_addr.split("@", 1)[0] or "User").title()
    except Exception:
        pass

    return redirect("email_inbox")

# --------------------------------------------------------------------------------------
# Gmail utilities
# --------------------------------------------------------------------------------------

def _get_header(headers: List[Dict[str, str]], name: str) -> Optional[str]:
    for h in headers:
        if h.get("name", "").lower() == name.lower():
            return h.get("value")
    return None

def _decode_b64url(data: str) -> str:
    try:
        return base64.urlsafe_b64decode(data.encode("utf-8")).decode("utf-8", errors="replace")
    except Exception:
        return ""

def _extract_bodies(payload: Dict) -> Tuple[str, str]:
    """
    Returns (text_plain, text_html). Handles nested multiparts.
    """
    text_plain = ""
    text_html = ""

    def walk(part):
        nonlocal text_plain, text_html
        mime = part.get("mimeType", "")
        body = part.get("body", {}) or {}
        data = body.get("data")

        if mime == "text/plain" and data and not text_plain:
            text_plain = _decode_b64url(data)
        elif mime == "text/html" and data and not text_html:
            text_html = _decode_b64url(data)

        for sub in part.get("parts", []) or []:
            walk(sub)

    walk(payload)
    return text_plain, text_html

def _normalize_metadata(g: Dict) -> Dict:
    payload = g.get("payload", {}) or {}
    headers = payload.get("headers", []) or []
    subject = _get_header(headers, "Subject") or "(No Subject)"
    from_ = _get_header(headers, "From") or ""
    to_ = _get_header(headers, "To") or ""
    date = _get_header(headers, "Date") or ""
    snippet = g.get("snippet", "")
    return {
        "id": g.get("id"),
        "thread_id": g.get("threadId"),
        "subject": subject,
        "from": from_,
        "to": to_,
        "date": date,
        "snippet": snippet,
        "label_ids": g.get("labelIds", []),
    }

def _list_messages(service, label_id: str, q: str, page_token: Optional[str], max_results: int = MAX_RESULTS_DEFAULT):
    res = service.users().messages().list(
        userId="me",
        labelIds=[label_id],
        q=q or None,
        pageToken=page_token or None,
        maxResults=max_results,
    ).execute()

    emails = []
    for m in res.get("messages", []) or []:
        g = service.users().messages().get(
            userId="me",
            id=m["id"],
            format="metadata",
            metadataHeaders=["Subject", "From", "To", "Date"],
        ).execute()
        emails.append(_normalize_metadata(g))

    return emails, res.get("nextPageToken", "")

# --------------------------------------------------------------------------------------
# Inbox / Sent / Spam (no DB)
# --------------------------------------------------------------------------------------

_SPAM_FULL = None

def _load_spam_csv_full(filename="email_addresses.csv"):
    global _SPAM_FULL
    if _SPAM_FULL is not None:
        return _SPAM_FULL
    path = os.path.join(settings.BASE_DIR, "static", filename)
    if not os.path.exists(path):
        _SPAM_FULL = set()
        return _SPAM_FULL
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception:
        try:
            df = pd.read_excel(path, dtype=str, keep_default_na=False)
        except Exception:
            _SPAM_FULL = set()
            return _SPAM_FULL
    df.columns = df.columns.str.strip().str.lower()
    if "email" in df.columns:
        emails_col = df["email"].astype(str)
    else:
        emails_col = df.iloc[:, 0].astype(str)
    labels_col = None
    if "label" in df.columns:
        labels_col = df["label"].astype(str)
    elif df.shape[1] > 1:
        labels_col = df.iloc[:, 1].astype(str)
    full_set = set()
    for i, raw in enumerate(emails_col):
        e = str(raw).strip().lower()
        label_val = ""
        if labels_col is not None and i < len(labels_col):
            try:
                label_val = str(labels_col.iloc[i]).strip().lower()
            except Exception:
                label_val = ""
        if label_val in {"1", "true", "spam", "yes", "y"}:
            if e:
                full_set.add(e)
    _SPAM_FULL = full_set
    return _SPAM_FULL

def _extract_email_address(from_header: str) -> str:
    if not from_header:
        return ""
    from_header = from_header.strip()
    m = re.search(r'([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})', from_header)
    if m:
        return m.group(1).lower()
    s = re.sub(r'^[\'"]+|[\'"]+$', '', from_header)
    s = re.sub(r'^[^<]*<|>[^>]*$', '', s).strip()
    if " " in s and "<" not in from_header:
        parts = s.split()
        possible = [p for p in parts if "@" in p]
        if possible:
            return possible[0].lower()
    return s.lower()

def email_inbox(request):
    service, redir = _ensure_service(request)
    if redir:
        return redir
    q = request.GET.get("q", "")
    page = request.GET.get("page", "")
    limit = int(request.GET.get("limit", MAX_RESULTS_DEFAULT))
    res = service.users().messages().list(
        userId="me",
        labelIds=["INBOX"],
        q=q or None,
        pageToken=page or None,
        maxResults=limit
    ).execute()
    next_page = res.get("nextPageToken", "")
    spam_full = _load_spam_csv_full("email_addresses.csv")
    # print(spam_full)
    email_data = []
    for msg in res.get("messages", []) or []:
        g = service.users().messages().get(
            userId="me",
            id=msg["id"],
            format="metadata",
            metadataHeaders=["Subject", "From", "To", "Date"],
        ).execute()
        meta = _normalize_metadata(g)
        snippet = meta.get("snippet", "")
        from_header = meta.get("from", "") or meta.get("From", "")
        sender_email = _extract_email_address(from_header)
        if sender_email and sender_email in spam_full:
            print(sender_email);
            mark_as_spam(request, msg["id"])
        email_data.append({
            "id": meta.get("id", ""),
            "em_subject": meta.get("subject", ""),
            "em_to": meta.get("from", ""),
            "em_message": snippet,
        })
    return render(
        request,
        "web/email.html",
        {"emails": email_data, "section": "inbox", "next_page": next_page, "q": q},
    )

def gmail_sent(request):
    service, redir = _ensure_service(request)
    if redir:
        return redir

    q = request.GET.get("q", "")
    page = request.GET.get("page", "")
    limit = int(request.GET.get("limit", MAX_RESULTS_DEFAULT))

    emails, next_page = _list_messages(service, "SENT", q, page, limit)

    email_data = [{
        "id": e["id"],
        "subject": e["subject"],
        "sender": e["to"],
        "snippet": e["snippet"],
    } for e in emails]

    return render(
        request,
        "web/sent_emails.html",
        {"emails": email_data, "section": "sent", "next_page": next_page, "q": q},
    )

def gmail_spam(request):
    service, redir = _ensure_service(request)
    if redir:
        return redir

    q = request.GET.get("q", "")
    page = request.GET.get("page", "")
    limit = int(request.GET.get("limit", MAX_RESULTS_DEFAULT))

    emails, next_page = _list_messages(service, "SPAM", q, page, limit)

    email_data = [{
        "id": e["id"],
        "subject": e["subject"],
        "sender": e["from"],
        "snippet": e["snippet"],
    } for e in emails]

    return render(
        request,
        "web/spam_emails.html",
        {"emails": email_data, "section": "spam", "next_page": next_page, "q": q},
    )

# --------------------------------------------------------------------------------------
# Message details (hide replies for Spam & Sent)
# --------------------------------------------------------------------------------------

def email_details(request, msg_id):
    service, redir = _ensure_service(request)
    if redir:
        return redir

    msg = service.users().messages().get(userId="me", id=msg_id, format="full").execute()
    payload = msg.get("payload", {}) or {}
    headers = payload.get("headers", []) or []

    subject = _get_header(headers, "Subject") or "(No Subject)"
    sender = _get_header(headers, "From") or ""
    text_plain, text_html = _extract_bodies(payload)

    label_ids = set(msg.get("labelIds", []) or [])
    is_spam = "SPAM" in label_ids
    is_sent = "SENT" in label_ids
    can_reply = not (is_spam or is_sent)

    return render(
        request,
        "web/email_details.html",
        {
            "id": msg_id,
            "subject": subject,
            "sender": sender,
            "body": text_plain or "",
            "body_html": text_html,
            "is_spam": is_spam,
            "is_sent": is_sent,
            "can_reply": can_reply,
        },
    )

# --------------------------------------------------------------------------------------
# Quick actions (no DB)
# --------------------------------------------------------------------------------------

@require_POST
def toggle_star(request, msg_id):
    service, redir = _ensure_service(request)
    if redir:
        return redir

    g = service.users().messages().get(userId="me", id=msg_id, format="metadata").execute()
    labels = set(g.get("labelIds", []))
    if "STARRED" in labels:
        body = {"removeLabelIds": ["STARRED"]}
        new_state = False
    else:
        body = {"addLabelIds": ["STARRED"]}
        new_state = True

    service.users().messages().modify(userId="me", id=msg_id, body=body).execute()
    return JsonResponse({"ok": True, "starred": new_state})

@require_POST
def mark_read(request, msg_id):
    service, redir = _ensure_service(request)
    if redir:
        return redir
    service.users().messages().modify(userId="me", id=msg_id, body={"removeLabelIds": ["UNREAD"]}).execute()
    return JsonResponse({"ok": True})

@require_POST
def mark_unread(request, msg_id):
    service, redir = _ensure_service(request)
    if redir:
        return redir
    service.users().messages().modify(userId="me", id=msg_id, body={"addLabelIds": ["UNREAD"]}).execute()
    return JsonResponse({"ok": True})

@require_POST
def archive(request, msg_id):
    service, redir = _ensure_service(request)
    if redir:
        return redir
    service.users().messages().modify(userId="me", id=msg_id, body={"removeLabelIds": ["INBOX"]}).execute()
    return JsonResponse({"ok": True})

def mark_as_spam(request, msg_id):
    service, redir = _ensure_service(request)
    if redir:
        return redir
    try:
        service.users().messages().modify(
            userId="me",
            id=msg_id,
            body={"addLabelIds": ["SPAM"], "removeLabelIds": ["INBOX"]},
        ).execute()
        return HttpResponse("Marked as spam.")
    except Exception as e:
        return HttpResponse(f"Failed to mark as spam: {str(e)}", status=500)

def mark_as_not_spam(request, msg_id):
    service, redir = _ensure_service(request)
    if redir:
        return redir
    try:
        service.users().messages().modify(
            userId="me",
            id=msg_id,
            body={"addLabelIds": ["INBOX"], "removeLabelIds": ["SPAM"]},
        ).execute()
        return HttpResponse("Marked as not spam.")
    except Exception as e:
        return HttpResponse(f"Failed to mark as not spam: {str(e)}", status=500)


def is_spam(text):
    X = spam_vectorizer.transform([text])
    return spam_model.predict(X)[0] == 1
# --------------------------------------------------------------------------------------
# Auto-reply (keyword-based), no DB
# --------------------------------------------------------------------------------------

INTENT_TEMPLATES: Dict[str, Dict[str, str]] = {
    "password": {"subject": "Password Reset Instructions",
                 "body": "Hi,\n\nTo reset your password, click the 'Forgot Password' link on the login page. "
                         "If you still face issues, reply with your registered email ID so we can assist further.\n\nRegards,\nSupport Team"},
    "track":    {"subject": "Order Tracking Information",
                 "body": "Hi,\n\nYou can track your order using the tracking link sent to your email. "
                         "If you need the link again, reply with your order ID.\n\nRegards,\nSupport Team"},
    "cancel":   {"subject": "Subscription Cancellation Update",
                 "body": "Hi,\n\nWe're sorry to see you go. Your subscription has been scheduled for cancellation. "
                         "If this was a mistake, reply to this email and we'll help you reverse it.\n\nRegards,\nSupport Team"},
    "refund":   {"subject": "Refund Request Received",
                 "body": "Hi,\n\nWe’ve noted your refund request. Please share your order ID so we can process it. "
                         "Refunds are available within 7 days of purchase per our policy.\n\nRegards,\nSupport Team"},
    "support":  {"subject": "We’re On It — Support Ticket Opened",
                 "body": "Hi,\n\nThanks for contacting us. Our support team will get back to you shortly with an update.\n\nRegards,\nSupport Team"},
    "login":    {"subject": "Login Help",
                 "body": "Hi,\n\nPlease ensure you’re using the correct email and password. "
                         "If you forgot your password, use the 'Forgot Password' link to reset it.\n\nRegards,\nSupport Team"},
    "delivery": {"subject": "Delivery Timeline",
                 "body": "Hi,\n\nDeliveries typically take 3–5 business days. "
                         "You can check live status using your tracking link.\n\nRegards,\nSupport Team"},
    "payment":  {"subject": "Payment Options",
                 "body": "Hi,\n\nWe accept credit/debit cards, UPI, and PayPal. "
                         "If a payment failed, please retry or share a screenshot for assistance.\n\nRegards,\nSupport Team"},
    "invoice":  {"subject": "Invoice Download",
                 "body": "Hi,\n\nYou can download your invoice from your account dashboard under 'Orders'. "
                         "Reply with your order ID if you need us to email it to you.\n\nRegards,\nSupport Team"},
    "return":   {"subject": "Return Request Steps",
                 "body": "Hi,\n\nYou may return any item within 30 days in original condition. "
                         "Reply with your order ID and reason for return to proceed.\n\nRegards,\nSupport Team"},
    "order":    {"subject": "Order Assistance",
                 "body": "Hi,\n\nPlease share your order ID so we can assist you better.\n\nRegards,\nSupport Team"},
    "subscription": {"subject": "Manage Your Subscription",
                     "body": "Hi,\n\nYou can manage your subscription in your account settings. "
                             "If you need help changing your plan, reply here.\n\nRegards,\nSupport Team"},
    "delay":    {"subject": "Apologies for the Delay",
                 "body": "Hi,\n\nWe’re sorry for the delay. Please allow a bit more time. "
                         "If you don’t see progress soon, reply with your order ID and we’ll escalate.\n\nRegards,\nSupport Team"},
    "damage":   {"subject": "Damaged Item — Next Steps",
                 "body": "Hi,\n\nSorry your item arrived damaged. Please share a clear photo and your order ID so we can resolve this quickly.\n\nRegards,\nSupport Team"},
    "contact":  {"subject": "Contact Details",
                 "body": "Hi,\n\nYou can reach us via email or phone between 9am–6pm, Monday to Friday.\n\nRegards,\nSupport Team"},
    "meeting":  {"subject": "Scheduling a Meeting",
                 "body": "Hi,\n\nWe’d be happy to schedule a meeting. Please share your availability and preferred timezone.\n\nRegards,\nSupport Team"},
    "reschedule": {"subject": "Meeting Reschedule",
                   "body": "Hi,\n\nNo problem. Share your preferred new date and time, and we’ll send an updated invite.\n\nRegards,\nSupport Team"},
    "appointment": {"subject": "Appointment Request Received",
                    "body": "Hi,\n\nWe’ve received your appointment request. Our team will follow up shortly to confirm.\n\nRegards,\nSupport Team"},
    "call":     {"subject": "We Can Arrange a Call",
                 "body": "Hi,\n\nPlease share your availability and phone number, and we’ll arrange a call at your convenience.\n\nRegards,\nSupport Team"},
}

def _detect_intent(subject_text: str, body_text: str) -> Optional[str]:
    text = f"{subject_text or ''} {body_text or ''}".lower()
    for intent in INTENT_TEMPLATES.keys():
        if intent in text:
            return intent
    return None

@require_POST
def send_reply(request, msg_id):
    service, redir = _ensure_service(request)
    if redir:
        return redir

    # Block replies for Spam or Sent
    try:
        g = service.users().messages().get(userId="me", id=msg_id, format="metadata").execute()
        labels = set(g.get("labelIds", []) or [])
        if "SPAM" in labels or "SENT" in labels:
            return JsonResponse({"error": "Replies are disabled for messages in Spam or Sent."}, status=403)
    except Exception:
        pass

    reply_type   = request.POST.get("reply_type")           # "auto" or "manual"
    raw_to       = request.POST.get("to", "")               # e.g., "Alice <alice@x.com>"
    original_sub = request.POST.get("subject", "")
    manual_body  = request.POST.get("manual_body", "")
    original_body = request.POST.get("body", "")            # optional (original content)

    # Extract address
    match = re.search(r"[\w\.-]+@[\w\.-]+", raw_to)
    to = match.group(0) if match else None
    if not to:
        return JsonResponse({"error": "Invalid email address"}, status=400)

    # Thread info
    thread_id = None
    message_id_hdr = None
    try:
        orig = service.users().messages().get(userId="me", id=msg_id, format="metadata").execute()
        thread_id = orig.get("threadId")
        hdrs = orig.get("payload", {}).get("headers", []) or []
        if not original_sub:
            original_sub = _get_header(hdrs, "Subject") or original_sub
        for h in hdrs:
            if h.get("name", "").lower() == "message-id":
                message_id_hdr = h.get("value")
                break
    except Exception:
        pass

    # Compose subject/body
    if reply_type == "auto":
        intent = _detect_intent(original_sub, original_body)
        if intent and intent in INTENT_TEMPLATES:
            out_subject = INTENT_TEMPLATES[intent]["subject"]
            out_body = INTENT_TEMPLATES[intent]["body"]
        else:
            out_subject = "Thank you for contacting us"
            out_body = (
                "Hi,\n\nThank you for reaching out. We have received your message and will get back to you soon."
                "\n\nRegards,\nSupport Team"
            )
    else:
        out_subject = f"Re: {original_sub}".strip() if original_sub else "Re: Your message"
        out_body = manual_body or "No message provided."

    # RFC822
    msg = MIMEText(out_body, _charset="utf-8")
    msg["to"] = to
    msg["subject"] = out_subject
    if message_id_hdr:
        msg["In-Reply-To"] = message_id_hdr
        msg["References"] = message_id_hdr

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
    body = {"raw": raw}
    if thread_id:
        body["threadId"] = thread_id

    try:
        service.users().messages().send(userId="me", body=body).execute()
        if request.headers.get("x-requested-with") == "XMLHttpRequest":
            return JsonResponse({"message": "Reply sent successfully."})
        return redirect("email_inbox")
    except Exception as e:
        return JsonResponse({"error": f"Gmail API error: {str(e)}"}, status=500)

def _build_mime_message(
    to: str,
    cc: str,
    bcc: str,
    subject: str,
    text_body: str | None,
    html_body: str | None,
    attachments,                   # list of Django UploadedFile objects
    from_email: str | None = None  # optional verified send-as alias
) -> MIMEMultipart:
    """
    Creates a multipart/mixed email with multipart/alternative body (text + html)
    and optional attachments. Gmail total size limit ~35MB after base64 encoding.
    """
    root = MIMEMultipart("mixed")
    if from_email:
        root["from"] = from_email
    if to:
        root["to"] = to
    if cc:
        root["cc"] = cc
    if bcc:
        root["bcc"] = bcc
    if subject:
        root["subject"] = subject

    # Plain + HTML body
    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText((text_body or ""), "plain", "utf-8"))
    if html_body:
        alt.attach(MIMEText(html_body, "html", "utf-8"))
    root.attach(alt)

    # Attachments
    for f in attachments or []:
        filename = f.name
        ctype, encoding = mimetypes.guess_type(filename)
        if ctype is None or encoding is not None:
            ctype = "application/octet-stream"
        maintype, subtype = ctype.split("/", 1)
        part = MIMEBase(maintype, subtype)
        part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", "attachment", filename=filename)
        root.attach(part)

    return root

def compose(request):
    """
    Render compose page. Lists verified send-as aliases for the From dropdown.
    """
    service, redir = _ensure_service(request)
    if redir:
        return redir

    aliases = []
    try:
        resp = service.users().settings().sendAs().list(userId="me").execute()
        aliases = resp.get("sendAs", [])
    except Exception:
        pass

    return render(request, "web/compose.html", {
        "aliases": aliases,
        "from_default": request.session.get("web_email", ""),
    })

@require_POST
def compose_send(request):
    """
    Send a new email via Gmail API (users.messages.send).
    """
    service, redir = _ensure_service(request)
    if redir:
        return redir

    to = request.POST.get("to", "").strip()
    cc = request.POST.get("cc", "").strip()
    bcc = request.POST.get("bcc", "").strip()
    subject = request.POST.get("subject", "").strip()

    text_body = request.POST.get("body", "").rstrip()
    html_body = request.POST.get("html_body", "").strip() if request.POST.get("use_html") == "1" else None

    from_email = request.POST.get("from_email", "").strip() or None
    files = request.FILES.getlist("attachments")

    if not to:
        return JsonResponse({"error": "Recipient (To) is required."}, status=400)

    mime = _build_mime_message(to, cc, bcc, subject, text_body, html_body, files, from_email)
    raw = base64.urlsafe_b64encode(mime.as_bytes()).decode("utf-8")
    body = {"raw": raw}

    try:
        service.users().messages().send(userId="me", body=body).execute()
        return redirect("gmail_sent")
    except Exception as e:
        return JsonResponse({"error": f"Gmail API send error: {str(e)}"}, status=500)

@require_POST
def compose_save_draft(request):
    """
    Save a draft (create or update) via Gmail API.
    """
    service, redir = _ensure_service(request)
    if redir:
        return redir

    to = request.POST.get("to", "").strip()
    cc = request.POST.get("cc", "").strip()
    bcc = request.POST.get("bcc", "").strip()
    subject = request.POST.get("subject", "").strip()
    text_body = request.POST.get("body", "").rstrip()
    html_body = request.POST.get("html_body", "").strip() if request.POST.get("use_html") == "1" else None
    from_email = request.POST.get("from_email", "").strip() or None
    files = request.FILES.getlist("attachments")
    draft_id = request.POST.get("draft_id")

    mime = _build_mime_message(to, cc, bcc, subject, text_body, html_body, files, from_email)
    raw = base64.urlsafe_b64encode(mime.as_bytes()).decode("utf-8")

    try:
        if draft_id:
            service.users().drafts().update(
                userId="me",
                id=draft_id,
                body={"message": {"raw": raw}}
            ).execute()
        else:
            service.users().drafts().create(
                userId="me",
                body={"message": {"raw": raw}}
            ).execute()

        if request.headers.get("x-requested-with") == "XMLHttpRequest":
            return JsonResponse({"ok": True, "draftId": draft_id})
        return redirect("gmail_drafts")
    except Exception as e:
        return JsonResponse({"ok": False, "error": f"Gmail API draft error: {str(e)}"}, status=500)

# ---------- LIST DRAFTS ----------
def drafts_list(request):
    service, redir = _ensure_service(request)
    if redir:
        return redir

    page = request.GET.get("page", "")
    q = request.GET.get("q", "")  # optional client-side filter

    resp = service.users().drafts().list(userId="me", maxResults=50, pageToken=page or None).execute()
    drafts = resp.get("drafts", []) or []
    next_page = resp.get("nextPageToken")

    items = []
    for d in drafts:
        g = service.users().drafts().get(userId="me", id=d["id"], format="metadata").execute()
        msg = g.get("message", {}) or {}
        payload = msg.get("payload", {}) or {}
        headers = payload.get("headers", []) or []

        subject = _get_header(headers, "Subject") or "(No Subject)"
        to_ = _get_header(headers, "To") or ""
        date_ = _get_header(headers, "Date") or ""
        snippet = msg.get("snippet", "")

        if q:
            hay = f"{subject} {to_} {snippet}".lower()
            if q.lower() not in hay:
                continue

        items.append({
            "id": d["id"],
            "subject": subject,
            "to": to_,
            "date": date_,
            "snippet": snippet,
        })

    return render(request, "web/drafts.html", {
        "drafts": items,
        "next_page": next_page,
        "q": q,
        "section": "drafts",
    })

# ---------- DRAFT DETAILS ----------
def draft_details(request, draft_id):
    service, redir = _ensure_service(request)
    if redir:
        return redir

    g = service.users().drafts().get(userId="me", id=draft_id, format="full").execute()
    msg = g.get("message", {}) or {}
    payload = msg.get("payload", {}) or {}
    headers = payload.get("headers", []) or []

    subject = _get_header(headers, "Subject") or "(No Subject)"
    from_ = _get_header(headers, "From") or ""
    to_ = _get_header(headers, "To") or ""
    cc_ = _get_header(headers, "Cc") or ""
    bcc_ = _get_header(headers, "Bcc") or ""
    text_plain, text_html = _extract_bodies(payload)

    return render(request, "web/draft_details.html", {
        "draft_id": draft_id,
        "subject": subject,
        "from_email": from_,
        "to": to_,
        "cc": cc_,
        "bcc": bcc_,
        "body": text_plain or "",
        "body_html": text_html or "",
    })

# ---------- SEND A DRAFT ----------
@require_POST
def draft_send(request, draft_id):
    service, redir = _ensure_service(request)
    if redir:
        return redir

    try:
        service.users().drafts().send(userId="me", body={"id": draft_id}).execute()
        if request.headers.get("x-requested-with") == "XMLHttpRequest":
            return JsonResponse({"ok": True})
        return redirect("gmail_sent")
    except Exception as e:
        return JsonResponse({"ok": False, "error": f"Gmail API send draft error: {str(e)}"}, status=500)

# ---------- DELETE A DRAFT ----------
@require_POST
def draft_delete(request, draft_id):
    service, redir = _ensure_service(request)
    if redir:
        return redir

    try:
        service.users().drafts().delete(userId="me", id=draft_id).execute()
        if request.headers.get("x-requested-with") == "XMLHttpRequest":
            return JsonResponse({"ok": True})
        return redirect("gmail_drafts")
    except Exception as e:
        return JsonResponse({"ok": False, "error": f"Gmail API delete draft error: {str(e)}"}, status=500)

# ---------- EDIT DRAFT ----------
def compose_edit_draft(request, draft_id):
    service, redir = _ensure_service(request)
    if redir:
        return redir

    # Load aliases for “From” dropdown
    aliases = []
    from_default = request.session.get("web_email", "")
    try:
        resp = service.users().settings().sendAs().list(userId="me").execute()
        aliases = resp.get("sendAs", [])
    except Exception:
        pass

    g = service.users().drafts().get(userId="me", id=draft_id, format="full").execute()
    msg = g.get("message", {}) or {}
    payload = msg.get("payload", {}) or {}
    headers = payload.get("headers", []) or []

    subject = _get_header(headers, "Subject") or ""
    from_ = _get_header(headers, "From") or from_default
    to_ = _get_header(headers, "To") or ""
    cc_ = _get_header(headers, "Cc") or ""
    bcc_ = _get_header(headers, "Bcc") or ""
    text_plain, text_html = _extract_bodies(payload)

    return render(request, "web/compose.html", {
        "aliases": aliases,
        "from_default": from_default,
        "from_email": from_,
        "to": to_,
        "cc": cc_,
        "bcc": bcc_,
        "subject": subject,
        "body": text_plain or "",
        "html_body": text_html or "",
        "use_html": bool(text_html),
        "draft_id": draft_id,
    })

# ======================================================================================
#                                    CHATBOT (NEW)
# ======================================================================================

def _safe(s, fallback=""):
    return s if isinstance(s, str) else fallback

def _headers_to_dict(msg):
    hdrs = {}
    for h in msg.get("payload", {}).get("headers", []) or []:
        name = h.get("name")
        value = h.get("value")
        if name and value:
            hdrs[name.lower()] = value
    return hdrs

def _get_header_val(headers, key, default=""):
    return headers.get(key.lower(), default)

def _human_sender(h):
    try:
        name, addr = email.utils.parseaddr(h)
        return name or addr or ""
    except Exception:
        return _safe(h)

def _list_ids(service, q=None, label_ids=None, max_results=10):
    kw = {"userId": "me", "maxResults": max_results}
    if q: kw["q"] = q
    if label_ids: kw["labelIds"] = label_ids
    resp = service.users().messages().list(**kw).execute()
    return resp.get("messages", []) or []

def _get_meta(service, msg_id):
    return service.users().messages().get(
        userId="me",
        id=msg_id,
        format="metadata",
        metadataHeaders=["From", "Subject", "Date", "To"]
    ).execute()

def _gmail_search(service, q: str, max_results: int = 25):
    """
    Search Gmail with a query string; return normalized list sorted by newest first.
    """
    res = service.users().messages().list(userId="me", q=q or None, maxResults=max_results).execute()
    items = []
    for m in res.get("messages", []) or []:
        g = service.users().messages().get(
            userId="me",
            id=m["id"],
            format="metadata",
            metadataHeaders=["Subject", "From", "To", "Date"],
        ).execute()
        meta = _normalize_metadata(g)
        items.append({
            "id": meta["id"],
            "subject": meta["subject"],
            "from": meta["from"],
            "to": meta["to"],
            "date": meta["date"],
            "snippet": meta["snippet"],
            "internal": int(g.get("internalDate", "0")),
        })
    items.sort(key=lambda x: x.get("internal", 0), reverse=True)
    return items

def _count_unread_inbox(service):
    msgs = _list_ids(service, q="label:INBOX is:unread", max_results=50)
    return len(msgs)

def _latest_inbox_glance(service):
    msgs = _list_ids(service, q="label:INBOX", max_results=1)
    if not msgs: return None
    meta = _get_meta(service, msgs[0]["id"])
    h = _headers_to_dict(meta)
    return {
        "from": _human_sender(_get_header_val(h, "From", "")),
        "subject": _get_header_val(h, "Subject", "(No Subject)"),
    }

def _spam_subjects(service, limit=5):
    msgs = _list_ids(service, label_ids=["SPAM"], max_results=limit)
    subjects = []
    for m in msgs:
        meta = _get_meta(service, m["id"])
        h = _headers_to_dict(meta)
        subjects.append(_get_header_val(h, "Subject", "(No Subject)"))
    return subjects

def _set_vacation_responder(service, enable, start_ts=None, end_ts=None, subject="", body=""):
    payload = {
        "enableAutoReply": bool(enable),
        "responseSubject": subject or None,
        "responseBodyPlainText": body or None,
    }
    if start_ts: payload["startTime"] = int(start_ts * 1000)
    if end_ts:   payload["endTime"]   = int(end_ts   * 1000)
    service.users().settings().updateVacation(userId="me", body=payload).execute()

def _detect_chat_intent(text):
    t = (_safe(text).lower()).strip()

    # Greeting / help
    if re.search(r"\b(hi|hello|hey|help|what can you do)\b", t): return "help"

    # Inbox status
    if re.search(r"\b(show (my )?inbox|do i have (any )?new (mails?|emails?)|new (mails?|emails?))\b", t):
        return "inbox_status"

    # Spam status
    if re.search(r"\b(any|show)\s+spam\b|\bspam (mails?|emails?)\b", t):
        return "spam_status"

    # Unread summary for today (NEW)
    if (
        re.search(r"\b(summarize|summary|sum\s*up)\b.*\bunread\b.*\b(today|todays|from today|last 24h|24 hours)\b", t)
        or t in {"summarize unread today", "unread today", "summarize unread emails from today"}
    ):
        return "unread_today"

    # Draft reply scaffold
    if re.search(r"^reply to .+? (email|mail) about .+", t):
        return "draft_reply"

    # Set OOO / auto-reply
    if re.search(r"(i'?m|i am)\s+out of office|set (auto[-\s]?reply|vacation)", t):
        return "set_ooo"

    # How to compose
    if re.search(r"how (do|to) (i )?compose (an )?email", t):
        return "how_compose"

    return "fallback"

def _h_help():
    return (
        "Hello 👋 I’m your Email Assistant Bot. I can help you with:\n"
        "1️⃣ Checking Inbox, Sent, Spam, Drafts\n"
        "2️⃣ Composing quick replies\n"
        "3️⃣ Answering FAQs\n"
        "4️⃣ Detecting spam & suggesting actions\n"
        "5️⃣ Summarizing today’s unread emails\n"
        "How can I help you today?"
    )

def _h_inbox(service):
    unread = _count_unread_inbox(service)
    latest = _latest_inbox_glance(service)
    if latest:
        return (
            f"📩 You have {unread} new mails in your inbox. "
            f"The latest is from {latest['from']} with subject '{latest['subject']}'. "
            f"Do you want to preview it?"
        )
    return f"📩 You have {unread} new mails in your inbox."

def _h_spam(service):
    subs = _spam_subjects(service, limit=3)
    if subs:
        subjects = ", ".join([f"‘{s}’" for s in subs])
        return f"⚠️ Yes, you have {len(subs)} spam mails. Subjects: {subjects}. Do you want me to delete them?"
    return "✅ No spam found right now."

def _h_unread_today(service):
    """
    Summarize INBOX unread emails received today (local tz).
    """
    # Prefer project tz; fall back to Django TIME_ZONE; then Asia/Kolkata
    tz_str = getattr(settings, "LOCAL_TZ", None) or getattr(settings, "TIME_ZONE", None) or "Asia/Kolkata"
    tz = ZoneInfo(tz_str)

    today = dt.datetime.now(tz).date()
    after_str = today.strftime("%Y/%m/%d")  # Gmail "after:" matches midnight of that date

    q = f"in:inbox is:unread after:{after_str}"
    msgs = _gmail_search(service, q, max_results=50)

    if not msgs:
        return "No unread emails received today."

    # Count by sender (human readable)
    counts: Dict[str, int] = {}
    for m in msgs:
        frm = _human_sender(m.get("from", ""))
        counts[frm] = counts.get(frm, 0) + 1
    top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:3]
    top_str = ", ".join([f"{n} ({c})" for n, c in top]) if top else "—"

    # Preview up to 5
    preview_lines = []
    for m in msgs[:5]:
        subj = m["subject"]
        frm = _human_sender(m["from"])
        snip = (m["snippet"] or "").replace("\n", " ")[:90]
        preview_lines.append(f"• {subj} — {frm} — {snip}")

    reply = (
        f"📬 {len(msgs)} unread email(s) today.\n"
        f"Top senders: {top_str}\n" +
        "\n".join(preview_lines) +
        "\n\nTry: “Mark the first two as read”, “Open the latest from <name>”, or “Draft a quick reply”."
    )
    return reply

def _h_draft_reply(text):
    m = re.search(r"^reply to (.+?)['’]?(s)? (?:email|mail) about (.+)$", text.strip(), flags=re.I)
    name = (m.group(1) if m else "there")
    topic = (m.group(3) if m else "your request")
    suggestion = (
        f"Here’s a suggested reply:\n\n"
        f"Dear {name},\n"
        f"Your {topic} will be processed shortly. Thank you for your patience.\n"
        f"Best regards,\n"
        f"[Your Name]"
    )
    return suggestion + "\n\nDo you want to send this or edit it?"

def _h_set_ooo(service):
    now = dt.datetime.now()
    start = int(now.timestamp())
    end = int((now + dt.timedelta(days=7)).timestamp())
    subject = "Out of office"
    body = (
        "I am currently out of office and will return on [Date]. "
        "For urgent matters, please contact [Colleague Email]."
    )
    try:
        _set_vacation_responder(service, True, start, end, subject, body)
        return f"✅ Auto-reply set: ‘{body}’"
    except HttpError as e:
        return f"⚠️ Could not set auto-reply ({e}). Please try again."

def _h_how_compose():
    return ("✉️ To compose a new email, click the Compose button (bottom right), enter the recipient’s address, "
            "add a subject and your message, then click Send. You can also add Cc/Bcc and attachments.")

def _h_fallback():
    return ("I didn’t catch that. Try: “Summarize unread emails from today”, “Show my inbox”, "
            "“Any spam mails?”, or “Draft a follow-up for the last client email.”")

@require_POST
def chatbot_ask(request):
    """
    Chatbot endpoint used by the bottom-right widget.
    Expects JSON: { "message": "..." } -> returns { "reply": "..." }
    """
    try:
        data = json.loads(request.body or "{}")
        user_text = _safe(data.get("message", "")).strip()
        if not user_text:
            return JsonResponse({"reply": "Please type a message."}, status=400)

        intent = _detect_chat_intent(user_text)

        # Intents that require Gmail access
        if intent in {"inbox_status", "spam_status", "set_ooo", "unread_today"}:
            service, redir = _ensure_service(request)
            if redir:
                return JsonResponse({"reply": "Please connect your Gmail account first (Login with Google)."}, status=401)

        if intent == "help":
            reply = _h_help()
        elif intent == "inbox_status":
            reply = _h_inbox(service)
        elif intent == "spam_status":
            reply = _h_spam(service)
        elif intent == "unread_today":
            reply = _h_unread_today(service)
        elif intent == "draft_reply":
            reply = _h_draft_reply(user_text)
        elif intent == "set_ooo":
            reply = _h_set_ooo(service)
        elif intent == "how_compose":
            reply = _h_how_compose()
        else:
            reply = _h_fallback()

        return JsonResponse({"reply": reply})
    except HttpError as e:
        return JsonResponse({"reply": f"⚠️ Gmail API error: {e}"}, status=500)
    except Exception as e:
        return JsonResponse({"reply": f"⚠️ Server error: {e}"}, status=500)
