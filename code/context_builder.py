import os
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from config import config
from models import (
    RawMessage, UserProfile, GroupProfile, GroupMemberProfile,
    BusinessAccount, UserBusinessHistory, MessageHistoryItem,
    MediaItem, DailyNotificationSummary, Context
)

logger = logging.getLogger(__name__)


def _read_csv_safely(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        logger.warning(f"Dataset file not found: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str, encoding="utf-8-sig",
                           encoding_errors="replace", on_bad_lines="skip").fillna("")
    except Exception as exc:
        logger.warning(f"Falling back to tolerant CSV loading for {path}: {exc}")
        try:
            return pd.read_csv(path, dtype=str, encoding="utf-8-sig",
                               encoding_errors="replace", on_bad_lines="skip",
                               engine="python").fillna("")
        except Exception:
            return pd.DataFrame()


def safe_float(val, default: float = 0.0) -> float:
    try:
        if pd.isna(val) or val == "":
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


def safe_int(val, default: int = 0) -> int:
    try:
        if pd.isna(val) or val == "":
            return default
        return int(float(val))
    except (ValueError, TypeError):
        return default


def safe_str(val, default: str = "") -> str:
    if pd.isna(val) or val is None:
        return default
    return str(val).strip()


def safe_bool(val, default: bool = False) -> bool:
    if pd.isna(val) or val is None:
        return default
    s = str(val).strip().lower()
    return s in ("true", "1", "yes", "t")


def _parse_dnd_window(window: str) -> Tuple[str, str]:
    """Parse '22:00-07:00' → ('22:00', '07:00'). Returns ('', '') on failure."""
    if not window or "-" not in window:
        return "", ""
    parts = window.split("-", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return "", ""


class ContextBuilder:
    def __init__(self, dataset_dir: str = config.dataset_dir):
        self.dataset_dir = dataset_dir
        self.users: Dict[str, UserProfile] = {}
        self.groups: Dict[str, GroupProfile] = {}
        # (group_id, user_id) → GroupMemberProfile
        self.group_members: Dict[Tuple[str, str], GroupMemberProfile] = {}
        # group_id → set of admin user_ids (resolved from group_members)
        self.group_admins: Dict[str, set] = {}
        self.businesses: Dict[str, BusinessAccount] = {}
        self.user_business_history: Dict[Tuple[str, str], UserBusinessHistory] = {}
        self.daily_summaries: Dict[str, DailyNotificationSummary] = {}
        self.images: Dict[str, MediaItem] = {}
        self.voice_notes: Dict[str, MediaItem] = {}
        self.message_history: List[MessageHistoryItem] = []
        # message_id → event row  (from message_events.csv)
        self.history_events: Dict[str, dict] = {}

        self.load_all()

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def load_all(self):
        logger.info(f"Loading relational tables from {self.dataset_dir}...")
        self._load_users()
        self._load_groups()
        self._load_group_members()
        self._load_businesses()
        self._load_user_business_history()
        self._load_daily_summaries()
        self._load_images()
        self._load_voice_notes()
        self._load_message_events()
        self._load_message_history()

    def _load_users(self):
        """
        Real columns: user_id, do_not_disturb_window,
          messages_opened_30d, messages_replied_30d,
          notifications_dismissed_30d, messages_reported_30d
        """
        path = os.path.join(self.dataset_dir, "users.csv")
        df = _read_csv_safely(path)
        if df.empty:
            return
        for _, row in df.iterrows():
            uid = safe_str(row.get("user_id"))
            if not uid:
                continue
            dnd = safe_str(row.get("do_not_disturb_window"))
            qs, qe = _parse_dnd_window(dnd)

            opened = safe_int(row.get("messages_opened_30d"), 0)
            replied = safe_int(row.get("messages_replied_30d"), 0)
            dismissed = safe_int(row.get("notifications_dismissed_30d"), 0)
            reported = safe_int(row.get("messages_reported_30d"), 0)
            total = max(1, opened + dismissed + reported)

            self.users[uid] = UserProfile(
                user_id=uid,
                do_not_disturb_window=dnd,
                quiet_hours_start=qs,
                quiet_hours_end=qe,
                messages_opened_30d=opened,
                messages_replied_30d=replied,
                notifications_dismissed_30d=dismissed,
                messages_reported_30d=reported,
                open_rate=round(opened / total, 3),
                reply_rate=round(replied / max(1, opened), 3),
                dismissal_rate=round(dismissed / total, 3),
                report_rate=round(reported / total, 3),
            )

    def _load_groups(self):
        """
        Real columns: group_id, group_name, group_type,
          member_count, admin_count, created_at, messages_30d
        """
        path = os.path.join(self.dataset_dir, "groups.csv")
        df = _read_csv_safely(path)
        if df.empty:
            return
        for _, row in df.iterrows():
            gid = safe_str(row.get("group_id"))
            if not gid:
                continue
            self.groups[gid] = GroupProfile(
                group_id=gid,
                group_name=safe_str(row.get("group_name")),
                group_type=safe_str(row.get("group_type"), "general"),
                member_count=safe_int(row.get("member_count"), 1),
                admin_count=safe_int(row.get("admin_count"), 1),
                messages_30d=safe_int(row.get("messages_30d"), 0),
            )

    def _load_group_members(self):
        """
        Real columns: group_id, user_id, role, joined_at,
          messages_sent_30d, messages_read_30d, replies_sent_30d,
          notifications_dismissed_30d, group_muted_by_user
        """
        path = os.path.join(self.dataset_dir, "group_members.csv")
        df = _read_csv_safely(path)
        if df.empty:
            return
        for _, row in df.iterrows():
            gid = safe_str(row.get("group_id"))
            uid = safe_str(row.get("user_id"))
            if not gid or not uid:
                continue
            role = safe_str(row.get("role"), "member")
            gm = GroupMemberProfile(
                group_id=gid,
                user_id=uid,
                role=role,
                joined_at=safe_str(row.get("joined_at")),
                messages_sent_30d=safe_int(row.get("messages_sent_30d"), 0),
                messages_read_30d=safe_int(row.get("messages_read_30d"), 0),
                replies_sent_30d=safe_int(row.get("replies_sent_30d"), 0),
                notifications_dismissed_30d=safe_int(row.get("notifications_dismissed_30d"), 0),
                group_muted_by_user=safe_bool(row.get("group_muted_by_user")),
            )
            self.group_members[(gid, uid)] = gm
            # Track admins for sender-is-admin lookups
            if role == "admin":
                self.group_admins.setdefault(gid, set()).add(uid)

    def _load_businesses(self):
        """
        Real columns: business_id, display_name, brand_name, category,
          verified, official_domain, domain_used_by_sender,
          account_age_days, messages_sent_30d, user_reports_30d,
          domain_used_by_sender_age_days
        """
        path = os.path.join(self.dataset_dir, "business_accounts.csv")
        df = _read_csv_safely(path)
        if df.empty:
            return
        for _, row in df.iterrows():
            bid = safe_str(row.get("business_id"))
            if not bid:
                continue
            self.businesses[bid] = BusinessAccount(
                business_id=bid,
                display_name=safe_str(row.get("display_name")),
                brand_name=safe_str(row.get("brand_name")),
                category=safe_str(row.get("category")),
                is_verified=safe_bool(row.get("verified")),
                official_domain=safe_str(row.get("official_domain")),
                domain_used_by_sender=safe_str(row.get("domain_used_by_sender")),
                account_age_days=safe_int(row.get("account_age_days"), 0),
                messages_sent_30d=safe_int(row.get("messages_sent_30d"), 0),
                user_reports_30d=safe_int(row.get("user_reports_30d"), 0),
                domain_used_by_sender_age_days=safe_int(row.get("domain_used_by_sender_age_days"), 0),
            )

    def _load_user_business_history(self):
        """
        Real columns: user_id, business_id, why_user_knows_account,
          last_activity_at, allows_promotions, promotions_opted_out_at,
          activity_count_180d, messages_opened_30d, messages_dismissed_30d,
          messages_replied_30d, last_reply_at
        """
        path = os.path.join(self.dataset_dir, "user_business_history.csv")
        df = _read_csv_safely(path)
        if df.empty:
            return
        for _, row in df.iterrows():
            uid = safe_str(row.get("user_id"))
            bid = safe_str(row.get("business_id"))
            if not uid or not bid:
                continue
            self.user_business_history[(uid, bid)] = UserBusinessHistory(
                user_id=uid,
                business_id=bid,
                why_user_knows_account=safe_str(row.get("why_user_knows_account")),
                last_activity_at=safe_str(row.get("last_activity_at")),
                allows_promotions=safe_bool(row.get("allows_promotions")),
                promotions_opted_out_at=safe_str(row.get("promotions_opted_out_at")),
                activity_count_180d=safe_int(row.get("activity_count_180d"), 0),
                messages_opened_30d=safe_int(row.get("messages_opened_30d"), 0),
                messages_dismissed_30d=safe_int(row.get("messages_dismissed_30d"), 0),
                messages_replied_30d=safe_int(row.get("messages_replied_30d"), 0),
                last_reply_at=safe_str(row.get("last_reply_at")),
            )

    def _load_daily_summaries(self):
        """
        Real columns: user_id, date, notifications_sent, notifications_dismissed
        """
        path = os.path.join(self.dataset_dir, "daily_notification_summary.csv")
        df = _read_csv_safely(path)
        if df.empty:
            return
        # Keep the most recent row per user_id
        for _, row in df.iterrows():
            uid = safe_str(row.get("user_id"))
            if not uid:
                continue
            existing = self.daily_summaries.get(uid)
            date = safe_str(row.get("date"))
            if existing and existing.date >= date:
                continue
            self.daily_summaries[uid] = DailyNotificationSummary(
                user_id=uid,
                date=date,
                notifications_sent=safe_int(row.get("notifications_sent"), 0),
                notifications_dismissed=safe_int(row.get("notifications_dismissed"), 0),
            )

    def _load_images(self):
        """
        Real columns: image_id, file_path
        (no ocr_text in the real dataset — media must be processed live)
        """
        path = os.path.join(self.dataset_dir, "images.csv")
        df = _read_csv_safely(path)
        if df.empty:
            return
        for _, row in df.iterrows():
            iid = safe_str(row.get("image_id"))
            if iid:
                fp = safe_str(row.get("file_path"))
                # Resolve relative path to the dataset directory
                if fp and not os.path.isabs(fp):
                    fp = os.path.join(self.dataset_dir, fp)
                self.images[iid] = MediaItem(
                    media_id=iid,
                    media_type="image",
                    file_path=fp,
                    extracted_text="",  # populated by media_processor at runtime
                    category="unknown",
                )

    def _load_voice_notes(self):
        """
        Real columns: voice_note_id, file_path
        """
        path = os.path.join(self.dataset_dir, "voice_notes.csv")
        df = _read_csv_safely(path)
        if df.empty:
            return
        for _, row in df.iterrows():
            vid = safe_str(row.get("voice_note_id"))
            if vid:
                fp = safe_str(row.get("file_path"))
                if fp and not os.path.isabs(fp):
                    fp = os.path.join(self.dataset_dir, fp)
                self.voice_notes[vid] = MediaItem(
                    media_id=vid,
                    media_type="voice",
                    file_path=fp,
                    extracted_text="",
                    category="unknown",
                )

    def _load_message_events(self):
        """
        Real columns: user_id, message_id, message_opened, message_replied,
          reaction_time_minutes, notification_dismissed, muted_after_message,
          message_reported
        """
        path = os.path.join(self.dataset_dir, "message_events.csv")
        df = _read_csv_safely(path)
        if df.empty:
            return
        for _, row in df.iterrows():
            mid = safe_str(row.get("message_id"))
            if mid:
                self.history_events[mid] = {
                    "message_opened": safe_bool(row.get("message_opened")),
                    "message_replied": safe_bool(row.get("message_replied")),
                    "reaction_time_minutes": safe_float(row.get("reaction_time_minutes"), 0.0),
                    "notification_dismissed": safe_bool(row.get("notification_dismissed")),
                    "muted_after_message": safe_bool(row.get("muted_after_message")),
                    "message_reported": safe_bool(row.get("message_reported")),
                }

    def _load_message_history(self):
        """
        Real columns: same schema as messages.csv
          (message_id, user_id, conversation_type, group_id, business_id,
           sender_user_id, created_at, message_text, media_type, media_id,
           forwarded_count)
        Joined with message_events on message_id.
        """
        path = os.path.join(self.dataset_dir, "message_history.csv")
        df = _read_csv_safely(path)
        if df.empty:
            return
        for _, row in df.iterrows():
            mid = safe_str(row.get("message_id"))
            if not mid:
                continue
            ev = self.history_events.get(mid, {})
            self.message_history.append(MessageHistoryItem(
                message_id=mid,
                user_id=safe_str(row.get("user_id")),
                sender_user_id=safe_str(row.get("sender_user_id")),
                group_id=safe_str(row.get("group_id")),
                business_id=safe_str(row.get("business_id")),
                created_at=safe_str(row.get("created_at")),
                message_text=safe_str(row.get("message_text")),
                media_type=safe_str(row.get("media_type")),
                media_id=safe_str(row.get("media_id")),
                forwarded_count=safe_int(row.get("forwarded_count"), 0),
                message_opened=ev.get("message_opened", False),
                message_replied=ev.get("message_replied", False),
                reaction_time_minutes=ev.get("reaction_time_minutes", 0.0),
                notification_dismissed=ev.get("notification_dismissed", False),
                muted_after_message=ev.get("muted_after_message", False),
                message_reported=ev.get("message_reported", False),
            ))

    # ------------------------------------------------------------------
    # Context assembly
    # ------------------------------------------------------------------

    def build_context(self, message: RawMessage) -> Context:
        user = self.users.get(message.user_id)
        group = self.groups.get(message.group_id) if message.group_id else None
        group_member = self.group_members.get((message.group_id, message.user_id)) if message.group_id else None
        business = self.businesses.get(message.business_id) if message.business_id else None
        user_biz = self.user_business_history.get((message.user_id, message.business_id)) if message.business_id else None
        daily = self.daily_summaries.get(message.user_id)

        # Resolve whether the sender is a group admin
        sender_is_group_admin = False
        if message.group_id and message.sender_user_id:
            sender_is_group_admin = message.sender_user_id in self.group_admins.get(message.group_id, set())

        # Media — file_path resolved; extracted_text populated by media_processor at runtime
        media = None
        if message.media_type == "image" and message.media_id in self.images:
            media = self.images[message.media_id]
        elif message.media_type == "voice" and message.media_id in self.voice_notes:
            media = self.voice_notes[message.media_id]

        # Relevant history: messages this user received that share the same
        # sender, group, or business as the incoming message
        relevant_hist: List[MessageHistoryItem] = []
        for h in self.message_history:
            if h.user_id != message.user_id:
                continue
            if message.business_id and h.business_id == message.business_id:
                relevant_hist.append(h)
            elif message.sender_user_id and h.sender_user_id == message.sender_user_id:
                relevant_hist.append(h)
            elif message.group_id and h.group_id == message.group_id:
                relevant_hist.append(h)

        return Context(
            message=message,
            user=user,
            group=group,
            group_member=group_member,
            business=business,
            user_business_history=user_biz,
            daily_summary=daily,
            media=media,
            relevant_history=relevant_hist,
            sender_is_group_admin=sender_is_group_admin,
        )
