from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass
class UserProfile:
    user_id: str
    do_not_disturb_window: str = ""      # e.g. "22:00-07:00"
    quiet_hours_start: str = ""          # parsed from do_not_disturb_window
    quiet_hours_end: str = ""            # parsed from do_not_disturb_window
    messages_opened_30d: int = 0
    messages_replied_30d: int = 0
    notifications_dismissed_30d: int = 0
    messages_reported_30d: int = 0
    # Derived rates (computed at load time from the 30d counts)
    open_rate: float = 0.5
    reply_rate: float = 0.2
    dismissal_rate: float = 0.3
    report_rate: float = 0.0

@dataclass
class GroupProfile:
    group_id: str
    group_name: str = ""
    group_type: str = "general"
    member_count: int = 1
    admin_count: int = 1
    messages_30d: int = 0
    # keep size as alias for member_count for backward compat with rule_engine
    @property
    def size(self) -> int:
        return self.member_count
    # admin_user_ids not in real dataset — resolved via group_members
    @property
    def admin_user_ids(self) -> List[str]:
        return []
    # activity_level derived from messages_30d
    @property
    def activity_level(self) -> str:
        if self.messages_30d >= 100:
            return "high"
        if self.messages_30d >= 20:
            return "medium"
        return "low"

@dataclass
class GroupMemberProfile:
    group_id: str
    user_id: str
    role: str = "member"
    joined_at: str = ""
    messages_sent_30d: int = 0
    messages_read_30d: int = 0
    replies_sent_30d: int = 0
    notifications_dismissed_30d: int = 0
    group_muted_by_user: bool = False
    # alias for backward compat
    @property
    def is_muted(self) -> bool:
        return self.group_muted_by_user
    @property
    def activity_level(self) -> str:
        if self.messages_sent_30d >= 20:
            return "high"
        if self.messages_sent_30d >= 5:
            return "medium"
        return "low"

@dataclass
class BusinessAccount:
    business_id: str
    display_name: str = ""
    brand_name: str = ""
    category: str = ""
    is_verified: bool = False           # mapped from 'verified' column
    official_domain: str = ""
    domain_used_by_sender: str = ""
    account_age_days: int = 0
    messages_sent_30d: int = 0
    user_reports_30d: int = 0
    domain_used_by_sender_age_days: int = 0
    # backward compat aliases
    @property
    def domain(self) -> str:
        return self.domain_used_by_sender or self.official_domain
    @property
    def report_count(self) -> int:
        return self.user_reports_30d
    # domain mismatch detection
    @property
    def domain_mismatch(self) -> bool:
        od = self.official_domain.strip().lower()
        ds = self.domain_used_by_sender.strip().lower()
        return bool(od and ds and od != ds)

@dataclass
class UserBusinessHistory:
    user_id: str
    business_id: str
    why_user_knows_account: str = ""
    last_activity_at: str = ""
    allows_promotions: bool = False
    promotions_opted_out_at: str = ""
    activity_count_180d: int = 0
    messages_opened_30d: int = 0
    messages_dismissed_30d: int = 0
    messages_replied_30d: int = 0
    last_reply_at: str = ""
    # backward compat aliases
    @property
    def orders_count(self) -> int:
        return self.activity_count_180d
    @property
    def bookings_count(self) -> int:
        return 0
    @property
    def payments_count(self) -> int:
        return 0
    @property
    def opt_in_status(self) -> str:
        if self.promotions_opted_out_at:
            return "opted_out"
        if self.allows_promotions:
            return "opted_in"
        return "none"

@dataclass
class MessageHistoryItem:
    message_id: str
    user_id: str
    sender_user_id: str = ""
    group_id: str = ""
    business_id: str = ""
    created_at: str = ""
    message_text: str = ""
    media_type: str = ""
    media_id: str = ""
    forwarded_count: int = 0
    # from message_events join
    message_opened: bool = False
    message_replied: bool = False
    reaction_time_minutes: float = 0.0
    notification_dismissed: bool = False
    muted_after_message: bool = False
    message_reported: bool = False
    # backward compat alias for retriever
    @property
    def action_taken(self) -> str:
        if self.message_reported:
            return "reported"
        if self.muted_after_message:
            return "muted"
        if self.message_replied:
            return "replied"
        if self.message_opened:
            return "opened"
        if self.notification_dismissed:
            return "dismissed"
        return "none"

@dataclass
class MediaItem:
    media_id: str
    media_type: str
    file_path: str = ""
    extracted_text: str = ""
    category: str = "unknown"
    duration_seconds: float = 0.0

@dataclass
class DailyNotificationSummary:
    user_id: str
    date: str = ""
    notifications_sent: int = 0
    notifications_dismissed: int = 0
    # backward compat aliases
    @property
    def total_notifications_received(self) -> int:
        return self.notifications_sent
    @property
    def high_priority_count(self) -> int:
        return max(0, self.notifications_sent - self.notifications_dismissed)
    @property
    def muted_count(self) -> int:
        return self.notifications_dismissed

@dataclass
class RawMessage:
    message_id: str
    user_id: str
    conversation_type: str
    group_id: str = ""
    business_id: str = ""
    sender_user_id: str = ""
    created_at: str = ""
    message_text: str = ""
    media_type: str = ""
    media_id: str = ""
    forwarded_count: int = 0

@dataclass
class Context:
    message: RawMessage
    user: Optional[UserProfile] = None
    group: Optional[GroupProfile] = None
    group_member: Optional[GroupMemberProfile] = None
    business: Optional[BusinessAccount] = None
    user_business_history: Optional[UserBusinessHistory] = None
    daily_summary: Optional[DailyNotificationSummary] = None
    media: Optional[MediaItem] = None
    relevant_history: List[MessageHistoryItem] = field(default_factory=list)
    # sender_is_group_admin is resolved by context_builder from group_members
    sender_is_group_admin: bool = False

@dataclass
class FeatureSet:
    # Dictionary storing all engineered signals
    features: Dict[str, Any] = field(default_factory=dict)

    # Backward compatible attributes for rules / reasoning_engine
    is_quiet_hours: bool = False
    user_dismissal_rate: float = 0.0
    user_report_rate: float = 0.0
    daily_notification_load: int = 0
    is_group_muted: bool = False
    is_user_group_admin: bool = False
    is_sender_group_admin: bool = False
    group_size: int = 0
    is_direct_mention: bool = False
    is_business_verified: bool = False
    business_report_count: int = 0
    is_opted_out: bool = False
    has_active_customer_relationship: bool = False
    has_urgency_keywords: bool = False
    has_payment_ask: bool = False
    has_otp: bool = False
    has_short_url: bool = False
    forwarded_magnitude: int = 0
    media_type: str = ""
    media_extracted_text: str = ""
    media_category: str = "unknown"
    history_matches_count: int = 0
    prior_reported_count: int = 0
    prior_opened_count: int = 0

@dataclass
class Decision:
    message_id: str
    action: str
    message_type: str
    reason: str
    confidence: float
    evidence_message_ids: str
