import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    # Directory Paths
    dataset_dir: str = "dataset"
    output_path: str = "dataset/output.csv"
    messages_path: str = "dataset/messages.csv"
    sample_messages_path: str = "dataset/sample_messages.csv"
    users_path: str = "dataset/users.csv"
    groups_path: str = "dataset/groups.csv"
    group_members_path: str = "dataset/group_members.csv"
    business_accounts_path: str = "dataset/business_accounts.csv"
    user_business_history_path: str = "dataset/user_business_history.csv"
    message_history_path: str = "dataset/message_history.csv"
    message_events_path: str = "dataset/message_events.csv"
    images_path: str = "dataset/images.csv"
    voice_notes_path: str = "dataset/voice_notes.csv"
    daily_summary_path: str = "dataset/daily_notification_summary.csv"
    media_dir: str = "dataset/media"
    
    # Model & Reasoning Settings
    default_model: str = "gemini-2.5-flash"
    temperature: float = 0.1
    max_retries: int = 3
    
    # Required Output Schema (exact order)
    required_output_columns: List[str] = field(
        default_factory=lambda: [
            "message_id",
            "action",
            "message_type",
            "reason",
            "confidence",
            "evidence_message_ids"
        ]
    )
    
    # Feature Thresholds
    high_forwarded_count_threshold: int = 5
    high_reports_threshold: int = 5
    quiet_hours_default_start: str = "22:00"
    quiet_hours_default_end: str = "07:00"

config = Config()
