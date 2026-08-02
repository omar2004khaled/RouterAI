import csv
import logging
import os
from typing import List
from config import config
from models import Decision

logger = logging.getLogger(__name__)


class CSVWriter:
    def __init__(self, output_path: str = config.output_path):
        self.output_path = output_path

    def write_decisions(self, decisions: List[Decision]):
        cols = config.required_output_columns
        logger.info(f"Writing {len(decisions)} predictions to {self.output_path}...")

        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        with open(self.output_path, "w", newline="", encoding="utf-8", errors="replace") as f:
            writer = csv.writer(f)
            writer.writerow(cols)
            for d in decisions:
                writer.writerow([
                    d.message_id,
                    d.action,
                    d.message_type,
                    d.reason,
                    f"{d.confidence:.2f}",
                    d.evidence_message_ids
                ])
        logger.info(f"Successfully saved {self.output_path}.")
