from __future__ import annotations

from datetime import datetime

from configuration.shared import IoNames, LogSettings


class RunTag:

    @staticmethod
    def now() -> str:
        return datetime.now().strftime(IoNames.run_tag_format)

    @staticmethod
    def timestamp() -> str:
        return datetime.now().strftime(LogSettings.timestamp_format)
