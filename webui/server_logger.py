from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.monitoring.logger import Logger


class ServerLogger:

    def __init__(self) -> None:
        self.logger = Logger(log_dir=str(REPOSITORY_ROOT / "logs" / "webui"), name="webui")

    def info(self, message: str) -> None:
        self.logger.info(message)

    def ok(self, message: str) -> None:
        self.logger.ok(message)

    def warning(self, message: str) -> None:
        self.logger.warning(message)

    def error(self, message: str) -> None:
        self.logger.error(message)

    def section(self, title: str) -> None:
        self.logger.section(title)

    def banner(self, title: str, lines: list[str]) -> None:
        self.logger.section(title)
        for line in lines:
            self.logger.info(line)
