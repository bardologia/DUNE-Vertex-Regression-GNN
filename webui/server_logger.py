from __future__ import annotations

from pathlib import Path

from tools.monitoring.logger import Logger

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


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
