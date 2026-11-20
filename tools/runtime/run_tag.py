from __future__ import annotations

import json
from datetime import datetime
from pathlib  import Path

from configuration.shared import IoNames, LogSettings


class RunTag:

    @staticmethod
    def now() -> str:
        return datetime.now().strftime(IoNames.run_tag_format)

    @staticmethod
    def timestamp() -> str:
        return datetime.now().strftime(LogSettings.timestamp_format)


class CompletionMarker:

    @staticmethod
    def path(directory) -> Path:
        return Path(directory) / IoNames.completion_file

    @staticmethod
    def is_complete(directory) -> bool:
        return CompletionMarker.path(directory).is_file()

    @staticmethod
    def clear(directory) -> None:
        CompletionMarker.path(directory).unlink(missing_ok=True)

    @staticmethod
    def read(directory) -> dict:
        return json.loads(CompletionMarker.path(directory).read_text(encoding="utf-8"))

    @staticmethod
    def stamp(directory, payload: dict) -> Path:
        path = CompletionMarker.path(directory)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"completed_at": RunTag.timestamp(), **payload}, indent=IoNames.json_indent), encoding="utf-8")
        return path
