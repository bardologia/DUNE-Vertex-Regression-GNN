from __future__ import annotations

import json
from pathlib import Path

from tools.runtime.run_tag import RunTag


class CompletionMarker:

    FILENAME = "complete.json"

    @staticmethod
    def path(directory) -> Path:
        return Path(directory) / CompletionMarker.FILENAME

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
        path.write_text(json.dumps({"completed_at": RunTag.timestamp(), **payload}, indent=2), encoding="utf-8")
        return path
