from __future__ import annotations

from configuration.webui.general import ScriptRegistry

from project_paths import ProjectPaths


class ScriptCatalog:

    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths

    def get(self, key: str) -> dict | None:
        entry = self.paths.script_entry(key)
        if entry is None or not entry["path"].exists():
            return None

        metadata = dict(ScriptRegistry.metadata).get(key, {"label": key, "category": "Other", "purpose": ""})

        return {
            "key"          : key,
            "file"         : entry["relative"],
            "label"        : metadata["label"],
            "category"     : metadata["category"],
            "purpose"      : metadata["purpose"],
            "has_config"   : entry["has_config"],
            "config_class" : entry["entry_config_class"],
        }

    def source(self, key: str) -> dict:
        entry = self.paths.script_entry(key)
        if entry is None or not entry["path"].exists():
            return {"ok": False, "error": f"unknown script '{key}'"}

        return {"ok": True, "file": entry["relative"], "source": entry["path"].read_text(encoding="utf-8")}

    def list(self) -> list[dict]:
        scripts = []
        for key in ScriptRegistry.order:
            description = self.get(key)
            if description is not None:
                scripts.append(description)
        return scripts
