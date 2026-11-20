from __future__ import annotations

import copy
from collections import Counter

from configuration.webui.general import LaunchForms


class LayoutError(Exception):
    pass


class LaunchLayout:

    def _expand(self, key: str) -> dict:
        spec    = dict(LaunchForms.layouts)[key]
        widgets = {}

        essentials = [self._field_entry(item, "", widgets) for item in spec.get("essentials", [])]
        sections   = [{"key": section["key"], "title": section["title"], "panels": [self._expand_panel(panel, widgets) for panel in section["panels"]]} for section in spec["sections"]]

        return {"essentials": essentials, "sections": sections, "widgets": widgets}

    def _expand_panel(self, panel, widgets: dict) -> dict:
        if panel["kind"] == "special":
            return {"kind": "special", "panel": panel["panel"], "fields": list(panel["fields"])}

        if panel["kind"] == "hidden":
            return {"kind": "hidden", "fields": list(panel["fields"])}

        groups = dict(LaunchForms.templates)[panel["template"]] if "template" in panel else panel["groups"]
        return {"kind": "fields", "title": panel["title"], "groups": self._expand_groups(groups, panel.get("at", ""), widgets)}

    def _expand_groups(self, groups, prefix: str, widgets: dict) -> list:
        return [{"title": group["title"], "fields": [self._field_entry(item, prefix, widgets) for item in group["fields"]]} for group in groups]

    def _field_entry(self, item, prefix: str, widgets: dict) -> dict:
        if isinstance(item, str):
            return {"path": self._join(prefix, item)}

        if "gate" in item:
            return {"gate": self._join(prefix, item["gate"]), "fields": [self._field_entry(sub, prefix, widgets) for sub in item["fields"]]}

        path = self._join(prefix, item["path"])
        if "widget" in item:
            widgets[path] = dict(item["widget"])
        return {"path": path}

    def _join(self, prefix: str, name: str) -> str:
        return f"{prefix}.{name}" if prefix else name

    def _validate(self, key: str, layout: dict, leaves: list) -> None:
        paths   = [leaf["path"] for leaf in leaves]
        claimed = Counter(self._claims(layout))

        duplicates = sorted(path for path, count in claimed.items() if count > 1)
        unknown    = sorted(set(claimed) - set(paths))
        unclaimed  = [path for path in paths if path not in claimed]

        problems = []
        if unknown:
            problems.append(f"layout for {key} names unknown fields: {', '.join(unknown)}")
        if duplicates:
            problems.append(f"layout for {key} claims fields twice: {', '.join(duplicates)}")
        if unclaimed:
            problems.append(f"layout for {key} leaves fields unclaimed: {', '.join(unclaimed)}")

        if problems:
            raise LayoutError("\n".join(problems))

    def _claims(self, layout: dict) -> list:
        claimed = []

        for entry in layout["essentials"]:
            self._entry_claims(entry, claimed)

        for section in layout["sections"]:
            for panel in section["panels"]:
                if panel["kind"] in ("special", "hidden"):
                    claimed.extend(panel["fields"])
                    continue

                for group in panel["groups"]:
                    for entry in group["fields"]:
                        self._entry_claims(entry, claimed)

        return claimed

    def _entry_claims(self, entry, claimed: list) -> None:
        if "gate" in entry:
            claimed.append(entry["gate"])
            for sub in entry["fields"]:
                self._entry_claims(sub, claimed)
            return

        claimed.append(entry["path"])

    def build(self, key: str, leaves: list) -> dict:
        if key not in dict(LaunchForms.layouts):
            raise LayoutError(f"no launch layout declared for {key}")

        layout = copy.deepcopy(self._expand(key))
        self._validate(key, layout, leaves)

        layout["mode"] = "single" if len(layout["sections"]) == 1 and not layout["essentials"] else "sections"
        return layout
