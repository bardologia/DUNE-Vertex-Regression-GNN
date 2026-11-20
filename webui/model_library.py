from __future__ import annotations

import json
import subprocess
import threading

from configuration.shared        import IoNames
from configuration.webui.general import LaunchSettings, ModelFamilies

from project_paths import ProjectPaths


class ModelLibrary:

    BOOTSTRAP = (
        "import dataclasses, json, sys\n"
        "repository_root = sys.argv[1]\n"
        "sys.path.insert(0, repository_root)\n"
        "from models import ModelFactory\n"
        "from configuration.architectures import ModelConfigRegistry\n"
        "def encode(value):\n"
        "    if isinstance(value, (list, tuple)):\n"
        "        return [encode(item) for item in value]\n"
        "    if isinstance(value, dict):\n"
        "        return {key: encode(item) for key, item in value.items()}\n"
        "    if isinstance(value, (bool, int, float, str)) or value is None:\n"
        "        return value\n"
        "    return str(value)\n"
        "models = []\n"
        "for name in ModelFactory.by_name():\n"
        "    factory = ModelConfigRegistry.by_name().get(name)\n"
        "    defaults = encode(dataclasses.asdict(factory())) if factory is not None else {}\n"
        "    models.append({'name': name, 'config_defaults': defaults})\n"
        "print(json.dumps({'models': models}))\n"
    )

    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths
        self.cache = None
        self.lock  = threading.Lock()

    def _signature(self) -> tuple:
        watched  = sorted((self.paths.repo_root / IoNames.models_dir).rglob(IoNames.python_glob))
        watched += sorted((self.paths.repo_root / IoNames.configuration_dir / IoNames.architectures_dir).rglob(IoNames.python_glob))

        stamps = []
        for path in watched:
            try:
                stamps.append((path.name, path.stat().st_mtime_ns))
            except OSError:
                continue
        return tuple(stamps)

    def _run_bootstrap(self, interpreter: str) -> dict:
        argv = [interpreter, "-c", self.BOOTSTRAP, str(self.paths.repo_root)]

        try:
            completed = subprocess.run(argv, cwd=str(self.paths.repo_root), capture_output=True, text=True, timeout=LaunchSettings.introspection_timeout_s)
        except (OSError, subprocess.TimeoutExpired) as error:
            return {"ok": False, "error": f"model introspection failed: {error}"}

        if completed.returncode != 0:
            tail = "\n".join(completed.stderr.strip().splitlines()[-LaunchSettings.introspection_error_lines:])
            return {"ok": False, "error": f"model introspection failed:\n{tail}"}

        try:
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return {"ok": False, "error": "model introspection produced no output"}

        return {"ok": True, "models": payload["models"]}

    def _capacity(self, defaults: dict) -> str:
        hidden = defaults.get("hidden_dim")
        layers = defaults.get("num_layers")
        if hidden is None or layers is None:
            return ""
        return f"{hidden} dim · {layers} layers"

    def _families(self, models: list[dict]) -> dict:
        defaults  = {model["name"]: model["config_defaults"] for model in models}
        described = {entry["key"] for family in ModelFamilies.families for entry in family["models"]}

        missing = sorted(set(defaults) - described)
        unknown = sorted(described - set(defaults))
        if missing or unknown:
            problems = []
            if missing:
                problems.append(f"models with no family entry: {', '.join(missing)}")
            if unknown:
                problems.append(f"family entries for unknown models: {', '.join(unknown)}")
            return {"ok": False, "error": "; ".join(problems)}

        families = []
        for family in ModelFamilies.families:
            entries = []
            for entry in family["models"]:
                entries.append({
                    "key"         : entry["key"],
                    "name"        : entry["name"],
                    "blurb"       : entry["blurb"],
                    "capacity"    : self._capacity(defaults[entry["key"]]),
                    "recommended" : bool(entry.get("recommended")),
                    "head_type"   : defaults[entry["key"]].get("head_type", ""),
                })
            families.append({"family": family["family"], "models": entries})

        return {"ok": True, "families": families}

    def list(self, interpreter: str) -> dict:
        signature = self._signature()

        with self.lock:
            if self.cache is not None and self.cache[0] == signature:
                return self.cache[1]

        introspected = self._run_bootstrap(interpreter)
        if not introspected.get("ok"):
            return introspected

        result = self._families(introspected["models"])
        if result.get("ok"):
            with self.lock:
                self.cache = (signature, result)
        return result
