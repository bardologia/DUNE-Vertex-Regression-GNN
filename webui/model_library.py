from __future__ import annotations

import json
import subprocess
import threading

from project_paths import ProjectPaths


class ModelLibrary:

    BOOTSTRAP = (
        "import dataclasses, json, sys\n"
        "repository_root = sys.argv[1]\n"
        "sys.path.insert(0, repository_root)\n"
        "from models import MODEL_REGISTRY\n"
        "from configuration.architectures import MODEL_CONFIG_REGISTRY\n"
        "def encode(value):\n"
        "    if isinstance(value, (list, tuple)):\n"
        "        return [encode(item) for item in value]\n"
        "    if isinstance(value, dict):\n"
        "        return {key: encode(item) for key, item in value.items()}\n"
        "    if isinstance(value, (bool, int, float, str)) or value is None:\n"
        "        return value\n"
        "    return str(value)\n"
        "models = []\n"
        "for name in MODEL_REGISTRY:\n"
        "    factory = MODEL_CONFIG_REGISTRY.get(name)\n"
        "    defaults = encode(dataclasses.asdict(factory())) if factory is not None else {}\n"
        "    models.append({'name': name, 'config_defaults': defaults})\n"
        "print(json.dumps({'models': models}))\n"
    )

    def __init__(self, paths: ProjectPaths) -> None:
        self.paths  = paths
        self.cache  = None
        self.lock   = threading.Lock()

    def _run_bootstrap(self, interpreter: str) -> dict:
        argv = [interpreter, "-c", self.BOOTSTRAP, str(self.paths.repo_root)]

        try:
            completed = subprocess.run(argv, cwd=str(self.paths.repo_root), capture_output=True, text=True, timeout=180)
        except (OSError, subprocess.TimeoutExpired) as error:
            return {"ok": False, "error": f"model introspection failed: {error}"}

        if completed.returncode != 0:
            tail = "\n".join(completed.stderr.strip().splitlines()[-4:])
            return {"ok": False, "error": f"model introspection failed:\n{tail}"}

        try:
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return {"ok": False, "error": "model introspection produced no output"}

        return {"ok": True, "models": payload["models"]}

    def list(self, interpreter: str) -> dict:
        with self.lock:
            if self.cache is not None:
                return self.cache

        result = self._run_bootstrap(interpreter)
        if result.get("ok"):
            with self.lock:
                self.cache = result
        return result
