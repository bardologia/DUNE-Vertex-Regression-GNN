from __future__ import annotations

import json
import subprocess
import threading

from project_paths import ProjectPaths


class ConfigRegistry:

    BOOTSTRAP = (
        "import ast, inspect, json, sys, textwrap\n"
        "from dataclasses import fields, is_dataclass\n"
        "from pathlib import Path\n"
        "repository_root = sys.argv[1]\n"
        "sys.path.insert(0, repository_root)\n"
        "from tools.runtime.config_cli import ConfigCli, _SUPPORTED_TYPES\n"
        "module = __import__(sys.argv[2], fromlist=[sys.argv[3]])\n"
        "configuration = getattr(module, sys.argv[3])()\n"
        "block_cache = {}\n"
        "def blocks_for(cls):\n"
        "    if cls in block_cache:\n"
        "        return block_cache[cls]\n"
        "    mapping = {}\n"
        "    try:\n"
        "        body  = ast.parse(textwrap.dedent(inspect.getsource(cls))).body[0].body\n"
        "        group = 0\n"
        "        previous = None\n"
        "        for item in body:\n"
        "            if not isinstance(item, ast.AnnAssign) or not isinstance(item.target, ast.Name):\n"
        "                continue\n"
        "            if previous is not None and item.lineno - previous > 1:\n"
        "                group += 1\n"
        "            previous = item.end_lineno\n"
        "            mapping[item.target.id] = group\n"
        "    except Exception:\n"
        "        pass\n"
        "    block_cache[cls] = mapping\n"
        "    return mapping\n"
        "def walk(node, prefix, section):\n"
        "    blocks = blocks_for(type(node))\n"
        "    for field_definition in fields(node):\n"
        "        value = getattr(node, field_definition.name)\n"
        "        path  = prefix + field_definition.name\n"
        "        if is_dataclass(value):\n"
        "            yield from walk(value, path + '.', path)\n"
        "        else:\n"
        "            yield path, value, section, blocks.get(field_definition.name, 0)\n"
        "leaves = []\n"
        "for path, value, section, block in walk(configuration, '', ''):\n"
        "    editable = value is None or isinstance(value, _SUPPORTED_TYPES)\n"
        "    if isinstance(value, Path):\n"
        "        rendered = str(value)\n"
        "    elif isinstance(value, (list, tuple)):\n"
        "        rendered = str(list(value))\n"
        "    elif value is None:\n"
        "        rendered = 'None'\n"
        "    else:\n"
        "        rendered = str(value)\n"
        "    kind = 'none' if value is None else type(value).__name__\n"
        "    leaves.append({'path': path, 'value': rendered, 'type': kind, 'editable': editable, 'section': section, 'block': block})\n"
        "print(json.dumps({'class': sys.argv[3], 'leaves': leaves}))\n"
    )

    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths
        self.cache = {}
        self.lock  = threading.Lock()

    def _entry(self, script_key: str) -> dict | None:
        entry = self.paths.script_entry(script_key)
        if entry is None or not entry["has_config"]:
            return None
        return entry

    def _signature(self) -> tuple:
        watched = sorted((self.paths.repo_root / "configuration").rglob("*.py"))
        watched.append(self.paths.repo_root / "tools" / "runtime" / "config_cli.py")

        stamps = []
        for path in watched:
            try:
                stamps.append((path.name, path.stat().st_mtime_ns))
            except OSError:
                continue
        return tuple(stamps)

    def _run_bootstrap(self, entry: dict, interpreter: str) -> dict:
        argv = [interpreter, "-c", self.BOOTSTRAP, str(self.paths.repo_root), entry["entry_config_module"], entry["entry_config_class"]]

        try:
            completed = subprocess.run(argv, cwd=str(self.paths.repo_root), capture_output=True, text=True, timeout=180)
        except (OSError, subprocess.TimeoutExpired) as error:
            return {"ok": False, "error": f"config introspection failed: {error}"}

        if completed.returncode != 0:
            tail = "\n".join(completed.stderr.strip().splitlines()[-4:])
            return {"ok": False, "error": f"config introspection failed:\n{tail}"}

        try:
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return {"ok": False, "error": "config introspection produced no output"}

        return {"ok": True, "config_class": payload["class"], "leaves": payload["leaves"]}

    def schema(self, script_key: str, interpreter: str) -> dict:
        entry = self._entry(script_key)
        if entry is None:
            return {"ok": False, "error": "unknown or configuration-free script"}

        signature = self._signature()
        cache_key = (script_key, interpreter)

        with self.lock:
            cached = self.cache.get(cache_key)
            if cached is not None and cached[0] == signature:
                return cached[1]

        result = self._run_bootstrap(entry, interpreter)
        if result.get("ok"):
            with self.lock:
                self.cache[cache_key] = (signature, result)
        return result
