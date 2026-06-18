from __future__ import annotations

import json
import mimetypes
from pathlib       import Path
from urllib.parse  import parse_qs, urlparse

from config_registry     import ConfigRegistry
from gpu_watchdog        import GpuWatchdog
from model_library       import ModelLibrary
from process_manager     import ProcessManager
from project_paths       import ProjectPaths
from results_browser     import ResultsBrowser
from script_catalog      import ScriptCatalog
from server_logger       import ServerLogger
from system_monitor      import SystemMonitor
from tensorboard_manager import TensorboardManager


class RequestRouter:

    PROJECT = {
        "name"    : "DUNE-GNN",
        "tagline" : "Graph neural network vertex reconstruction control console",
    }

    def __init__(self, paths: ProjectPaths, logger: ServerLogger, catalog: ScriptCatalog, configs: ConfigRegistry, processes: ProcessManager, system: SystemMonitor, gpu_watchdog: GpuWatchdog, tensorboard: TensorboardManager, results: ResultsBrowser, models: ModelLibrary) -> None:
        self.paths        = paths
        self.logger       = logger
        self.catalog      = catalog
        self.configs      = configs
        self.processes    = processes
        self.system       = system
        self.gpu_watchdog = gpu_watchdog
        self.tensorboard  = tensorboard
        self.results      = results
        self.models       = models

    def _route_get(self, handler, path: str) -> None:
        if path == "/" or path == "":
            self._serve_static(handler, "index.html")
            return
        if path.startswith("/static/"):
            self._serve_static(handler, path[len("/static/"):])
            return
        if path == "/api/system/status" or path == "/api/system":
            self._send_json(handler, self._system_payload())
            return
        if path == "/api/system/gpu":
            self._send_json(handler, self.gpu_watchdog.latest())
            return
        if path == "/api/interpreters":
            interpreters = self.paths.discover_interpreters()
            self._send_json(handler, {"interpreters": interpreters, "preferred": self.paths.preferred_interpreter(interpreters)})
            return
        if path == "/api/scripts":
            self._send_json(handler, {"scripts": self.catalog.list()})
            return
        if path.startswith("/api/scripts/") and path.endswith("/config"):
            key    = path[len("/api/scripts/"):-len("/config")]
            result = self.configs.schema(key, self.paths.preferred_interpreter())
            self._send_json(handler, result, 200 if result.get("ok") else 400)
            return
        if path == "/api/models":
            result = self.models.list(self.paths.preferred_interpreter())
            self._send_json(handler, result, 200 if result.get("ok") else 400)
            return
        if path == "/api/runs":
            self._send_json(handler, self.results.list_runs())
            return
        if path == "/api/runs/file":
            target = (parse_qs(urlparse(handler.path).query).get("path") or [""])[0]
            self._serve_run_file(handler, target)
            return
        if path == "/api/processes":
            self._send_json(handler, {"processes": self.processes.list()})
            return
        if path.startswith("/api/processes/") and path.endswith("/log"):
            identifier = path[len("/api/processes/"):-len("/log")]
            count      = int((parse_qs(urlparse(handler.path).query).get("lines") or ["200"])[0])
            result     = self._with_pid(identifier, lambda pid: self.processes.tail(pid, count))
            self._send_json(handler, result, 200 if result.get("ok") else 400)
            return
        if path == "/api/tensorboard":
            self._send_json(handler, {"instances": self.tensorboard.list()})
            return

        self._send_json(handler, {"error": "not found"}, 404)

    def _route_post(self, handler, path: str) -> None:
        body = self._read_json(handler)

        if path == "/api/launch":
            script      = body.get("script", "")
            overrides   = body.get("overrides", {})
            interpreter = body.get("interpreter") or self.paths.preferred_interpreter()
            result      = self.processes.launch(script, overrides, interpreter)
            self._send_json(handler, result, 200 if result.get("ok") else 400)
            return

        if path.startswith("/api/processes/") and path.endswith("/kill"):
            identifier = path[len("/api/processes/"):-len("/kill")]
            result     = self._with_pid(identifier, self.processes.kill)
            self._send_json(handler, result, 200 if result.get("ok") else 400)
            return

        if path == "/api/tensorboard":
            result = self.tensorboard.launch(body.get("run", ""))
            self._send_json(handler, result, 200 if result.get("ok") else 400)
            return

        if path.startswith("/api/tensorboard/") and path.endswith("/stop"):
            identifier = path[len("/api/tensorboard/"):-len("/stop")]
            result     = self._with_pid(identifier, self.tensorboard.stop)
            self._send_json(handler, result, 200 if result.get("ok") else 400)
            return

        self._send_json(handler, {"error": "not found"}, 404)

    def _system_payload(self) -> dict:
        payload   = self.system.status()
        watchdog  = self.gpu_watchdog.latest()
        payload["watchdog"] = {
            "armed"     : bool(watchdog.get("armed")),
            "interval"  : self.gpu_watchdog.INTERVAL_SECONDS,
            "gpu_count" : len(payload.get("gpus", [])),
            "updated"   : watchdog.get("updated"),
        }
        return payload

    def _serve_run_file(self, handler, raw_path: str) -> None:
        runs_root = self.paths.runs_dir.resolve()
        target    = (runs_root / raw_path) if not Path(raw_path).is_absolute() else Path(raw_path)
        target    = target.resolve()

        if not target.is_relative_to(runs_root) or not target.is_file():
            self._send_json(handler, {"error": "not found"}, 404)
            return

        data         = target.read_bytes()
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"

        handler.send_response(200)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(data)))
        handler.send_header("Cache-Control", "max-age=300")
        handler.end_headers()
        handler.wfile.write(data)

    def _with_pid(self, identifier: str, action):
        try:
            pid = int(identifier)
        except ValueError:
            return {"ok": False, "error": "invalid pid"}
        return action(pid)

    def _read_json(self, handler) -> dict:
        length = int(handler.headers.get("Content-Length", 0) or 0)
        if length <= 0:
            return {}
        raw = handler.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}

    def _send_json(self, handler, payload: dict, status: int = 200) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(encoded)))
        handler.end_headers()
        handler.wfile.write(encoded)

    def _serve_static(self, handler, relative: str) -> None:
        target = (self.paths.static_dir / relative).resolve()
        if not target.is_relative_to(self.paths.static_dir.resolve()):
            self._send_json(handler, {"error": "forbidden"}, 403)
            return
        if not target.is_file():
            self._send_json(handler, {"error": "not found"}, 404)
            return

        data         = target.read_bytes()
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"

        handler.send_response(200)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(data)))
        handler.send_header("Cache-Control", "no-cache")
        handler.end_headers()
        handler.wfile.write(data)

    def route(self, handler) -> None:
        parsed = urlparse(handler.path)
        path   = parsed.path.rstrip("/") or "/"
        method = handler.command

        try:
            if method == "GET":
                self._route_get(handler, path)
            elif method == "POST":
                self._route_post(handler, path)
            else:
                self._send_json(handler, {"error": "method not allowed"}, 405)
        except BrokenPipeError:
            pass
        except Exception as error:
            self.logger.error(f"router error on {method} {path}: {error}")
            try:
                self._send_json(handler, {"error": str(error)}, 500)
            except Exception:
                pass
