from __future__ import annotations

import http.client
import json
import mimetypes
import queue
import threading
from datetime      import datetime
from pathlib       import Path
from urllib.parse  import parse_qs, urlparse

from pipelines.shared.run_paths import RunPaths

from config_registry     import ConfigRegistry
from event_explorer      import EventExplorer
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

    def __init__(self, paths: ProjectPaths, logger: ServerLogger, catalog: ScriptCatalog, configs: ConfigRegistry, processes: ProcessManager, system: SystemMonitor, gpu_watchdog: GpuWatchdog, tensorboard: TensorboardManager, results: ResultsBrowser, models: ModelLibrary, events: EventExplorer) -> None:
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
        self.events       = events

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
        if path.startswith("/api/processes/") and path.endswith("/stream"):
            identifier = path[len("/api/processes/"):-len("/stream")]
            self._stream_process(handler, identifier)
            return
        if path == "/api/tensorboard":
            self._send_json(handler, {"instances": self.tensorboard.list_instances()})
            return
        if path == "/api/events/runs":
            self._send_json(handler, self.events.list_runs())
            return
        if path == "/api/events/status":
            self._send_json(handler, self.events.load_status())
            return
        if path == "/api/events/list":
            query = parse_qs(urlparse(handler.path).query)
            result = self.events.events((query.get("run") or [""])[0], (query.get("split") or [""])[0])
            self._send_json(handler, result, 200 if result.get("ok") else 400)
            return
        if path == "/api/events/detail":
            query = parse_qs(urlparse(handler.path).query)
            try:
                index = int((query.get("index") or ["-1"])[0])
            except ValueError:
                index = -1
            result = self.events.detail((query.get("run") or [""])[0], (query.get("split") or [""])[0], index)
            self._send_json(handler, result, 200 if result.get("ok") else 400)
            return

        self._send_json(handler, {"error": "not found"}, 404)

    def _route_post(self, handler, path: str) -> None:
        body = self._read_json(handler)

        if path == "/api/launch":
            script      = body.get("script", "")
            overrides   = dict(body.get("overrides", {}))
            interpreter = body.get("interpreter") or self.paths.preferred_interpreter()

            if script == "train" and not overrides.get("run_name"):
                overrides["run_name"] = datetime.now().strftime("%Y%m%d-%H%M%S")

            result = self.processes.launch(script, overrides, interpreter)

            if result.get("ok") and script == "train":
                logdir = self._scoped_train_logdir(overrides, interpreter)
                threading.Thread(target=self.tensorboard.ensure, args=(logdir,), daemon=True).start()

            self._send_json(handler, result, 200 if result.get("ok") else 400)
            return

        if path.startswith("/api/processes/") and path.endswith("/kill"):
            identifier = path[len("/api/processes/"):-len("/kill")]
            result     = self._with_pid(identifier, self.processes.kill)
            self._send_json(handler, result, 200 if result.get("ok") else 400)
            return

        if path == "/api/tensorboard/start":
            result = self.tensorboard.ensure(self.tensorboard.default_logdir())
            self._send_json(handler, result, 200 if result.get("ok") else 400)
            return

        if path == "/api/tensorboard":
            logdir = self.tensorboard.resolve_run_logdir(body.get("run", ""))
            if logdir is None:
                self._send_json(handler, {"ok": False, "error": "run directory not found under runs/"}, 400)
                return
            result = self.tensorboard.ensure(logdir)
            self._send_json(handler, result, 200 if result.get("ok") else 400)
            return

        if path.startswith("/api/tensorboard/") and path.endswith("/stop"):
            tb_id  = path[len("/api/tensorboard/"):-len("/stop")]
            result = self.tensorboard.stop(tb_id)
            self._send_json(handler, result, 200 if result.get("ok") else 400)
            return

        if path == "/api/events/load":
            result = self.events.start_load(body.get("run", ""), body.get("split", "test"))
            self._send_json(handler, result, 200 if result.get("ok") else 400)
            return

        self._send_json(handler, {"error": "not found"}, 404)

    def _scoped_train_logdir(self, overrides: dict, interpreter: str) -> str:
        defaults    = self._train_config_defaults(interpreter)
        model_name  = overrides.get("model_name")              or defaults.get("model_name", "gps")
        base_logdir = overrides.get("training.io.log_base_dir") or defaults.get("training.io.log_base_dir", "runs")

        base = Path(base_logdir)
        if not base.is_absolute():
            base = self.paths.repo_root / base

        return str(RunPaths(base, model_name, overrides["run_name"]).tensorboard_dir)

    def _train_config_defaults(self, interpreter: str) -> dict:
        schema = self.configs.schema("train", interpreter)
        if not schema.get("ok"):
            return {}
        return {leaf["path"]: leaf["value"] for leaf in schema["leaves"]}

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

    def _stream_process(self, handler, identifier: str) -> None:
        try:
            pid = int(identifier)
        except ValueError:
            self._send_json(handler, {"error": "invalid pid"}, 400)
            return

        stream = self.processes.get_stream(pid)
        if stream is None:
            self._send_json(handler, {"error": "unknown process"}, 404)
            return

        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Connection", "keep-alive")
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.end_headers()

        subscriber = stream.subscribe()
        try:
            while True:
                try:
                    event = subscriber.get(timeout=15)
                except queue.Empty:
                    handler.wfile.write(b": keepalive\n\n")
                    handler.wfile.flush()
                    continue

                payload = json.dumps(event)
                handler.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                handler.wfile.flush()

                if event.get("type") == "end":
                    break
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            stream.unsubscribe(subscriber)

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

    def _proxy_tensorboard(self, handler) -> None:
        segments = handler.path.split("/")
        tb_id    = segments[2].split("?")[0] if len(segments) > 2 else ""
        record   = self.tensorboard.get(tb_id)

        if record is None:
            self._send_json(handler, {"error": "unknown tensorboard instance"}, 404)
            return

        length = int(handler.headers.get("Content-Length", 0) or 0)
        body   = handler.rfile.read(length) if length > 0 else None

        headers = {"Host": f"127.0.0.1:{record['port']}", "Accept-Encoding": "identity"}
        for name in ("Content-Type", "Accept", "X-XSRF-Protected"):
            value = handler.headers.get(name)
            if value:
                headers[name] = value

        connection = http.client.HTTPConnection("127.0.0.1", record["port"], timeout=60)
        try:
            connection.request(handler.command, handler.path, body=body, headers=headers)
            response = connection.getresponse()
            payload  = response.read()
            status   = response.status
            passthru = {name: response.getheader(name) for name in ("Content-Type", "Content-Encoding", "Cache-Control", "Location")}
        except OSError as error:
            self._send_json(handler, {"error": f"tensorboard unreachable: {error}"}, 502)
            return
        finally:
            connection.close()

        handler.send_response(status)
        for name, value in passthru.items():
            if value:
                handler.send_header(name, value)
        handler.send_header("Content-Length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)

    def route(self, handler) -> None:
        parsed = urlparse(handler.path)
        path   = parsed.path.rstrip("/") or "/"
        method = handler.command

        try:
            if parsed.path.startswith("/tb/"):
                self._proxy_tensorboard(handler)
            elif method == "GET":
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
