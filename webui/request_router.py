from __future__ import annotations

import http.client
import json
import mimetypes
import queue
import threading
from datetime     import datetime
from http         import HTTPStatus
from pathlib      import Path
from urllib.parse import parse_qs, urlparse

from configuration.shared          import IoNames
from configuration.webui.general   import ApiRoutes, HttpSettings, LaunchSettings, SystemMonitorSettings, TensorboardSettings
from pipelines.shared.run_metadata import RunPaths

from config_registry       import ConfigRegistry
from event_explorer        import EventExplorer
from gpu_watchdog          import GpuWatchdog
from model_library         import ModelLibrary
from preprocessing_preview import PreprocessingPreview
from process_manager       import ProcessManager
from project_paths         import ProjectPaths
from results_browser       import ResultsBrowser
from script_catalog        import ScriptCatalog
from server_logger         import ServerLogger
from system_monitor        import SystemMonitor
from tensorboard_manager   import TensorboardManager


class RequestRouter:

    def __init__(self, paths: ProjectPaths, logger: ServerLogger, catalog: ScriptCatalog, configs: ConfigRegistry, processes: ProcessManager, system: SystemMonitor, gpu_watchdog: GpuWatchdog, tensorboard: TensorboardManager, results: ResultsBrowser, models: ModelLibrary, events: EventExplorer, preprocess: PreprocessingPreview) -> None:
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
        self.preprocess   = preprocess

        self.processes.add_exit_listener(self._on_job_exit)

    def _route_get(self, handler, path: str) -> None:
        if path == ApiRoutes.index:
            self._serve_static(handler, IoNames.index_file)
            return
        if path.startswith(ApiRoutes.static_prefix):
            self._serve_static(handler, path[len(ApiRoutes.static_prefix):])
            return
        if path == ApiRoutes.system_status or path == ApiRoutes.system:
            self._send_json(handler, self._system_payload())
            return
        if path == ApiRoutes.system_gpu:
            self._send_json(handler, self.gpu_watchdog.latest())
            return
        if path == ApiRoutes.interpreters:
            interpreters = self.paths.discover_interpreters()
            self._send_json(handler, {"interpreters": interpreters, "preferred": self.paths.preferred_interpreter(interpreters)})
            return
        if path == ApiRoutes.scripts:
            self._send_json(handler, {"scripts": self.catalog.list()})
            return
        if path.startswith(ApiRoutes.scripts_prefix) and path.endswith(ApiRoutes.config_suffix):
            key    = path[len(ApiRoutes.scripts_prefix):-len(ApiRoutes.config_suffix)]
            result = self.configs.schema(key, self.paths.preferred_interpreter())
            self._send_result(handler, result)
            return
        if path.startswith(ApiRoutes.scripts_prefix) and path.endswith(ApiRoutes.source_suffix):
            key    = path[len(ApiRoutes.scripts_prefix):-len(ApiRoutes.source_suffix)]
            result = self.catalog.source(key)
            self._send_result(handler, result, HTTPStatus.NOT_FOUND)
            return
        if path == ApiRoutes.models:
            result = self.models.list(self.paths.preferred_interpreter())
            self._send_result(handler, result)
            return
        if path == ApiRoutes.runs:
            self._send_json(handler, self.results.list_runs())
            return
        if path == ApiRoutes.runs_file:
            target = (parse_qs(urlparse(handler.path).query).get("path") or [""])[0]
            self._serve_run_file(handler, target)
            return
        if path == ApiRoutes.processes:
            self._send_json(handler, {"processes": self.processes.list()})
            return
        if path.startswith(ApiRoutes.processes_prefix) and path.endswith(ApiRoutes.stream_suffix):
            identifier = path[len(ApiRoutes.processes_prefix):-len(ApiRoutes.stream_suffix)]
            self._stream_process(handler, identifier)
            return
        if path == ApiRoutes.tensorboard:
            self._send_json(handler, {"instances": self.tensorboard.list_instances()})
            return
        if path == ApiRoutes.events_sources:
            self._send_json(handler, self.events.list_sources())
            return
        if path == ApiRoutes.events_status:
            self._send_json(handler, self.events.load_status())
            return
        if path == ApiRoutes.events_list:
            query  = parse_qs(urlparse(handler.path).query)
            result = self.events.events((query.get("kind") or [""])[0], (query.get("name") or [""])[0], (query.get("split") or [""])[0])
            self._send_result(handler, result)
            return
        if path == ApiRoutes.events_detail:
            query = parse_qs(urlparse(handler.path).query)
            try:
                index = int((query.get("index") or ["-1"])[0])
            except ValueError:
                index = -1
            result = self.events.detail((query.get("kind") or [""])[0], (query.get("name") or [""])[0], (query.get("split") or [""])[0], index)
            self._send_result(handler, result)
            return
        if path == ApiRoutes.preprocess_status:
            self._send_json(handler, self.preprocess.load_status())
            return
        if path == ApiRoutes.preprocess_events:
            result = self.preprocess.events()
            self._send_result(handler, result)
            return

        self._send_json(handler, {"error": "not found"}, HTTPStatus.NOT_FOUND)

    def _route_post(self, handler, path: str) -> None:
        body = self._read_json(handler)

        if path == ApiRoutes.launch:
            script      = body.get("script", "")
            overrides   = dict(body.get("overrides", {}))
            interpreter = body.get("interpreter") or self.paths.preferred_interpreter()

            if script == "train" and not overrides.get("run_name"):
                overrides["run_name"] = datetime.now().strftime(IoNames.run_stamp_format)

            result = self.processes.launch(script, overrides, interpreter)

            if result["ok"] and script == "train" and self._tensorboard_requested(overrides):
                logdir = self._scoped_train_logdir(overrides, interpreter)
                threading.Thread(target=self.tensorboard.ensure, args=(logdir, result["pid"]), daemon=True).start()

            self._send_result(handler, result)
            return

        if path.startswith(ApiRoutes.processes_prefix) and path.endswith(ApiRoutes.kill_suffix):
            identifier = path[len(ApiRoutes.processes_prefix):-len(ApiRoutes.kill_suffix)]
            result     = self._with_pid(identifier, self.processes.kill)
            self._send_result(handler, result)
            return

        if path == ApiRoutes.tensorboard_start:
            result = self.tensorboard.ensure(self.tensorboard.default_logdir())
            self._send_result(handler, result)
            return

        if path == ApiRoutes.tensorboard:
            logdir = self.tensorboard.resolve_run_logdir(body.get("run", ""))
            if logdir is None:
                self._send_json(handler, {"ok": False, "error": "run directory not found under runs/"}, HTTPStatus.BAD_REQUEST)
                return
            result = self.tensorboard.ensure(logdir)
            self._send_result(handler, result)
            return

        if path.startswith(ApiRoutes.tensorboard_prefix) and path.endswith(ApiRoutes.stop_suffix):
            tb_id  = path[len(ApiRoutes.tensorboard_prefix):-len(ApiRoutes.stop_suffix)]
            result = self.tensorboard.stop(tb_id)
            self._send_result(handler, result)
            return

        if path == ApiRoutes.events_load:
            result = self.events.start_load(body.get("kind", "run"), body.get("name", ""), body.get("split", "test"))
            self._send_result(handler, result)
            return

        if path == ApiRoutes.preprocess_load:
            result = self.preprocess.start_load()
            self._send_result(handler, result)
            return

        if path == ApiRoutes.preprocess_preview:
            try:
                index = int(body.get("index", -1))
            except (TypeError, ValueError):
                index = -1
            result = self.preprocess.preview(index, body.get("config", {}))
            self._send_result(handler, result)
            return

        self._send_json(handler, {"error": "not found"}, HTTPStatus.NOT_FOUND)

    @staticmethod
    def _tensorboard_requested(overrides: dict) -> bool:
        return str(overrides.get("tensorboard", "")).strip().lower() in LaunchSettings.tensorboard_flags

    def _on_job_exit(self, pid: int, code: int) -> None:
        self.tensorboard.stop_for_owner(pid)

    def _scoped_train_logdir(self, overrides: dict, interpreter: str) -> str:
        defaults    = self._train_config_defaults(interpreter)
        model_name  = overrides.get("model_name")              or defaults["model_name"]
        base_logdir = overrides.get("training.io.log_base_dir") or defaults["training.io.log_base_dir"]

        base = Path(base_logdir)
        if not base.is_absolute():
            base = self.paths.repo_root / base

        return str(RunPaths(base, model_name, overrides["run_name"]).tensorboard_dir)

    def _train_config_defaults(self, interpreter: str) -> dict:
        schema = self.configs.schema("train", interpreter)
        return {leaf["path"]: leaf["value"] for leaf in schema["leaves"]}

    def _system_payload(self) -> dict:
        payload   = self.system.status()
        watchdog  = self.gpu_watchdog.latest()
        payload["watchdog"] = {
            "armed"     : watchdog["armed"],
            "interval"  : SystemMonitorSettings.gpu_poll_interval_s,
            "gpu_count" : len(payload["gpus"]),
            "updated"   : watchdog["updated"],
        }
        return payload

    def _serve_run_file(self, handler, raw_path: str) -> None:
        runs_root = self.paths.runs_dir.resolve()
        target    = (runs_root / raw_path) if not Path(raw_path).is_absolute() else Path(raw_path)
        target    = target.resolve()

        if not target.is_relative_to(runs_root) or not target.is_file():
            self._send_json(handler, {"error": "not found"}, HTTPStatus.NOT_FOUND)
            return

        self._send_file(handler, target, HttpSettings.run_file_cache_control)

    def _stream_process(self, handler, identifier: str) -> None:
        try:
            pid = int(identifier)
        except ValueError:
            self._send_json(handler, {"error": "invalid pid"}, HTTPStatus.BAD_REQUEST)
            return

        stream = self.processes.get_stream(pid)
        if stream is None:
            self._send_json(handler, {"error": "unknown process"}, HTTPStatus.NOT_FOUND)
            return

        handler.send_response(HTTPStatus.OK)
        handler.send_header("Content-Type", HttpSettings.stream_content_type)
        handler.send_header("Cache-Control", HttpSettings.no_cache_control)
        handler.send_header("Connection", HttpSettings.stream_connection)
        handler.send_header("Access-Control-Allow-Origin", HttpSettings.stream_allowed_origin)
        handler.end_headers()

        subscriber = stream.subscribe()
        try:
            while True:
                try:
                    event = subscriber.get(timeout=HttpSettings.stream_keepalive_s)
                except queue.Empty:
                    handler.wfile.write(b": keepalive\n\n")
                    handler.wfile.flush()
                    continue

                payload = json.dumps(event)
                handler.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                handler.wfile.flush()

                if event["type"] == "end":
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

    def _send_result(self, handler, result: dict, failure: HTTPStatus = HTTPStatus.BAD_REQUEST) -> None:
        self._send_json(handler, result, HTTPStatus.OK if result["ok"] else failure)

    def _send_file(self, handler, target: Path, cache_control: str) -> None:
        data         = target.read_bytes()
        content_type = mimetypes.guess_type(str(target))[0] or HttpSettings.binary_content_type

        handler.send_response(HTTPStatus.OK)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(data)))
        handler.send_header("Cache-Control", cache_control)
        handler.end_headers()
        handler.wfile.write(data)

    def _send_json(self, handler, payload: dict, status: int = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", HttpSettings.json_content_type)
        handler.send_header("Content-Length", str(len(encoded)))
        handler.end_headers()
        handler.wfile.write(encoded)

    def _serve_static(self, handler, relative: str) -> None:
        target = (self.paths.static_dir / relative).resolve()
        if not target.is_relative_to(self.paths.static_dir.resolve()):
            self._send_json(handler, {"error": "forbidden"}, HTTPStatus.FORBIDDEN)
            return
        if not target.is_file():
            self._send_json(handler, {"error": "not found"}, HTTPStatus.NOT_FOUND)
            return

        self._send_file(handler, target, HttpSettings.no_cache_control)

    def _proxy_tensorboard(self, handler) -> None:
        segments = handler.path.split("/")
        tb_id    = segments[2].split("?")[0] if len(segments) > 2 else ""
        record   = self.tensorboard.get(tb_id)

        if record is None:
            self._send_json(handler, {"error": "unknown tensorboard instance"}, HTTPStatus.NOT_FOUND)
            return

        length = int(handler.headers.get("Content-Length", 0) or 0)
        body   = handler.rfile.read(length) if length > 0 else None

        headers = {"Host": f"{TensorboardSettings.host}:{record['port']}", "Accept-Encoding": TensorboardSettings.proxy_accept_encoding}
        for name in TensorboardSettings.proxy_request_headers:
            value = handler.headers.get(name)
            if value:
                headers[name] = value

        connection = http.client.HTTPConnection(TensorboardSettings.host, record["port"], timeout=TensorboardSettings.proxy_timeout_s)
        try:
            connection.request(handler.command, handler.path, body=body, headers=headers)
            response = connection.getresponse()
            payload  = response.read()
            status   = response.status
            passthru = {name: response.getheader(name) for name in TensorboardSettings.proxy_response_headers}
        except OSError as error:
            self._send_json(handler, {"error": f"tensorboard unreachable: {error}"}, HTTPStatus.BAD_GATEWAY)
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
        path   = parsed.path.rstrip("/") or ApiRoutes.index
        method = handler.command

        try:
            if parsed.path.startswith(TensorboardSettings.path_prefix):
                self._proxy_tensorboard(handler)
            elif method == "GET":
                self._route_get(handler, path)
            elif method == "POST":
                self._route_post(handler, path)
            else:
                self._send_json(handler, {"error": "method not allowed"}, HTTPStatus.METHOD_NOT_ALLOWED)
        except BrokenPipeError:
            pass
        except Exception as error:
            self.logger.error(f"router error on {method} {path}: {error}")
            try:
                self._send_json(handler, {"error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            except Exception:
                pass
