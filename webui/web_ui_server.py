from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from config_registry     import ConfigRegistry
from gpu_watchdog        import GpuWatchdog
from model_library       import ModelLibrary
from process_manager     import ProcessManager
from project_paths       import ProjectPaths
from request_router      import RequestRouter
from results_browser     import ResultsBrowser
from script_catalog      import ScriptCatalog
from server_logger       import ServerLogger
from system_monitor      import SystemMonitor
from tensorboard_manager import TensorboardManager


class _Handler(BaseHTTPRequestHandler):

    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        self.server.router.route(self)

    def do_POST(self) -> None:
        self.server.router.route(self)

    def log_message(self, format_string: str, *arguments) -> None:
        return


class WebUIServer:

    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.host   = host
        self.port   = port
        self.logger = ServerLogger()
        self.paths  = ProjectPaths()

        self.catalog      = ScriptCatalog(self.paths)
        self.configs      = ConfigRegistry(self.paths)
        self.processes    = ProcessManager(self.paths, self.logger)
        self.system       = SystemMonitor(self.paths)
        self.gpu_watchdog = GpuWatchdog(self.system, self.logger)
        self.tensorboard  = TensorboardManager(self.paths, self.logger)
        self.results      = ResultsBrowser(self.paths, self.logger)
        self.models       = ModelLibrary(self.paths)

        self.router = RequestRouter(
            paths        = self.paths,
            logger       = self.logger,
            catalog      = self.catalog,
            configs      = self.configs,
            processes    = self.processes,
            system       = self.system,
            gpu_watchdog = self.gpu_watchdog,
            tensorboard  = self.tensorboard,
            results      = self.results,
            models       = self.models,
        )

    def _report_ready(self) -> None:
        interpreters = self.paths.discover_interpreters()
        preferred    = self.paths.preferred_interpreter(interpreters)

        self.logger.banner("DUNE-GNN Control Console", [
            f"url          http://{self.host}:{self.port}",
            f"repo root    {self.paths.repo_root}",
            f"scripts      {len(self.catalog.list())} entry points",
            f"interpreter  {preferred}",
            f"envs found   {len(interpreters)}",
        ])

    def serve(self) -> None:
        self._report_ready()
        self.gpu_watchdog.start()

        server               = ThreadingHTTPServer((self.host, self.port), _Handler)
        server.router        = self.router
        server.daemon_threads = True

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            self.logger.warning("shutting down")
        finally:
            self.processes.stop_all()
            self.tensorboard.stop_all()
            server.server_close()
