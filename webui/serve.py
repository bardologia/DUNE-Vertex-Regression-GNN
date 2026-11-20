from __future__ import annotations

import argparse
import sys
from pathlib import Path


class RepositoryPath:

    @staticmethod
    def register() -> None:
        repository_root = str(Path(__file__).resolve().parent.parent)
        if repository_root not in sys.path:
            sys.path.insert(0, repository_root)


RepositoryPath.register()

from tools.runtime.warnings_filter import ThirdPartyWarnings

ThirdPartyWarnings.silence()

from configuration.webui.general import ServerSettings
from web_ui_server               import WebUIServer


class ServeEntry:

    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(description="DUNE-GNN control console")
        self.parser.add_argument("--host", default=ServerSettings.host)
        self.parser.add_argument("--port", type=int, default=ServerSettings.port)

    def run(self) -> None:
        arguments = self.parser.parse_args()
        server    = WebUIServer(host=arguments.host, port=arguments.port)
        server.serve()


if __name__ == "__main__":
    ServeEntry().run()
