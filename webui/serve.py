from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from web_ui_server import WebUIServer


class ServeEntry:

    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(description="DUNE-GNN control console")
        self.parser.add_argument("--host", default="127.0.0.1")
        self.parser.add_argument("--port", type=int, default=8765)

    def run(self) -> None:
        arguments = self.parser.parse_args()
        server    = WebUIServer(host=arguments.host, port=arguments.port)
        server.serve()


if __name__ == "__main__":
    ServeEntry().run()
