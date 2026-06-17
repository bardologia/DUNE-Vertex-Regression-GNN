from __future__ import annotations

import os
import sys


class EnvironmentPinner:
    @staticmethod
    def _requested_gpu() -> str:
        argv = sys.argv[1:]
        gpu  = "0"
        for index, token in enumerate(argv):
            if token == "--gpu" and index + 1 < len(argv):
                gpu = argv[index + 1]
            elif token.startswith("--gpu="):
                gpu = token.split("=", 1)[1]
        return gpu

    @staticmethod
    def pin() -> None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", EnvironmentPinner._requested_gpu())
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
