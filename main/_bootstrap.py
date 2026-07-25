from __future__ import annotations

import os
import sys

from tools.runtime.warnings_filter import ThirdPartyWarnings


class EnvironmentPinner:
    DEFAULT_GPU = "0"

    @staticmethod
    def _requested_gpu() -> str | None:
        argv = sys.argv[1:]
        gpu  = None
        for index, token in enumerate(argv):
            if token == "--gpu" and index + 1 < len(argv):
                gpu = argv[index + 1]
            elif token.startswith("--gpu="):
                gpu = token.split("=", 1)[1]
        return gpu

    @staticmethod
    def pin() -> None:
        ThirdPartyWarnings.silence()

        requested = EnvironmentPinner._requested_gpu()
        if requested is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = requested
        else:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", EnvironmentPinner.DEFAULT_GPU)

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
