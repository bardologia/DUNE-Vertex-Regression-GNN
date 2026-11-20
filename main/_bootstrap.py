from __future__ import annotations

import os
import sys
from dataclasses import dataclass

from tools.runtime.warnings_filter import ThirdPartyWarnings


@dataclass(frozen=True)
class BootstrapSettings:
    default_gpu     : str = "0"
    gpu_flag        : str = "--gpu"
    cuda_alloc_conf : str = "expandable_segments:True"


class EnvironmentPinner:
    @staticmethod
    def _requested_gpu() -> str | None:
        argv = sys.argv[1:]
        gpu  = None
        for index, token in enumerate(argv):
            if token == BootstrapSettings.gpu_flag and index + 1 < len(argv):
                gpu = argv[index + 1]
            elif token.startswith(f"{BootstrapSettings.gpu_flag}="):
                gpu = token.split("=", 1)[1]
        return gpu

    @staticmethod
    def pin() -> None:
        ThirdPartyWarnings.silence()

        requested = EnvironmentPinner._requested_gpu()
        if requested is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = requested
        else:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", BootstrapSettings.default_gpu)

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", BootstrapSettings.cuda_alloc_conf)
