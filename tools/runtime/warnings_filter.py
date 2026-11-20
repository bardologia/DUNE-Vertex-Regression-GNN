from __future__ import annotations

import warnings
from dataclasses import dataclass


@dataclass(frozen=True)
class WarningPatterns:
    ignored : tuple = ((r".*pynvml package is deprecated.*", FutureWarning), (r".*can be accelerated via the 'torch-scatter' package.*", UserWarning))


class ThirdPartyWarnings:
    @staticmethod
    def silence() -> None:
        for message, category in WarningPatterns.ignored:
            warnings.filterwarnings("ignore", message=message, category=category)
