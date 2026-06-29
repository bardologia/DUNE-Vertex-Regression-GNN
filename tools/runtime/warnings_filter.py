from __future__ import annotations

import warnings


class ThirdPartyWarnings:
    PATTERNS = (
        (r".*pynvml package is deprecated.*", FutureWarning),
        (r".*can be accelerated via the 'torch-scatter' package.*", UserWarning),
    )

    @classmethod
    def silence(cls) -> None:
        for message, category in cls.PATTERNS:
            warnings.filterwarnings("ignore", message=message, category=category)
