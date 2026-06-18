from __future__ import annotations

import random

import numpy as np
import torch


class Reproducibility:
    SEED_MODULUS = 2 ** 32

    @staticmethod
    def seed_everything(seed: int) -> None:
        seed = int(seed)

        random.seed(seed)
        np.random.seed(seed % Reproducibility.SEED_MODULUS)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    @staticmethod
    def generator(seed: int) -> torch.Generator:
        generator = torch.Generator()
        generator.manual_seed(int(seed))

        return generator
