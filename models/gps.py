from __future__ import annotations

from models.blocks import GPSEncoder, GraphRegressor


class GPS(GraphRegressor):
    def __init__(self, config):
        super().__init__(config, GPSEncoder(config))
