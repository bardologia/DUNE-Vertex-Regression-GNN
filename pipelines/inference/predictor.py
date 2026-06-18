from __future__ import annotations

import torch


class Predictor:
    def __init__(self, model, checkpoint_path, stats, device, logger):
        self.model           = model.to(device)
        self.checkpoint_path = checkpoint_path
        self.stats           = stats
        self.device          = device
        self.logger          = logger

    def prepare(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["params"])
        self.model.eval()
        return self

    def _denormalize(self, target):
        return self.stats.target.inverse_torch(target, self.device)

    def predict(self, loader):
        prediction_segments = []
        target_segments     = []

        with torch.no_grad():
            for data in loader:
                data        = data.to(self.device, non_blocking=True)
                predictions = self.model(data)
                prediction_segments.append(self._denormalize(predictions).cpu())
                target_segments.append(self._denormalize(data.y).cpu())

        return torch.cat(prediction_segments), torch.cat(target_segments)
