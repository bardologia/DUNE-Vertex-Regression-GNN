from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import scatter


class Loss:
    def __init__(self, loss_config, stats, logger, tracker):
        self.config  = loss_config
        self.stats   = stats
        self.logger  = logger
        self.tracker = tracker

        self.logger.section("[Loss Function]")
        self.logger.kv_table({
            "Data term"            : f"{self.config.data_term} (weight {self.config.data_weight})",
            "Euclidean weight"     : self.config.euclidean_weight,
            "Containment weight"   : self.config.containment_weight,
            "Light falloff weight" : self.config.light_falloff_weight,
        })

    def _data_term(self, predictions, targets):
        if self.config.data_term == "mse":
            return F.mse_loss(predictions, targets)
        if self.config.data_term == "huber":
            return F.huber_loss(predictions, targets, delta=self.config.huber_delta)
        raise ValueError(f"Unknown data term '{self.config.data_term}'")

    def _euclidean_term(self, predictions, targets):
        return torch.sqrt(((predictions - targets) ** 2).sum(dim=1) + 1e-12).mean()

    def _containment_term(self, predictions):
        excess = torch.clamp(predictions.abs() - self.config.containment_radius, min=0.0)
        return (excess ** 2).mean()

    def _light_falloff_term(self, predictions, data):
        device         = predictions.device
        physical_nodes = self.stats.node.inverse_torch(data.x, device)
        positions      = physical_nodes[:, 0:3]
        light          = physical_nodes[:, 3]

        physical_vertex = self.stats.target.inverse_torch(predictions, device)
        vertex_per_node = physical_vertex[data.batch]

        squared_distance   = ((positions - vertex_per_node) ** 2).sum(dim=1) + self.config.light_falloff_epsilon
        predicted_intensity = 1.0 / squared_distance

        light_mean     = scatter(light, data.batch, reduce="mean")[data.batch]
        intensity_mean = scatter(predicted_intensity, data.batch, reduce="mean")[data.batch]

        covariance     = scatter((light - light_mean) * (predicted_intensity - intensity_mean), data.batch, reduce="mean")
        light_variance = scatter((light - light_mean) ** 2, data.batch, reduce="mean")
        intensity_var  = scatter((predicted_intensity - intensity_mean) ** 2, data.batch, reduce="mean")

        correlation = covariance / torch.sqrt(light_variance * intensity_var + 1e-12)
        return (1.0 - correlation).mean()

    def __call__(self, predictions, data, step):
        targets    = data.y
        components = {}

        data_term          = self._data_term(predictions, targets)
        total_loss         = self.config.data_weight * data_term
        components["data"] = float(data_term.item())

        if self.config.euclidean_weight > 0.0 or self.config.containment_weight > 0.0:
            physical_predictions = self.stats.target.inverse_torch(predictions, predictions.device)
            physical_targets     = self.stats.target.inverse_torch(targets, predictions.device)

        if self.config.euclidean_weight > 0.0:
            euclidean              = self._euclidean_term(physical_predictions, physical_targets)
            total_loss             = total_loss + self.config.euclidean_weight * euclidean
            components["euclidean"] = float(euclidean.item())

        if self.config.containment_weight > 0.0:
            containment              = self._containment_term(physical_predictions)
            total_loss               = total_loss + self.config.containment_weight * containment
            components["containment"] = float(containment.item())

        if self.config.light_falloff_weight > 0.0 and hasattr(data, "batch") and data.batch is not None:
            light_falloff               = self._light_falloff_term(predictions, data)
            total_loss                  = total_loss + self.config.light_falloff_weight * light_falloff
            components["light_falloff"] = float(light_falloff.item())

        components["total"]     = float(total_loss.item())
        components["log_total"] = float(np.log1p(total_loss.item()))
        self.tracker.log_metrics("loss", components, step)

        return {"total_loss": total_loss}
