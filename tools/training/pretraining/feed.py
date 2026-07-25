from __future__ import annotations


class TrainerFeed:
    def __init__(self, trainer) -> None:
        self.trainer = trainer

    @staticmethod
    def to_model_input(batch, device):
        return batch.to(device, non_blocking=True)

    def forward_loss(self, model, batch):
        return self.trainer.criterion(model(batch), batch)["total_loss"]
