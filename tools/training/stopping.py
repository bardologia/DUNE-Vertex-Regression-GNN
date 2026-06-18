from __future__ import annotations


class EarlyStopping:
    def __init__(self, config, logger, tracker):
        self.config  = config
        self.logger  = logger
        self.tracker = tracker

        self.patience  = self.config.early_stopping.patience
        self.min_delta = self.config.early_stopping.min_delta

        self.logger.section("[Early Stopping]")
        self.logger.kv_table({
            "Patience"  : self.patience,
            "Min Delta" : self.min_delta,
        })

        self.best_loss  = None
        self.counter    = 0
        self.best_epoch = -1
        self.triggered  = False

    def __call__(self, val_loss: float, epoch: int) -> bool:
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss  = float(val_loss)
            self.best_epoch = int(epoch)
            self.counter    = 0
            stop            = False
            self.tracker.log_scalar("early_stop/best_val_loss", self.best_loss, epoch)

        else:
            self.counter += 1
            stop          = (self.counter >= self.patience)

        self.tracker.log_scalar("early_stop/counter", self.counter, epoch)

        if stop:
            self.triggered = True
            self.logger.warning(f"Early stopping triggered at epoch {epoch + 1}. Best epoch was {self.best_epoch + 1}.")

        return stop

    def reset(self) -> None:
        self.best_loss  = None
        self.counter    = 0
        self.best_epoch = -1
        self.triggered  = False
