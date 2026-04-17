import torch
import torch.nn as nn


# =============================================================================
# EXPONENTIAL MOVING AVERAGE (EMA)
# =============================================================================
class EMA:
    """Exponential Moving Average of model weights.

    Maintains shadow weights updated as:
        shadow_t = decay * shadow_{t-1} + (1 - decay) * theta_t

    With decay=0.999, shadow weights average ~1000 recent batches,
    yielding smoother and often more accurate eval/test performance.

    Usage::
        ema = EMA(model, decay=0.999)
        for batch in loader:
            optimizer.step()
            ema.update(model)          # after every optimizer step

        # Evaluate with EMA weights
        ema.apply(model)
        try:
            evaluate(model)
        finally:
            ema.restore(model)

        # Permanently bake EMA weights before export
        ema.apply_to_model(model)
        torch.save(model.state_dict(), "best.pt")
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.backup = {}

    def update(self, model: nn.Module):
        """ Update shadow weights every batch."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow[name].copy_(
                        self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                    )

    def apply(self, model: nn.Module):
        """Swap model to EMA weights, backup training weights."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """ Restore training weights from backup."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup.clear()

    def apply_to_model(self, model: nn.Module):
        """Bake EMA weights permanently to model."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.data.copy_(self.shadow[name])