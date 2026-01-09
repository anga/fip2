"""Training loop y funciones de pérdida."""
from .losses import compute_loss, diversity_loss
from .trainer import Trainer

__all__ = ["compute_loss", "diversity_loss", "Trainer"]
