"""
Trainer para FIP2.

Maneja el loop de entrenamiento, evaluación y checkpointing.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Dict, List, Tuple
import os
import time
import signal
import glob
from dataclasses import dataclass

from ..config import FIP2Config, TrainingConfig
from ..model import ZonedBrainModel
from ..plasticity import PlasticityManager
from .losses import compute_loss, compute_accuracy, compute_perplexity


class ByteDataset(Dataset):
    """Dataset de bytes para entrenamiento."""

    def __init__(
        self,
        data_path: str,
        context_length: int,
        split: str = "train",
        train_ratio: float = 0.9,
    ):
        # Leer archivo como bytes
        with open(data_path, "rb") as f:
            data = f.read()

        # Convertir a tensor
        self.data = torch.tensor(list(data), dtype=torch.long)

        # Split
        split_idx = int(len(self.data) * train_ratio)
        if split == "train":
            self.data = self.data[:split_idx]
        else:
            self.data = self.data[split_idx:]

        self.context_length = context_length

    def __len__(self) -> int:
        return max(0, len(self.data) - self.context_length - 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Contexto: context_length tokens
        x = self.data[idx : idx + self.context_length]
        # Target: siguiente token después del contexto
        y = self.data[idx + self.context_length]
        return x, y


@dataclass
class TrainStats:
    """Estadísticas de entrenamiento."""
    loss: float = 0.0
    accuracy: float = 0.0
    perplexity: float = 0.0
    diversity_loss: float = 0.0
    step: int = 0
    epoch: int = 0
    learning_rate: float = 0.0
    steps_per_second: float = 0.0


class Trainer:
    """
    Trainer principal para FIP2.
    """

    def __init__(
        self,
        model: ZonedBrainModel,
        config: FIP2Config,
        train_config: TrainingConfig,
    ):
        self.model = model
        self.config = config
        self.train_config = train_config
        self.device = model.device

        # Optimizador
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler
        self.scheduler = self._create_scheduler()

        # Datasets
        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None

        # Estado
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")

        # Historial
        self.train_history: List[TrainStats] = []
        self.val_history: List[TrainStats] = []

        # Flag para interrupción limpia
        self._interrupted = False
        self._setup_signal_handler()

        # Plasticity manager
        self.plasticity_manager = None
        if config.enable_plasticity:
            self.plasticity_manager = PlasticityManager(
                model=model,
                config=config,
                device=self.device,
            )

    def _setup_signal_handler(self):
        """Configura el handler para Ctrl+C."""
        def handler(signum, frame):
            if self._interrupted:
                print("\n\nForzando salida...")
                exit(1)
            print("\n\nInterrupción detectada. Guardando checkpoint...")
            self._interrupted = True

        signal.signal(signal.SIGINT, handler)

    def find_latest_checkpoint(self) -> Optional[str]:
        """Encuentra el checkpoint más reciente en el directorio."""
        checkpoint_dir = self.train_config.checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            return None

        # Buscar checkpoints por step
        pattern = os.path.join(checkpoint_dir, "checkpoint_step_*.pt")
        checkpoints = glob.glob(pattern)

        if not checkpoints:
            # Buscar otros checkpoints
            pattern = os.path.join(checkpoint_dir, "*.pt")
            checkpoints = glob.glob(pattern)
            # Excluir best_model.pt para preferir checkpoints con step
            checkpoints = [c for c in checkpoints if "best_model" not in c]

        if not checkpoints:
            return None

        # Ordenar por tiempo de modificación (más reciente primero)
        checkpoints.sort(key=os.path.getmtime, reverse=True)
        return checkpoints[0]

    def auto_resume(self) -> bool:
        """Intenta resumir desde el último checkpoint automáticamente."""
        latest = self.find_latest_checkpoint()
        if latest:
            print(f"\nCheckpoint encontrado: {latest}")
            print("Resumiendo entrenamiento...")
            self.load_checkpoint(latest)
            return True
        return False

    def _create_scheduler(self):
        """Crea el scheduler de learning rate."""
        total_steps = (
            self.train_config.num_epochs * self.train_config.steps_per_epoch
        )

        if self.train_config.lr_scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=self.config.learning_rate * 0.1,
            )
        elif self.train_config.lr_scheduler == "linear":
            return torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=total_steps,
            )
        else:
            return None

    def load_data(self, data_path: str) -> None:
        """Carga los datos de entrenamiento y validación."""
        self.train_dataset = ByteDataset(
            data_path,
            self.config.context_length,
            split="train",
            train_ratio=self.train_config.train_split,
        )
        self.val_dataset = ByteDataset(
            data_path,
            self.config.context_length,
            split="val",
            train_ratio=self.train_config.train_split,
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # MPS funciona mejor con 0
            pin_memory=False,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
        )

        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")

    def train_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
    ) -> TrainStats:
        """Ejecuta un paso de entrenamiento."""
        self.model.train()

        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        # Forward pass
        self.optimizer.zero_grad()
        logits, zone_outputs = self.model(x, return_zone_outputs=True)

        # Calcular pérdida
        loss = compute_loss(
            logits,
            y,
            zone_outputs,
            diversity_weight=self.config.diversity_weight,
        )

        # Backward pass
        loss.backward()

        # Gradient clipping
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )

        # Optimizer step
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        # Estadísticas
        accuracy = compute_accuracy(logits, y)
        perplexity = compute_perplexity(loss)

        stats = TrainStats(
            loss=loss.item(),
            accuracy=accuracy,
            perplexity=perplexity,
            step=self.global_step,
            epoch=self.epoch,
            learning_rate=self.optimizer.param_groups[0]["lr"],
        )

        self.global_step += 1
        return stats

    @torch.no_grad()
    def evaluate(self) -> TrainStats:
        """Evalúa el modelo en el conjunto de validación."""
        self.model.eval()

        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        for batch in self.val_loader:
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)

            logits, _ = self.model(x)
            loss = compute_loss(logits, y)

            total_loss += loss.item()
            total_accuracy += compute_accuracy(logits, y)
            num_batches += 1

            if num_batches >= 100:  # Limitar evaluación
                break

        avg_loss = total_loss / max(num_batches, 1)
        avg_accuracy = total_accuracy / max(num_batches, 1)

        return TrainStats(
            loss=avg_loss,
            accuracy=avg_accuracy,
            perplexity=compute_perplexity(torch.tensor(avg_loss)),
            step=self.global_step,
            epoch=self.epoch,
        )

    def save_checkpoint(self, path: str) -> None:
        """Guarda un checkpoint del modelo."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "config": self.config,
            "train_config": self.train_config,
            "best_val_loss": self.best_val_loss,
        }

        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str) -> None:
        """Carga un checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        print(f"Checkpoint loaded: {path}")
        print(f"  Resuming from step {self.global_step}, epoch {self.epoch}")

    def train(self) -> None:
        """Loop principal de entrenamiento."""
        if self.train_loader is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        print("\nStarting training...")
        print(f"  Epochs: {self.train_config.num_epochs}")
        print(f"  Steps per epoch: {self.train_config.steps_per_epoch}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Learning rate: {self.config.learning_rate}")

        for epoch in range(self.train_config.num_epochs):
            self.epoch = epoch
            epoch_start = time.time()
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            num_steps = 0

            train_iter = iter(self.train_loader)

            for step in range(self.train_config.steps_per_epoch):
                # Verificar interrupción
                if self._interrupted:
                    print("\nGuardando checkpoint antes de salir...")
                    self.save_checkpoint(
                        os.path.join(
                            self.train_config.checkpoint_dir,
                            f"checkpoint_step_{self.global_step}.pt"
                        )
                    )
                    print("Entrenamiento interrumpido. Usa el mismo comando para continuar.")
                    return

                # Obtener batch (con reciclaje si es necesario)
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)

                # Train step
                step_start = time.time()
                stats = self.train_step(batch)
                step_time = time.time() - step_start

                epoch_loss += stats.loss
                epoch_accuracy += stats.accuracy
                num_steps += 1

                # Logging
                if step % self.train_config.log_interval == 0:
                    print(
                        f"Epoch {epoch+1}/{self.train_config.num_epochs} | "
                        f"Step {step}/{self.train_config.steps_per_epoch} | "
                        f"Loss: {stats.loss:.4f} | "
                        f"Acc: {stats.accuracy:.4f} | "
                        f"PPL: {stats.perplexity:.2f} | "
                        f"LR: {stats.learning_rate:.2e} | "
                        f"{1/step_time:.1f} steps/s"
                    )

                # Evaluación
                if step % self.train_config.eval_interval == 0 and step > 0:
                    val_stats = self.evaluate()
                    print(
                        f"  [EVAL] Loss: {val_stats.loss:.4f} | "
                        f"Acc: {val_stats.accuracy:.4f} | "
                        f"PPL: {val_stats.perplexity:.2f}"
                    )

                    # Guardar mejor modelo
                    if val_stats.loss < self.best_val_loss:
                        self.best_val_loss = val_stats.loss
                        self.save_checkpoint(
                            os.path.join(
                                self.train_config.checkpoint_dir,
                                "best_model.pt"
                            )
                        )

                # Checkpoint periódico
                if step % self.train_config.save_interval == 0 and step > 0:
                    self.save_checkpoint(
                        os.path.join(
                            self.train_config.checkpoint_dir,
                            f"checkpoint_step_{self.global_step}.pt"
                        )
                    )

                # Plasticidad
                if (self.plasticity_manager is not None and
                    self.global_step % self.config.plasticity_interval == 0 and
                    self.global_step > 0):
                    plasticity_stats = self.plasticity_manager.step()
                    # Log básico
                    print(
                        f"  [PLASTICITY] Edge U: {plasticity_stats.mean_edge_utility:.4f} | "
                        f"Neuron U: {plasticity_stats.mean_neuron_utility:.4f} | "
                        f"Weak: {plasticity_stats.edges_weakened} | "
                        f"Strong: {plasticity_stats.edges_strengthened} | "
                        f"Refresh: {plasticity_stats.neurons_refreshed}"
                    )
                    # Log de Hebbian si hubo actividad
                    if plasticity_stats.edges_killed > 0 or plasticity_stats.edges_born > 0:
                        print(
                            f"  [HEBBIAN] Killed: {plasticity_stats.edges_killed} | "
                            f"Born: {plasticity_stats.edges_born}"
                        )

            # Fin de época
            epoch_time = time.time() - epoch_start
            avg_loss = epoch_loss / max(num_steps, 1)
            avg_acc = epoch_accuracy / max(num_steps, 1)

            print(
                f"\n=== Epoch {epoch+1} Complete ===\n"
                f"  Avg Loss: {avg_loss:.4f}\n"
                f"  Avg Accuracy: {avg_acc:.4f}\n"
                f"  Time: {epoch_time:.1f}s"
            )

            # Resumen de plasticidad si está habilitada
            if self.plasticity_manager is not None:
                summary = self.plasticity_manager.get_summary()
                if summary:
                    print(
                        f"  Plasticity: {summary['total_edges_weakened']} weakened, "
                        f"{summary['total_edges_strengthened']} strengthened, "
                        f"{summary['total_edges_killed']} killed, "
                        f"{summary['total_edges_born']} born\n"
                    )
                else:
                    print()

        print("\nTraining complete!")
