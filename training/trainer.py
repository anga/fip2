"""
Trainer para FIP2.

Maneja el loop de entrenamiento, evaluación y checkpointing.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Dict, List, Tuple
import os
import mmap
import time
import signal
import glob
from dataclasses import dataclass

from ..config import FIP2Config, TrainingConfig
from ..model import ZonedBrainModel
from ..plasticity import PlasticityManager
from .losses import compute_loss, compute_accuracy, compute_perplexity


class ByteDataset(Dataset):
    """
    Dataset de bytes para entrenamiento con memory-mapping.

    Soporta:
    - Un archivo único: --data archivo.txt
    - Un directorio con múltiples archivos: --data data/
      (carga UN archivo a la vez, rotando entre ellos)

    Usa mmap para no cargar todo en RAM.
    """

    def __init__(
        self,
        data_path: str,
        context_length: int,
        split: str = "train",
        train_ratio: float = 0.9,
    ):
        self.context_length = context_length
        self.data_path = data_path

        # Detectar si es archivo o directorio
        if os.path.isdir(data_path):
            self._init_from_directory(data_path, split, train_ratio)
        else:
            self._init_from_file(data_path, split, train_ratio)

    def _init_from_file(self, file_path: str, split: str, train_ratio: float):
        """Inicializa desde un archivo único."""
        self._mode = "single"
        file_size = os.path.getsize(file_path)

        split_idx = int(file_size * train_ratio)
        if split == "train":
            self.start_idx = 0
            self.end_idx = split_idx
        else:
            self.start_idx = split_idx
            self.end_idx = file_size

        self.length = self.end_idx - self.start_idx
        self._file = open(file_path, "rb")
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

    def _init_from_directory(self, dir_path: str, split: str, train_ratio: float):
        """Inicializa desde un directorio - carga UN archivo a la vez."""
        self._mode = "multi"
        self._dir_path = dir_path

        # Encontrar todos los archivos .txt
        txt_files = sorted([f for f in os.listdir(dir_path) if f.endswith(".txt")])

        if not txt_files:
            raise ValueError(f"No se encontraron archivos .txt en {dir_path}")

        # Dividir archivos en train/val
        split_idx = max(1, int(len(txt_files) * train_ratio))

        if split == "train":
            self._files_list = txt_files[:split_idx]
        else:
            self._files_list = txt_files[split_idx:] if split_idx < len(txt_files) else [txt_files[-1]]

        self.num_files = len(self._files_list)

        # Calcular tamaños de cada archivo
        self._file_sizes = [
            os.path.getsize(os.path.join(dir_path, f))
            for f in self._files_list
        ]

        # Archivo actual cargado
        self._current_file_idx = -1
        self._current_file = None
        self._current_mmap = None
        self._current_size = 0

        # Cargar primer archivo
        self._load_file(0)

    def _load_file(self, file_idx: int):
        """Carga un archivo específico, cerrando el anterior."""
        if self._mode != "multi":
            return

        # Cerrar archivo anterior
        if self._current_mmap:
            self._current_mmap.close()
        if self._current_file:
            self._current_file.close()

        # Cargar nuevo archivo
        self._current_file_idx = file_idx % self.num_files
        file_path = os.path.join(self._dir_path, self._files_list[self._current_file_idx])

        self._current_file = open(file_path, "rb")
        self._current_mmap = mmap.mmap(
            self._current_file.fileno(), 0, access=mmap.ACCESS_READ
        )
        self._current_size = self._file_sizes[self._current_file_idx]
        self.length = self._current_size

    def next_file(self) -> str:
        """Avanza al siguiente archivo. Retorna el nombre del archivo cargado."""
        if self._mode != "multi":
            return self.data_path

        next_idx = (self._current_file_idx + 1) % self.num_files
        self._load_file(next_idx)
        return self._files_list[self._current_file_idx]

    def get_current_file(self) -> str:
        """Retorna el nombre del archivo actual."""
        if self._mode == "single":
            return self.data_path
        return self._files_list[self._current_file_idx]

    def __del__(self):
        """Cerrar archivos al destruir el objeto."""
        if hasattr(self, '_mode'):
            if self._mode == "single":
                if hasattr(self, '_mmap') and self._mmap:
                    self._mmap.close()
                if hasattr(self, '_file') and self._file:
                    self._file.close()
            else:
                if hasattr(self, '_current_mmap') and self._current_mmap:
                    self._current_mmap.close()
                if hasattr(self, '_current_file') and self._current_file:
                    self._current_file.close()

    def __len__(self) -> int:
        return max(0, self.length - self.context_length - 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._mode == "single":
            actual_idx = self.start_idx + idx
            self._mmap.seek(actual_idx)
            data = self._mmap.read(self.context_length + 1)
        else:
            # Modo multi: leer del archivo actual
            idx = idx % (self._current_size - self.context_length - 1)
            self._current_mmap.seek(idx)
            data = self._current_mmap.read(self.context_length + 1)

        # Padding si es necesario
        if len(data) < self.context_length + 1:
            data = data + b'\x00' * (self.context_length + 1 - len(data))

        bytes_tensor = torch.tensor(list(data), dtype=torch.long)
        x = bytes_tensor[:self.context_length]
        y = bytes_tensor[self.context_length]
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
            # Verificar compatibilidad de configuración antes de cargar
            try:
                checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
                saved_config = checkpoint.get("config")
                if saved_config:
                    # Comparar parámetros críticos
                    if (saved_config.num_zones != self.config.num_zones or
                        saved_config.neurons_per_zone != self.config.neurons_per_zone or
                        saved_config.hub_neurons != self.config.hub_neurons or
                        saved_config.neuron_dim != self.config.neuron_dim or
                        saved_config.context_length != self.config.context_length):
                        print(f"\nCheckpoint encontrado: {latest}")
                        print("  ADVERTENCIA: Configuración incompatible")
                        print(f"    Checkpoint: {saved_config.num_zones} zonas, {saved_config.neurons_per_zone} neurons/zone, dim={saved_config.neuron_dim}, ctx={saved_config.context_length}")
                        print(f"    Actual:     {self.config.num_zones} zonas, {self.config.neurons_per_zone} neurons/zone, dim={self.config.neuron_dim}, ctx={self.config.context_length}")
                        print("  Iniciando entrenamiento desde cero...")
                        return False
            except Exception as e:
                print(f"Error verificando checkpoint: {e}")
                return False

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
        # Mostrar info
        if os.path.isdir(data_path):
            txt_files = [f for f in os.listdir(data_path) if f.endswith(".txt")]
            total_size = sum(
                os.path.getsize(os.path.join(data_path, f))
                for f in txt_files
            )
            total_size_mb = total_size / (1024 * 1024)
            total_size_gb = total_size / (1024 * 1024 * 1024)

            print(f"Data directory: {data_path}/")
            print(f"  Files: {len(txt_files)} .txt files")
            if total_size_gb >= 1:
                print(f"  Total size: {total_size_gb:.2f} GB")
            else:
                print(f"  Total size: {total_size_mb:.1f} MB")
        else:
            file_size = os.path.getsize(data_path)
            file_size_mb = file_size / (1024 * 1024)
            file_size_gb = file_size / (1024 * 1024 * 1024)

            if file_size_gb >= 1:
                print(f"Data file: {data_path} ({file_size_gb:.2f} GB)")
            else:
                print(f"Data file: {data_path} ({file_size_mb:.1f} MB)")

        print("Using memory-mapped I/O (low RAM usage)")

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

        print(f"Train samples: {len(self.train_dataset):,}")
        print(f"Val samples: {len(self.val_dataset):,}")

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

    def _refresh_dataloader(self):
        """Recrea el DataLoader después de cambiar de archivo."""
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )

    def train(self) -> None:
        """Loop principal de entrenamiento."""
        if self.train_loader is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Info sobre modo multi-archivo
        is_multi = hasattr(self.train_dataset, '_mode') and self.train_dataset._mode == "multi"
        num_files = self.train_dataset.num_files if is_multi else 1

        if is_multi:
            print(f"\nMulti-file mode: {num_files} files")
            print(f"  Steps per file: {self.train_config.steps_per_epoch}")
            print(f"  Total steps per epoch: {self.train_config.steps_per_epoch * num_files}")

        print("\nStarting training...")
        print(f"  Epochs: {self.train_config.num_epochs}")
        print(f"  Steps per epoch: {self.train_config.steps_per_epoch}" + (f" x {num_files} files" if is_multi else ""))
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Learning rate: {self.config.learning_rate}")

        for epoch in range(self.train_config.num_epochs):
            self.epoch = epoch
            epoch_start = time.time()
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            num_steps = 0

            # Iterar por todos los archivos en cada epoch
            for file_idx in range(num_files):
                # Cambiar al archivo correspondiente
                if is_multi:
                    self.train_dataset._load_file(file_idx)
                    self._refresh_dataloader()
                    print(f"\n  [FILE {file_idx + 1}/{num_files}] {self.train_dataset.get_current_file()}")

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
                        file_info = f"F{file_idx + 1}/{num_files} | " if is_multi else ""
                        print(
                            f"Epoch {epoch+1}/{self.train_config.num_epochs} | "
                            f"{file_info}"
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
                        print(
                            f"  [PLASTICITY] Edge U: {plasticity_stats.mean_edge_utility:.4f} | "
                            f"Neuron U: {plasticity_stats.mean_neuron_utility:.4f} | "
                            f"Weak: {plasticity_stats.edges_weakened} | "
                            f"Strong: {plasticity_stats.edges_strengthened} | "
                            f"Refresh: {plasticity_stats.neurons_refreshed}"
                        )
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
