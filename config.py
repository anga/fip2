"""
Configuración del modelo FIP2.

Todas las configuraciones del modelo en un solo lugar.
"""
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class FIP2Config:
    """Configuración principal del modelo FIP2."""

    # === Arquitectura de Zonas ===
    num_zones: int = 5
    neurons_per_zone: int = 200
    hub_neurons: int = 400
    buffer_ratio: float = 0.1  # 10% de neuronas son buffer
    intra_zone_connectivity: float = 0.1  # 10% de conexiones internas

    # === Encoding ===
    vocab_size: int = 256  # Byte-level
    context_length: int = 128
    neuron_dim: int = 64

    # === Propagación ===
    num_waves: int = 5  # Más waves para mejor propagación
    decay: float = 0.3  # Bajo para permitir más flujo de información nueva
    activation: Literal["gelu", "relu", "silu", "tanh"] = "gelu"

    # === Plasticidad ===
    enable_plasticity: bool = True
    plasticity_interval: int = 100  # Cada 100 steps (antes 500)
    min_neurons_per_zone: int = 50
    max_neurons_per_zone: int = 500

    # Parámetros MUC (Métrica de Utilidad de Conexión)
    muc_alpha: float = 0.4  # Peso gradiente
    muc_beta: float = 0.4   # Peso flujo
    muc_gamma: float = 0.2  # Peso estabilidad
    muc_ema_decay: float = 0.99

    # Parámetros de utilidad de neurona
    neuron_alpha: float = 0.4  # Peso salud
    neuron_beta: float = 0.4   # Peso conectividad
    neuron_gamma: float = 0.2  # Peso unicidad
    neuron_death_threshold: float = 0.1
    neuron_birth_threshold: float = 0.8
    neuron_grace_period: int = 5

    # === Training ===
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    diversity_weight: float = 0.1  # λ para L_diversity
    max_grad_norm: float = 1.0

    # === Device ===
    device: str = "auto"  # auto, mps, cuda, cpu

    @property
    def total_neurons(self) -> int:
        """Número total de neuronas en el modelo."""
        return self.num_zones * self.neurons_per_zone + self.hub_neurons

    @property
    def buffer_neurons_per_zone(self) -> int:
        """Número de neuronas buffer por zona."""
        return int(self.neurons_per_zone * self.buffer_ratio)

    @property
    def total_buffer_neurons(self) -> int:
        """Número total de neuronas buffer."""
        return self.num_zones * self.buffer_neurons_per_zone


@dataclass
class TrainingConfig:
    """Configuración específica para entrenamiento."""

    # Data
    data_path: str = ""
    train_split: float = 0.9

    # Training loop
    num_epochs: int = 10
    steps_per_epoch: int = 1000
    eval_interval: int = 100
    save_interval: int = 1000
    log_interval: int = 10

    # Optimización
    warmup_steps: int = 100
    lr_scheduler: Literal["cosine", "linear", "constant"] = "cosine"

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    resume_from: str = ""


def get_small_config() -> FIP2Config:
    """Configuración pequeña para pruebas rápidas."""
    return FIP2Config(
        num_zones=3,
        neurons_per_zone=100,
        hub_neurons=200,
        context_length=64,
        neuron_dim=32,
        num_waves=2,
    )


def get_medium_config() -> FIP2Config:
    """Configuración media para experimentación."""
    return FIP2Config(
        num_zones=5,
        neurons_per_zone=200,
        hub_neurons=400,
        context_length=128,
        neuron_dim=64,
        num_waves=3,
    )


def get_large_config() -> FIP2Config:
    """Configuración grande para entrenamiento serio."""
    return FIP2Config(
        num_zones=7,
        neurons_per_zone=400,
        hub_neurons=800,
        context_length=256,
        neuron_dim=128,
        num_waves=5,
    )
