"""
Métrica de Utilidad de Conexión (MUC).

Implementa el cálculo de utilidad de edges según el modelo FIP:
U(e) = α * G(e) + β * I(e) + γ * S(e)

Donde:
- G(e): Sensibilidad al gradiente
- I(e): Flujo de información
- S(e): Estabilidad temporal
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class EdgeStats:
    """Estadísticas de una conexión."""
    gradient_sensitivity: float = 0.0
    information_flow: float = 0.0
    stability: float = 1.0
    utility: float = 0.0
    age: int = 0


class EdgeUtilityTracker(nn.Module):
    """
    Rastrea y calcula la utilidad de conexiones.

    Mantiene EMAs de las métricas para cada edge.
    """

    def __init__(
        self,
        num_edges: int,
        alpha: float = 0.4,
        beta: float = 0.4,
        gamma: float = 0.2,
        ema_decay: float = 0.99,
        device: torch.device = None,
    ):
        super().__init__()
        self.num_edges = num_edges
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ema_decay = ema_decay
        self.device = device or torch.device("cpu")

        # EMAs de métricas
        self.register_buffer(
            "gradient_ema",
            torch.zeros(num_edges, device=self.device)
        )
        self.register_buffer(
            "flow_ema",
            torch.zeros(num_edges, device=self.device)
        )
        self.register_buffer(
            "weight_history",
            torch.zeros(100, num_edges, device=self.device)  # Últimos 100 valores
        )
        self.register_buffer(
            "utility",
            torch.zeros(num_edges, device=self.device)
        )
        self.register_buffer(
            "age",
            torch.zeros(num_edges, dtype=torch.long, device=self.device)
        )

        self.history_ptr = 0
        self.history_filled = False

    def update_gradient_sensitivity(self, gradients: torch.Tensor) -> None:
        """
        Actualiza la sensibilidad al gradiente.

        Args:
            gradients: Tensor de shape (num_edges,) con |∂L/∂w|
        """
        self.gradient_ema = (
            self.ema_decay * self.gradient_ema
            + (1 - self.ema_decay) * gradients.abs()
        )

    def update_information_flow(
        self,
        source_activations: torch.Tensor,
        target_activations: torch.Tensor,
    ) -> None:
        """
        Actualiza el flujo de información basado en correlación.

        Args:
            source_activations: (num_edges, dim) activaciones fuente
            target_activations: (num_edges, dim) activaciones destino
        """
        # Calcular correlación por edge
        # Normalizar
        source_norm = source_activations - source_activations.mean(dim=-1, keepdim=True)
        target_norm = target_activations - target_activations.mean(dim=-1, keepdim=True)

        # Covarianza
        cov = (source_norm * target_norm).sum(dim=-1)

        # Desviaciones estándar
        source_std = source_norm.pow(2).sum(dim=-1).sqrt() + 1e-6
        target_std = target_norm.pow(2).sum(dim=-1).sqrt() + 1e-6

        # Correlación
        correlation = cov / (source_std * target_std)

        # Actualizar EMA
        self.flow_ema = (
            self.ema_decay * self.flow_ema
            + (1 - self.ema_decay) * correlation.abs()
        )

    def update_stability(self, weights: torch.Tensor) -> None:
        """
        Actualiza la estabilidad basada en varianza de pesos.

        Args:
            weights: Tensor de shape (num_edges,) con pesos actuales
        """
        # Guardar en historial
        idx = self.history_ptr % self.weight_history.shape[0]
        self.weight_history[idx] = weights.detach()
        self.history_ptr += 1

        if self.history_ptr >= self.weight_history.shape[0]:
            self.history_filled = True

    def compute_stability(self) -> torch.Tensor:
        """Calcula la estabilidad desde el historial de pesos."""
        if not self.history_filled and self.history_ptr < 10:
            # No hay suficiente historial, asumir estable
            return torch.ones(self.num_edges, device=self.device)

        # Usar el historial disponible
        if self.history_filled:
            history = self.weight_history
        else:
            history = self.weight_history[:self.history_ptr]

        # Varianza y media
        weight_var = history.var(dim=0)
        weight_mean = history.mean(dim=0).abs() + 1e-6

        # S(e) = 1 - Var(w) / |w̄|
        stability = 1 - (weight_var / weight_mean).clamp(0, 2)

        return stability.clamp(0, 1)

    def compute_utility(self) -> torch.Tensor:
        """
        Calcula la utilidad de cada edge.

        U(e) = α * G(e) + β * I(e) + γ * S(e)

        Returns:
            Tensor de shape (num_edges,) con utilidades en [0, 1]
        """
        # Normalizar componentes a [0, 1]
        G = self.gradient_ema / (self.gradient_ema.max() + 1e-6)
        I = self.flow_ema  # Ya está en [0, 1]
        S = self.compute_stability()

        # Calcular utilidad
        utility = self.alpha * G + self.beta * I + self.gamma * S

        self.utility = utility
        return utility

    def get_low_utility_edges(self, percentile: float = 5.0) -> torch.Tensor:
        """
        Obtiene índices de edges con baja utilidad.

        Args:
            percentile: Percentil inferior a considerar

        Returns:
            Tensor de índices de edges candidatos a poda
        """
        threshold = torch.quantile(self.utility, percentile / 100.0)
        return (self.utility < threshold).nonzero(as_tuple=False).squeeze(-1)

    def get_high_utility_edges(self, percentile: float = 95.0) -> torch.Tensor:
        """
        Obtiene índices de edges con alta utilidad.

        Args:
            percentile: Percentil superior a considerar

        Returns:
            Tensor de índices de edges de alto valor
        """
        threshold = torch.quantile(self.utility, percentile / 100.0)
        return (self.utility >= threshold).nonzero(as_tuple=False).squeeze(-1)

    def increment_age(self) -> None:
        """Incrementa la edad de todos los edges."""
        self.age += 1

    def reset_edge(self, edge_idx: int) -> None:
        """Resetea las estadísticas de un edge (para nuevos edges)."""
        self.gradient_ema[edge_idx] = 0.0
        self.flow_ema[edge_idx] = 0.0
        self.weight_history[:, edge_idx] = 0.0
        self.utility[edge_idx] = 0.0
        self.age[edge_idx] = 0
