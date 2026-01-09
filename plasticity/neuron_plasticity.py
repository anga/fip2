"""
Plasticidad de Neuronas: Nacimiento y Muerte.

Implementa la creación y eliminación de neuronas basada en utilidad:
U_N(n) = α * H(n) + β * C(n) + γ * R(n)

Donde:
- H(n): Salud (activación, variabilidad)
- C(n): Conectividad (utilidad promedio de edges)
- R(n): Unicidad (no redundante)
"""
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class NeuronStats:
    """Estadísticas de una neurona."""
    health: float = 1.0
    connectivity: float = 0.5
    uniqueness: float = 1.0
    utility: float = 0.5
    age: int = 0
    consecutive_low_utility: int = 0


class NeuronPlasticityEngine:
    """
    Motor de plasticidad para neuronas.

    Decide cuándo crear o eliminar neuronas basándose en métricas de utilidad.
    """

    def __init__(
        self,
        alpha: float = 0.4,
        beta: float = 0.4,
        gamma: float = 0.2,
        death_threshold: float = 0.1,
        birth_threshold: float = 0.8,
        grace_period: int = 5,
        min_neurons: int = 50,
        max_neurons: int = 500,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.death_threshold = death_threshold
        self.birth_threshold = birth_threshold
        self.grace_period = grace_period
        self.min_neurons = min_neurons
        self.max_neurons = max_neurons

        # Contadores de bajo utilidad consecutivo
        self.consecutive_low: Dict[int, int] = {}

    def compute_neuron_utility(
        self,
        health: torch.Tensor,
        connectivity: torch.Tensor,
        uniqueness: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calcula la utilidad de cada neurona.

        Args:
            health: (num_neurons,) salud de cada neurona
            connectivity: (num_neurons,) conectividad promedio
            uniqueness: (num_neurons,) unicidad de representación

        Returns:
            Tensor de utilidades
        """
        utility = (
            self.alpha * health
            + self.beta * connectivity
            + self.gamma * uniqueness
        )
        return utility

    def compute_health(self, activation_history: torch.Tensor) -> torch.Tensor:
        """
        Calcula la salud desde el historial de activaciones.

        Args:
            activation_history: (history_len, num_neurons) historial

        Returns:
            Tensor de salud por neurona
        """
        # D: Tasa de muerte (fracción de activaciones ~0)
        death_rate = (activation_history.abs() < 0.01).float().mean(dim=0)

        # V: Variabilidad
        mean_act = activation_history.mean(dim=0)
        variance = activation_history.var(dim=0)
        variability = torch.clamp(variance / (mean_act.pow(2) + 1e-6), 0, 1)

        # F: Flujo (activación media)
        flow = mean_act.abs()

        # Salud: H = (1 - D) * V * F
        health = (1 - death_rate) * variability * flow

        # Normalizar a [0, 1]
        health = health / (health.max() + 1e-6)

        return health

    def compute_connectivity(
        self,
        edge_utilities: torch.Tensor,
        source_indices: torch.Tensor,
        target_indices: torch.Tensor,
        num_neurons: int,
    ) -> torch.Tensor:
        """
        Calcula la conectividad promedio por neurona.

        Args:
            edge_utilities: (num_edges,) utilidades de edges
            source_indices: (num_edges,) índices fuente
            target_indices: (num_edges,) índices destino
            num_neurons: Número total de neuronas

        Returns:
            Tensor de conectividad por neurona
        """
        device = edge_utilities.device
        connectivity = torch.zeros(num_neurons, device=device)
        counts = torch.zeros(num_neurons, device=device)

        # Sumar utilidades entrantes
        for i in range(len(target_indices)):
            target = target_indices[i].item()
            connectivity[target] += edge_utilities[i]
            counts[target] += 1

        # Sumar utilidades salientes
        for i in range(len(source_indices)):
            source = source_indices[i].item()
            connectivity[source] += edge_utilities[i]
            counts[source] += 1

        # Promediar
        connectivity = connectivity / (counts + 1e-6)

        # Normalizar a [0, 1]
        connectivity = connectivity / (connectivity.max() + 1e-6)

        return connectivity

    def compute_uniqueness(self, states: torch.Tensor) -> torch.Tensor:
        """
        Calcula la unicidad de cada neurona (1 - max correlación con otras).

        Args:
            states: (batch, num_neurons, dim) estados de neuronas

        Returns:
            Tensor de unicidad por neurona
        """
        # Promediar sobre batch: (num_neurons, dim)
        mean_states = states.mean(dim=0)

        # Calcular matriz de correlación usando cosine similarity
        # (num_neurons, dim) @ (dim, num_neurons) -> (num_neurons, num_neurons)
        norms = mean_states.norm(dim=1, keepdim=True) + 1e-6
        normalized = mean_states / norms

        corr_matrix = torch.mm(normalized, normalized.t())

        # Ignorar diagonal (auto-correlación = 1)
        corr_matrix.fill_diagonal_(0)

        # Unicidad = 1 - max correlación con otras
        max_corr = corr_matrix.abs().max(dim=1).values
        uniqueness = 1 - max_corr

        return uniqueness

    def identify_death_candidates(
        self,
        utility: torch.Tensor,
        min_neurons: int,
    ) -> List[int]:
        """
        Identifica neuronas candidatas a muerte.

        Args:
            utility: (num_neurons,) utilidad de cada neurona
            min_neurons: Número mínimo de neuronas a mantener

        Returns:
            Lista de índices de neuronas a eliminar
        """
        num_neurons = len(utility)
        candidates = []

        # No eliminar si estamos en el mínimo
        if num_neurons <= min_neurons:
            return candidates

        for i in range(num_neurons):
            if utility[i] < self.death_threshold:
                # Incrementar contador
                self.consecutive_low[i] = self.consecutive_low.get(i, 0) + 1

                # Si supera grace period, es candidato
                if self.consecutive_low[i] >= self.grace_period:
                    candidates.append(i)
            else:
                # Resetear contador
                self.consecutive_low[i] = 0

        # Limitar para no bajar del mínimo
        max_to_remove = num_neurons - min_neurons
        return candidates[:max_to_remove]

    def should_birth_neuron(
        self,
        mean_health: float,
        num_neurons: int,
        max_neurons: int,
    ) -> bool:
        """
        Decide si crear una nueva neurona.

        Args:
            mean_health: Salud promedio de las neuronas
            num_neurons: Número actual de neuronas
            max_neurons: Máximo permitido

        Returns:
            True si se debe crear una neurona
        """
        if num_neurons >= max_neurons:
            return False

        # Crear si la capacidad está saturada
        return mean_health > self.birth_threshold

    def select_parent_for_birth(
        self,
        utility: torch.Tensor,
        uniqueness: torch.Tensor,
    ) -> int:
        """
        Selecciona la neurona padre para crear una nueva.

        Prefiere neuronas con alta utilidad y alta unicidad.

        Args:
            utility: (num_neurons,) utilidad
            uniqueness: (num_neurons,) unicidad

        Returns:
            Índice de la neurona padre
        """
        # Score combinado
        score = utility * uniqueness

        # Seleccionar la mejor
        return score.argmax().item()

    def evaluate(
        self,
        activation_history: torch.Tensor,
        states: torch.Tensor,
        edge_utilities: torch.Tensor,
        source_indices: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[int], bool, Optional[int]]:
        """
        Evaluación completa de plasticidad.

        Args:
            activation_history: Historial de activaciones
            states: Estados actuales de neuronas
            edge_utilities: Utilidades de edges
            source_indices: Índices fuente de edges
            target_indices: Índices destino de edges

        Returns:
            utility: Utilidad de cada neurona
            death_candidates: Lista de neuronas a eliminar
            should_birth: Si crear nueva neurona
            parent_idx: Índice del padre si should_birth
        """
        num_neurons = states.shape[1]

        # Calcular componentes
        health = self.compute_health(activation_history)
        connectivity = self.compute_connectivity(
            edge_utilities, source_indices, target_indices, num_neurons
        )
        uniqueness = self.compute_uniqueness(states)

        # Utilidad total
        utility = self.compute_neuron_utility(health, connectivity, uniqueness)

        # Identificar candidatos a muerte
        death_candidates = self.identify_death_candidates(
            utility, self.min_neurons
        )

        # Decidir nacimiento
        mean_health = health.mean().item()
        should_birth = self.should_birth_neuron(
            mean_health, num_neurons, self.max_neurons
        )

        parent_idx = None
        if should_birth:
            parent_idx = self.select_parent_for_birth(utility, uniqueness)

        return utility, death_candidates, should_birth, parent_idx
