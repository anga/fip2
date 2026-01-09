"""
Gestión de conectividad entre zonas y hub.

Este módulo maneja la creación y actualización de conexiones
entre componentes del modelo.
"""
import torch
import torch.nn as nn
from typing import List, Tuple, Optional


def create_random_connectivity_mask(
    num_source: int,
    num_target: int,
    connectivity: float,
    allow_self_connections: bool = False,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Crea una máscara de conectividad aleatoria.

    Args:
        num_source: Número de neuronas fuente
        num_target: Número de neuronas destino
        connectivity: Probabilidad de conexión (0-1)
        allow_self_connections: Permitir conexiones i->i
        device: Dispositivo para el tensor

    Returns:
        Tensor booleano de shape (num_source, num_target)
    """
    mask = torch.rand(num_source, num_target, device=device) < connectivity

    if not allow_self_connections and num_source == num_target:
        mask.fill_diagonal_(False)

    return mask


def aggregate_by_target_mps_safe(
    signals: torch.Tensor,
    target_indices: torch.Tensor,
    num_targets: int,
) -> torch.Tensor:
    """
    Agrega señales por neurona destino de forma MPS-safe.

    Esta función reemplaza scatter_add que no funciona bien en MPS.

    Args:
        signals: Tensor de shape (batch, num_edges, dim)
        target_indices: Tensor de shape (num_edges,) con índices destino
        num_targets: Número total de neuronas destino

    Returns:
        Tensor de shape (batch, num_targets, dim) con señales agregadas
    """
    batch_size, num_edges, dim = signals.shape
    device = signals.device

    # Método MPS-safe: usar one-hot encoding + matmul
    # One-hot: (num_edges, num_targets)
    one_hot = torch.zeros(num_edges, num_targets, device=device)
    one_hot.scatter_(1, target_indices.unsqueeze(1), 1.0)

    # Reshape signals: (batch, num_edges, dim) -> (batch * dim, num_edges)
    signals_flat = signals.permute(0, 2, 1).reshape(batch_size * dim, num_edges)

    # Matmul: (batch*dim, num_edges) @ (num_edges, num_targets) -> (batch*dim, num_targets)
    aggregated_flat = torch.mm(signals_flat, one_hot)

    # Reshape: (batch*dim, num_targets) -> (batch, dim, num_targets) -> (batch, num_targets, dim)
    aggregated = aggregated_flat.reshape(batch_size, dim, num_targets).permute(0, 2, 1)

    return aggregated


def gather_source_states(
    states: torch.Tensor,
    source_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Obtiene los estados de las neuronas fuente.

    Args:
        states: Tensor de shape (batch, num_neurons, dim)
        source_indices: Tensor de shape (num_edges,)

    Returns:
        Tensor de shape (batch, num_edges, dim)
    """
    batch_size, num_neurons, dim = states.shape
    num_edges = source_indices.shape[0]

    # Expandir índices para gather
    # source_indices: (num_edges,) -> (1, num_edges, 1) -> (batch, num_edges, dim)
    indices_expanded = source_indices.view(1, num_edges, 1).expand(batch_size, -1, dim)

    # Gather
    source_states = torch.gather(states, dim=1, index=indices_expanded)

    return source_states


class EdgeManager:
    """
    Gestiona las conexiones (edges) de una zona o del hub.

    Mantiene los índices de conexiones y facilita operaciones
    de propagación MPS-safe.
    """

    def __init__(
        self,
        num_neurons: int,
        connectivity: float,
        device: torch.device = None,
    ):
        self.num_neurons = num_neurons
        self.connectivity = connectivity
        self.device = device

        # Crear máscara inicial
        self.connectivity_mask = create_random_connectivity_mask(
            num_neurons, num_neurons, connectivity, device=device
        )

        # Extraer índices de edges
        self._update_edge_indices()

    def _update_edge_indices(self):
        """Actualiza los índices de edges desde la máscara."""
        # Encontrar conexiones activas
        edges = self.connectivity_mask.nonzero(as_tuple=False)

        if len(edges) > 0:
            self.source_indices = edges[:, 0]
            self.target_indices = edges[:, 1]
            self.num_edges = len(edges)
        else:
            self.source_indices = torch.tensor([], dtype=torch.long, device=self.device)
            self.target_indices = torch.tensor([], dtype=torch.long, device=self.device)
            self.num_edges = 0

    def add_edge(self, source: int, target: int) -> bool:
        """Añade una conexión si no existe."""
        if not self.connectivity_mask[source, target]:
            self.connectivity_mask[source, target] = True
            self._update_edge_indices()
            return True
        return False

    def remove_edge(self, source: int, target: int) -> bool:
        """Elimina una conexión si existe."""
        if self.connectivity_mask[source, target]:
            self.connectivity_mask[source, target] = False
            self._update_edge_indices()
            return True
        return False

    def get_edges_for_target(self, target: int) -> torch.Tensor:
        """Obtiene los índices de edges que apuntan a un target."""
        return (self.target_indices == target).nonzero(as_tuple=False).squeeze(-1)

    def get_edges_from_source(self, source: int) -> torch.Tensor:
        """Obtiene los índices de edges que salen de un source."""
        return (self.source_indices == source).nonzero(as_tuple=False).squeeze(-1)

    def get_degree_in(self) -> torch.Tensor:
        """Calcula el grado entrante de cada neurona."""
        degree = torch.zeros(self.num_neurons, device=self.device)
        for i in range(self.num_neurons):
            degree[i] = (self.target_indices == i).sum()
        return degree

    def get_degree_out(self) -> torch.Tensor:
        """Calcula el grado saliente de cada neurona."""
        degree = torch.zeros(self.num_neurons, device=self.device)
        for i in range(self.num_neurons):
            degree[i] = (self.source_indices == i).sum()
        return degree
