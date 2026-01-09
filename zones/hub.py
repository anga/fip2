"""
Hub central que integra información de todas las zonas.

El hub:
- Recibe información de todas las zonas (a través de sus buffers)
- Tiene conexiones internas propias para procesar
- Distribuye información procesada de vuelta a las zonas
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class Hub(nn.Module):
    """
    Hub central que conecta todas las zonas.

    Funciona como un "tálamo" que integra y redistribuye información.

    Arquitectura eficiente:
    - Cada zona tiene su propia proyección pequeña al hub
    - Las proyecciones se agregan (sum)
    - El hub procesa internamente con conexiones sparse
    - Una proyección de salida genera la señal para las zonas
    """

    def __init__(
        self,
        num_neurons: int,
        neuron_dim: int,
        num_zones: int,
        buffer_neurons_per_zone: int,
        connectivity: float = 0.15,
        decay: float = 0.9,
        activation: str = "gelu",
    ):
        super().__init__()
        self.num_neurons = num_neurons
        self.neuron_dim = neuron_dim
        self.num_zones = num_zones
        self.buffer_neurons_per_zone = buffer_neurons_per_zone
        self.decay = decay

        # === Proyecciones de entrada (una por zona, PEQUEÑAS) ===
        # Cada una: (buffer_neurons * dim) -> (neuron_dim)
        # Parámetros por zona: buffer_neurons * dim * neuron_dim
        # Con buffer=20, dim=64: 20 * 64 * 64 = 81,920 por zona
        self.zone_projections = nn.ModuleList([
            nn.Linear(buffer_neurons_per_zone * neuron_dim, neuron_dim)
            for _ in range(num_zones)
        ])

        # === Pesos aprendibles para agregación de zonas ===
        # Permite al modelo aprender qué zonas son más importantes
        self.zone_weights = nn.Parameter(torch.ones(num_zones) / num_zones)

        # === Proyección de integración ===
        # Combina las proyecciones de todas las zonas
        # (num_zones * neuron_dim) -> (num_neurons * neuron_dim)
        # Pero usamos una versión más eficiente:
        # Primero hacemos weighted sum de las proyecciones, luego expandimos
        self.integration_proj = nn.Linear(neuron_dim, num_neurons)

        # === Pesos de conexiones internas del hub ===
        self.internal_weights = nn.Parameter(
            torch.randn(num_neurons, num_neurons) * 0.02
        )

        # Máscara de conectividad interna
        mask = torch.rand(num_neurons, num_neurons) < connectivity
        mask.fill_diagonal_(False)
        self.register_buffer("connectivity_mask", mask.float())

        # Biases
        self.biases = nn.Parameter(torch.zeros(num_neurons, neuron_dim))

        # === Proyección de salida ===
        # (num_neurons) -> (neuron_dim) para enviar a cada zona
        self.output_proj = nn.Linear(num_neurons, neuron_dim)

        # Función de activación
        self.activation_fn = self._get_activation(activation)

        # Estado del hub
        self.register_buffer(
            "states", torch.zeros(1, num_neurons, neuron_dim), persistent=False
        )

    def _get_activation(self, name: str):
        """Obtiene la función de activación por nombre."""
        activations = {
            "relu": F.relu,
            "gelu": F.gelu,
            "silu": F.silu,
            "tanh": torch.tanh,
        }
        return activations.get(name, F.gelu)

    def reset_states(self, batch_size: int, device: torch.device) -> None:
        """Reinicia los estados del hub."""
        self.states = torch.zeros(
            batch_size, self.num_neurons, self.neuron_dim, device=device
        )

    def get_effective_weights(self) -> torch.Tensor:
        """Obtiene los pesos efectivos (con máscara aplicada)."""
        return self.internal_weights * self.connectivity_mask

    def receive_from_zones(self, zone_buffers: List[torch.Tensor]) -> None:
        """
        Recibe información de los buffers de todas las zonas con weighted sum.

        Args:
            zone_buffers: Lista de tensores, cada uno (batch, num_buffer, dim)
        """
        batch_size = zone_buffers[0].shape[0]
        device = zone_buffers[0].device

        # Proyectar cada buffer de zona individualmente
        zone_embeddings = []
        for i, buffer in enumerate(zone_buffers):
            # Flatten buffer: (batch, num_buffer, dim) -> (batch, num_buffer * dim)
            buffer_flat = buffer.reshape(batch_size, -1)
            # Proyectar: (batch, num_buffer * dim) -> (batch, neuron_dim)
            projected = self.zone_projections[i](buffer_flat)
            zone_embeddings.append(projected)

        # Weighted sum con pesos aprendibles (softmax para normalizar)
        # zone_embeddings: lista de (batch, neuron_dim)
        weights = F.softmax(self.zone_weights, dim=0)  # (num_zones,)
        stacked = torch.stack(zone_embeddings, dim=0)  # (num_zones, batch, neuron_dim)
        aggregated = (stacked * weights.view(-1, 1, 1)).sum(dim=0)  # (batch, neuron_dim)

        # Expandir a todos los neurons del hub: (batch, neuron_dim) -> (batch, num_neurons)
        expanded = self.integration_proj(aggregated)  # (batch, num_neurons)

        # Broadcast a todas las dimensiones: (batch, num_neurons) -> (batch, num_neurons, neuron_dim)
        expanded = expanded.unsqueeze(-1).expand(-1, -1, self.neuron_dim)

        # Integrar con estados existentes (alineado con decay)
        self.states = self.decay * self.states + (1 - self.decay) * expanded

    def internal_propagate(self) -> torch.Tensor:
        """
        Propaga señales dentro del hub con skip connection.

        Returns:
            Nuevos estados del hub
        """
        batch_size = self.states.shape[0]

        # Guardar estado anterior para skip connection
        residual = self.states

        # Obtener pesos efectivos
        effective_weights = self.get_effective_weights()

        # Propagación densa (MPS-safe)
        states_flat = self.states.permute(0, 2, 1).reshape(-1, self.num_neurons)
        pre_act_flat = torch.mm(states_flat, effective_weights)
        pre_act = pre_act_flat.reshape(batch_size, self.neuron_dim, self.num_neurons)
        pre_act = pre_act.permute(0, 2, 1)

        # Añadir biases y activación
        pre_act = pre_act + self.biases.unsqueeze(0)
        activated = self.activation_fn(pre_act)

        # Residual learning correcto: new = old + scale * new_info
        new_states = residual + (1 - self.decay) * activated

        self.states = new_states
        return new_states

    def get_output_for_zones(self) -> torch.Tensor:
        """
        Genera la señal de salida para distribuir a todas las zonas.

        Returns:
            Tensor de shape (batch, dim) que se enviará a cada zona
        """
        batch_size = self.states.shape[0]

        # Promediar sobre dimensión: (batch, num_neurons, dim) -> (batch, num_neurons)
        states_pooled = self.states.mean(dim=-1)

        # Proyectar a salida: (batch, num_neurons) -> (batch, neuron_dim)
        output = self.output_proj(states_pooled)

        return output

    def process(self, zone_buffers: List[torch.Tensor]) -> torch.Tensor:
        """
        Proceso completo del hub:
        1. Recibe de zonas
        2. Propaga internamente
        3. Genera salida

        Args:
            zone_buffers: Lista de buffers de cada zona

        Returns:
            Señal de salida para distribuir a las zonas
        """
        self.receive_from_zones(zone_buffers)
        self.internal_propagate()
        return self.get_output_for_zones()

    def forward(self, zone_buffers: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass: procesa y retorna salida."""
        return self.process(zone_buffers)
