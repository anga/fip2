"""
Zona cerebral con conexiones internas random y buffer de comunicación.

Cada zona tiene:
- Neuronas internas con conexiones random
- Neuronas buffer para comunicación con el hub
- Estados de activación por neurona
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class Zone(nn.Module):
    """
    Una zona cerebral con conexiones internas aleatorias.

    Las zonas se comunican con el hub central a través de neuronas buffer.
    Las conexiones internas son sparse (máscara fija) pero se implementan
    con operaciones densas para compatibilidad con MPS.
    """

    def __init__(
        self,
        zone_id: int,
        num_neurons: int,
        neuron_dim: int,
        buffer_ratio: float = 0.1,
        connectivity: float = 0.1,
        decay: float = 0.9,
        activation: str = "gelu",
    ):
        super().__init__()
        self.zone_id = zone_id
        self.num_neurons = num_neurons
        self.neuron_dim = neuron_dim
        self.num_buffer = max(1, int(num_neurons * buffer_ratio))
        self.num_internal = num_neurons - self.num_buffer
        self.decay = decay

        # Índices: las primeras num_internal son internas, las últimas son buffer
        self.buffer_start_idx = self.num_internal

        # Pesos de conexiones internas (denso con máscara)
        # Shape: (num_neurons, num_neurons)
        self.internal_weights = nn.Parameter(
            torch.randn(num_neurons, num_neurons) * 0.02
        )

        # Máscara de conectividad (fija, no entrenable)
        # Genera conexiones aleatorias según connectivity ratio
        mask = torch.rand(num_neurons, num_neurons) < connectivity
        # No permitir auto-conexiones
        mask.fill_diagonal_(False)
        self.register_buffer("connectivity_mask", mask.float())

        # Biases por neurona
        self.biases = nn.Parameter(torch.zeros(num_neurons, neuron_dim))

        # Proyección para recibir del hub
        self.hub_receive_proj = nn.Linear(neuron_dim, neuron_dim * self.num_buffer)

        # === Proyección para inyección multi-neurona ===
        # Inyectar en 30% de las neuronas internas (mínimo 20)
        self.num_input_neurons = max(20, int(self.num_internal * 0.3))
        self.input_proj = nn.Linear(neuron_dim, neuron_dim * self.num_input_neurons)

        # === Proyección para output (usa todas las neuronas) ===
        self.output_proj = nn.Linear(num_neurons * neuron_dim, neuron_dim)

        # Función de activación
        self.activation_fn = self._get_activation(activation)

        # Estado persistente de las neuronas (se reinicia cada forward)
        self.register_buffer(
            "states", torch.zeros(1, num_neurons, neuron_dim), persistent=False
        )

        # Estadísticas para plasticidad
        self.register_buffer(
            "activation_history",
            torch.zeros(100, num_neurons),  # Últimas 100 activaciones
            persistent=False,
        )
        self.history_ptr = 0

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
        """Reinicia los estados de las neuronas para un nuevo batch."""
        self.states = torch.zeros(
            batch_size, self.num_neurons, self.neuron_dim, device=device
        )

    def get_effective_weights(self) -> torch.Tensor:
        """Obtiene los pesos efectivos (con máscara aplicada)."""
        return self.internal_weights * self.connectivity_mask

    def internal_propagate(self) -> torch.Tensor:
        """
        Propaga señales dentro de la zona (conexiones internas) con skip connection.

        Returns:
            Nuevos estados de las neuronas
        """
        # states: (batch, num_neurons, dim)
        batch_size = self.states.shape[0]

        # Guardar estado anterior para skip connection
        residual = self.states

        # Obtener pesos efectivos
        effective_weights = self.get_effective_weights()

        # Propagación densa (MPS-safe)
        # Queremos: new_states[b, i, d] = sum_j(weights[j, i] * states[b, j, d])
        # Esto es: states @ weights.T para cada batch

        # Reshape para matmul: (batch, num_neurons, dim) -> (batch * dim, num_neurons)
        states_flat = self.states.permute(0, 2, 1).reshape(-1, self.num_neurons)

        # Matmul: (batch*dim, num_neurons) @ (num_neurons, num_neurons) -> (batch*dim, num_neurons)
        pre_act_flat = torch.mm(states_flat, effective_weights)

        # Reshape back: (batch*dim, num_neurons) -> (batch, dim, num_neurons) -> (batch, num_neurons, dim)
        pre_act = pre_act_flat.reshape(batch_size, self.neuron_dim, self.num_neurons)
        pre_act = pre_act.permute(0, 2, 1)

        # Añadir biases y aplicar activación
        pre_act = pre_act + self.biases.unsqueeze(0)
        activated = self.activation_fn(pre_act)

        # Residual learning correcto: new = old + scale * new_info
        # El decay controla cuánta información nueva se integra
        new_states = residual + (1 - self.decay) * activated

        self.states = new_states
        return new_states

    def get_buffer_output(self) -> torch.Tensor:
        """
        Obtiene la salida de las neuronas buffer para enviar al hub.

        Returns:
            Tensor de shape (batch, num_buffer, dim)
        """
        # Las neuronas buffer son las últimas num_buffer neuronas
        return self.states[:, self.buffer_start_idx:, :]

    def receive_from_hub(self, hub_output: torch.Tensor) -> None:
        """
        Recibe información del hub y la integra en las neuronas buffer.

        Args:
            hub_output: Tensor de shape (batch, hub_dim)
        """
        batch_size = hub_output.shape[0]

        # Proyectar la salida del hub a las neuronas buffer
        # (batch, hub_dim) -> (batch, num_buffer * dim)
        projected = self.hub_receive_proj(hub_output)

        # Reshape a (batch, num_buffer, dim)
        projected = projected.view(batch_size, self.num_buffer, self.neuron_dim)

        # Integrar con las neuronas buffer existentes (alineado con decay)
        buffer_states = self.states[:, self.buffer_start_idx:, :]
        new_buffer_states = self.decay * buffer_states + (1 - self.decay) * projected

        # Actualizar solo las neuronas buffer
        self.states = torch.cat([
            self.states[:, :self.buffer_start_idx, :],
            new_buffer_states
        ], dim=1)

    def inject_input(self, input_tensor: torch.Tensor) -> None:
        """
        Inyecta input externo en múltiples neuronas de la zona.

        Solo se usa en la primera zona (Z1).
        Inyecta en 10% de las neuronas internas para mejor distribución.

        Args:
            input_tensor: Tensor de shape (batch, input_dim)
        """
        batch_size = input_tensor.shape[0]

        if input_tensor.shape[-1] != self.neuron_dim:
            raise ValueError(
                f"Input dim {input_tensor.shape[-1]} != neuron_dim {self.neuron_dim}"
            )

        # Proyectar a múltiples neuronas
        # (batch, dim) -> (batch, num_input_neurons * dim)
        projected = self.input_proj(input_tensor)

        # Reshape a (batch, num_input_neurons, dim)
        projected = projected.view(batch_size, self.num_input_neurons, self.neuron_dim)

        # Inyectar en las primeras num_input_neurons neuronas internas
        self.states[:, :self.num_input_neurons, :] = projected

    def get_output(self) -> torch.Tensor:
        """
        Obtiene la salida de la zona usando todas las neuronas.

        Returns:
            Tensor de shape (batch, dim) combinando todas las neuronas
        """
        batch_size = self.states.shape[0]

        # Aplanar todas las neuronas: (batch, num_neurons, dim) -> (batch, num_neurons * dim)
        flat_states = self.states.view(batch_size, -1)

        # Proyectar a la dimensión de salida
        output = self.output_proj(flat_states)  # (batch, dim)

        return output

    def update_activation_history(self) -> None:
        """Actualiza el historial de activaciones para plasticidad."""
        # Calcular activación media por neurona
        activation_magnitude = self.states.abs().mean(dim=(0, 2))  # (num_neurons,)

        # Guardar en historial circular
        idx = self.history_ptr % self.activation_history.shape[0]
        self.activation_history[idx] = activation_magnitude
        self.history_ptr += 1

    def get_neuron_health(self) -> torch.Tensor:
        """
        Calcula la salud de cada neurona basada en activación y variabilidad.

        Returns:
            Tensor de shape (num_neurons,) con salud en [0, 1]
        """
        history = self.activation_history

        # D: Tasa de muerte (fracción de activaciones ~0)
        death_rate = (history < 0.01).float().mean(dim=0)

        # V: Variabilidad
        mean_act = history.mean(dim=0)
        variance = history.var(dim=0)
        variability = torch.clamp(variance / (mean_act.pow(2) + 1e-6), 0, 1)

        # F: Flujo (activación media)
        flow = mean_act

        # Salud: H = (1 - D) * V * F
        health = (1 - death_rate) * variability * flow

        return health

    def forward(self) -> torch.Tensor:
        """Forward pass: propaga internamente y retorna estados."""
        return self.internal_propagate()
