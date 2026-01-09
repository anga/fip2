"""
PlasticityManager: Integra la plasticidad de edges y neuronas en el entrenamiento.

Enfoque práctico:
- No cambia tamaños de tensores (rompería optimizer state)
- Debilita conexiones de baja utilidad
- Refuerza conexiones de alta utilidad
- "Reinicia" neuronas muertas re-inicializando sus pesos
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PlasticityStats:
    """Estadísticas de un paso de plasticidad."""
    mean_edge_utility: float = 0.0
    mean_neuron_utility: float = 0.0
    edges_weakened: int = 0
    edges_strengthened: int = 0
    edges_killed: int = 0  # Conexiones eliminadas (Hebbian)
    edges_born: int = 0    # Conexiones creadas (Hebbian)
    neurons_refreshed: int = 0
    zone_stats: Dict[int, Dict] = None


class PlasticityManager:
    """
    Gestiona la plasticidad de conexiones y neuronas.

    Características:
    - Rastrea utilidad de edges usando gradientes y flujo de información
    - Ajusta pesos de conexiones según utilidad
    - Detecta neuronas "muertas" y las reinicia
    """

    def __init__(
        self,
        model: nn.Module,
        config,
        device: torch.device,
    ):
        self.model = model
        self.config = config
        self.device = device

        # Parámetros de plasticidad
        self.edge_weaken_factor = 0.95  # Factor para debilitar edges de baja utilidad
        self.edge_strengthen_factor = 1.02  # Factor para reforzar edges de alta utilidad
        # Usar percentiles con límites mínimos para evitar actuar sobre edges neutrales
        self.low_utility_percentile = 10.0  # Bottom 10%
        self.high_utility_percentile = 90.0  # Top 10%
        # Límites absolutos: no actuar si la utilidad está en rango "neutral"
        self.min_utility_to_weaken = 0.25  # Solo debilitar si además está por debajo de esto
        self.min_utility_to_strengthen = 0.30  # Solo reforzar si además está por encima de esto

        # Parámetros de neurona
        self.neuron_death_threshold = config.neuron_death_threshold
        self.neuron_grace_period = config.neuron_grace_period

        # Parámetros de muerte/resurrección de conexiones (Hebbian)
        self.edge_death_threshold = 0.20  # Utilidad por debajo de esto = candidato a muerte
        self.edge_grace_period = 3  # Pasos consecutivos antes de matar
        self.max_edges_to_kill_per_step = 100  # Límite por paso para estabilidad
        self.min_correlation_for_birth = 0.25  # Correlación mínima para crear conexión

        # EMAs de utilidad por zona
        self.zone_edge_utilities: Dict[int, torch.Tensor] = {}
        self.zone_neuron_utilities: Dict[int, torch.Tensor] = {}
        self.ema_decay = config.muc_ema_decay

        # Historial de pesos para calcular estabilidad S(e)
        self.weight_history: Dict[int, torch.Tensor] = {}
        self.weight_history_ptr: Dict[int, int] = {}
        self.weight_history_size = 100  # Últimos 100 valores

        # EMA de flujo de información I(e)
        self.flow_ema: Dict[int, torch.Tensor] = {}

        # Contadores de bajo rendimiento consecutivo por neurona
        self.consecutive_low: Dict[int, torch.Tensor] = {}

        # Contadores de bajo rendimiento consecutivo por CONEXIÓN (para muerte/resurrección)
        self.consecutive_low_edges: Dict[int, torch.Tensor] = {}

        # Estadísticas
        self.total_steps = 0
        self.stats_history: List[PlasticityStats] = []

        # Inicializar
        self._initialize_tracking()

    def _initialize_tracking(self) -> None:
        """Inicializa el tracking de utilidades para cada zona."""
        for i, zone in enumerate(self.model.zones):
            num_neurons = zone.num_neurons

            # Inicializar EMAs en 0 - se llenarán con valores reales
            self.zone_edge_utilities[i] = torch.zeros(
                num_neurons, num_neurons, device=self.device
            )
            self.zone_neuron_utilities[i] = torch.zeros(
                num_neurons, device=self.device
            )
            self.consecutive_low[i] = torch.zeros(
                num_neurons, device=self.device, dtype=torch.long
            )

            # Historial de pesos para S(e) - buffer circular
            self.weight_history[i] = torch.zeros(
                self.weight_history_size, num_neurons, num_neurons, device=self.device
            )
            self.weight_history_ptr[i] = 0

            # EMA de flujo de información I(e)
            self.flow_ema[i] = torch.zeros(
                num_neurons, num_neurons, device=self.device
            )

            # Contador de baja utilidad consecutiva por CONEXIÓN
            self.consecutive_low_edges[i] = torch.zeros(
                num_neurons, num_neurons, device=self.device, dtype=torch.long
            )

        # Flag para saber si es el primer paso
        self._first_step = True

    def _update_weight_history(self, zone_idx: int, zone: nn.Module) -> None:
        """Actualiza el historial de pesos para calcular estabilidad."""
        weights = zone.internal_weights.detach()
        idx = self.weight_history_ptr[zone_idx] % self.weight_history_size
        self.weight_history[zone_idx][idx] = weights
        self.weight_history_ptr[zone_idx] += 1

    def _compute_stability(self, zone_idx: int, zone: nn.Module) -> torch.Tensor:
        """
        Calcula la estabilidad S(e) basada en varianza de pesos.

        S(e) = 1 - Var(w) / |w̄|

        Un peso estable tiene baja varianza relativa a su magnitud.
        """
        num_neurons = zone.num_neurons
        ptr = self.weight_history_ptr[zone_idx]

        # Si no hay suficiente historial, asumir estable
        if ptr < 10:
            return torch.ones(num_neurons, num_neurons, device=self.device)

        # Usar el historial disponible
        if ptr >= self.weight_history_size:
            history = self.weight_history[zone_idx]  # (100, n, n)
        else:
            history = self.weight_history[zone_idx][:ptr]  # (ptr, n, n)

        # Varianza y media sobre el tiempo
        weight_var = history.var(dim=0)  # (n, n)
        weight_mean = history.mean(dim=0).abs() + 1e-6  # (n, n)

        # S(e) = 1 - Var(w) / |w̄|, clampado a [0, 1]
        stability = 1 - (weight_var / weight_mean).clamp(0, 2)
        stability = stability.clamp(0, 1)

        return stability

    def _compute_information_flow(self, zone_idx: int, zone: nn.Module) -> torch.Tensor:
        """
        Calcula el flujo de información I(e) como correlación entre activaciones.

        Para cada edge (i, j), calcula la correlación de Pearson entre
        las activaciones de la neurona fuente i y la neurona destino j.
        """
        num_neurons = zone.num_neurons

        # states: (batch, num_neurons, dim)
        if zone.states.shape[0] == 0:
            return torch.zeros(num_neurons, num_neurons, device=self.device)

        # Aplanar batch y dim para calcular correlación
        # states: (batch, num_neurons, dim) -> (batch * dim, num_neurons)
        states_flat = zone.states.permute(0, 2, 1).reshape(-1, num_neurons)

        # Normalizar (restar media, dividir por std)
        states_centered = states_flat - states_flat.mean(dim=0, keepdim=True)
        states_std = states_flat.std(dim=0, keepdim=True) + 1e-6
        states_norm = states_centered / states_std  # (batch*dim, num_neurons)

        # Matriz de correlación: (num_neurons, num_neurons)
        # corr[i, j] = correlación entre neurona i y neurona j
        n_samples = states_norm.shape[0]
        correlation = torch.mm(states_norm.t(), states_norm) / n_samples

        # El flujo de información es el valor absoluto de la correlación
        # (correlación negativa también indica flujo de información)
        flow = correlation.abs()

        # Actualizar EMA
        self.flow_ema[zone_idx] = (
            self.ema_decay * self.flow_ema[zone_idx]
            + (1 - self.ema_decay) * flow
        )

        return self.flow_ema[zone_idx]

    def compute_edge_utility(
        self,
        zone_idx: int,
        zone: nn.Module,
    ) -> torch.Tensor:
        """
        Calcula la utilidad de cada edge en una zona.

        U(e) = α * G(e) + β * I(e) + γ * S(e)

        Donde:
        - G(e): Sensibilidad al gradiente (EMA de |∂L/∂w|)
        - I(e): Flujo de información (correlación entre activaciones)
        - S(e): Estabilidad temporal (1 - varianza relativa del peso)
        """
        weights = zone.internal_weights
        mask = zone.connectivity_mask

        # Actualizar historial de pesos para S(e)
        self._update_weight_history(zone_idx, zone)

        # G(e): Sensibilidad al gradiente
        if weights.grad is not None:
            G = weights.grad.abs()
            G = G / (G.max() + 1e-6)  # Normalizar a [0, 1]
        else:
            G = torch.zeros_like(weights)

        # I(e): Flujo de información (correlación de activaciones)
        I = self._compute_information_flow(zone_idx, zone)
        I = I / (I.max() + 1e-6)  # Normalizar a [0, 1]

        # S(e): Estabilidad temporal (varianza de pesos)
        S = self._compute_stability(zone_idx, zone)
        # Ya está en [0, 1]

        # Utilidad combinada
        alpha = self.config.muc_alpha
        beta = self.config.muc_beta
        gamma = self.config.muc_gamma

        utility = alpha * G + beta * I + gamma * S

        # Aplicar máscara (solo edges activos)
        utility = utility * mask

        return utility

    def compute_neuron_utility(
        self,
        zone_idx: int,
        zone: nn.Module,
        edge_utility: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calcula la utilidad de cada neurona.

        U_N(n) = α * H(n) + β * C(n) + γ * R(n)
        """
        num_neurons = zone.num_neurons

        # H: Salud (de activation_history)
        H = zone.get_neuron_health()
        H = H / (H.max() + 1e-6)

        # C: Conectividad (utilidad promedio de edges conectados)
        # Promedio de edges entrantes + salientes
        incoming_utility = edge_utility.sum(dim=0) / (zone.connectivity_mask.sum(dim=0) + 1e-6)
        outgoing_utility = edge_utility.sum(dim=1) / (zone.connectivity_mask.sum(dim=1) + 1e-6)
        C = (incoming_utility + outgoing_utility) / 2
        C = C / (C.max() + 1e-6)

        # R: Unicidad (1 - max correlación con otras)
        # Usar estados actuales de la zona
        if zone.states.shape[0] > 0:
            states_mean = zone.states.mean(dim=0)  # (num_neurons, dim)
            norms = states_mean.norm(dim=1, keepdim=True) + 1e-6
            normalized = states_mean / norms
            corr = torch.mm(normalized, normalized.t())
            corr.fill_diagonal_(0)
            max_corr = corr.abs().max(dim=1).values
            R = 1 - max_corr
        else:
            R = torch.ones(num_neurons, device=self.device)

        # Utilidad total
        alpha = self.config.neuron_alpha
        beta = self.config.neuron_beta
        gamma = self.config.neuron_gamma

        utility = alpha * H + beta * C + gamma * R

        return utility

    def _compute_hebbian_candidates(
        self,
        zone_idx: int,
        zone: nn.Module,
        num_candidates: int = 50,
    ) -> List[Tuple[int, int, float]]:
        """
        Encuentra pares de neuronas no conectadas con alta correlación (Hebbian).

        "Neurons that fire together, wire together"

        Returns:
            Lista de (source_idx, target_idx, correlation) ordenada por correlación
        """
        num_neurons = zone.num_neurons
        mask = zone.connectivity_mask

        # Necesitamos estados para calcular correlación
        if zone.states.shape[0] == 0:
            return []

        # Calcular matriz de correlación
        states_flat = zone.states.permute(0, 2, 1).reshape(-1, num_neurons)
        states_centered = states_flat - states_flat.mean(dim=0, keepdim=True)
        states_std = states_flat.std(dim=0, keepdim=True) + 1e-6
        states_norm = states_centered / states_std

        n_samples = states_norm.shape[0]
        correlation = torch.mm(states_norm.t(), states_norm) / n_samples

        # Enmascarar: solo nos interesan pares NO conectados
        # También excluir diagonal (auto-conexiones)
        not_connected = (mask == 0)
        not_connected.fill_diagonal_(False)

        # Correlación absoluta de pares no conectados
        candidates_corr = correlation.abs() * not_connected.float()

        # Obtener los top candidatos
        flat_corr = candidates_corr.flatten()
        k = min(num_candidates * 2, (candidates_corr > self.min_correlation_for_birth).sum().item())

        if k == 0:
            return []

        top_values, top_indices = torch.topk(flat_corr, k)

        # Filtrar por umbral mínimo de correlación
        valid_mask = top_values >= self.min_correlation_for_birth
        top_values = top_values[valid_mask]
        top_indices = top_indices[valid_mask]

        # Convertir a coordenadas (i, j)
        candidates = []
        for idx, corr_val in zip(top_indices.tolist(), top_values.tolist()):
            i = idx // num_neurons
            j = idx % num_neurons
            candidates.append((i, j, corr_val))

        return candidates[:num_candidates]

    def _apply_edge_death_resurrection(
        self,
        zone_idx: int,
        zone: nn.Module,
        edge_utility: torch.Tensor,
    ) -> Tuple[int, int]:
        """
        Aplica muerte de conexiones malas y resurrección en nuevas ubicaciones.

        Proceso:
        1. Identifica edges con utilidad baja por varios pasos consecutivos
        2. Los "mata" (pone máscara a 0 y peso a 0)
        3. Encuentra candidatos Hebbian (pares correlacionados sin conexión)
        4. Crea nuevas conexiones en esos lugares

        Returns:
            (edges_killed, edges_born)
        """
        weights = zone.internal_weights
        mask = zone.connectivity_mask
        num_neurons = zone.num_neurons

        active_mask = mask > 0
        if active_mask.sum() == 0:
            return 0, 0

        # === Paso 1: Actualizar contadores de baja utilidad ===
        low_utility = (edge_utility < self.edge_death_threshold) & active_mask
        high_utility = ~low_utility & active_mask

        # Incrementar contador para edges de baja utilidad
        self.consecutive_low_edges[zone_idx][low_utility] += 1
        # Resetear contador para edges que están bien
        self.consecutive_low_edges[zone_idx][high_utility] = 0
        # Los no activos no cuentan
        self.consecutive_low_edges[zone_idx][~active_mask] = 0

        # === Paso 2: Identificar edges a matar ===
        to_kill = (
            (self.consecutive_low_edges[zone_idx] >= self.edge_grace_period) &
            active_mask
        )

        # Limitar cuántos matamos por paso (para estabilidad)
        kill_indices = to_kill.nonzero(as_tuple=False)
        if len(kill_indices) > self.max_edges_to_kill_per_step:
            # Ordenar por utilidad más baja primero
            utilities_to_kill = edge_utility[kill_indices[:, 0], kill_indices[:, 1]]
            _, sort_idx = utilities_to_kill.sort()
            kill_indices = kill_indices[sort_idx[:self.max_edges_to_kill_per_step]]
            # Reconstruir máscara
            to_kill = torch.zeros_like(to_kill)
            to_kill[kill_indices[:, 0], kill_indices[:, 1]] = True

        edges_killed = to_kill.sum().item()

        # === Paso 3: Encontrar candidatos para resurrección ===
        hebbian_candidates = self._compute_hebbian_candidates(
            zone_idx, zone, num_candidates=edges_killed + 10
        )

        # === Paso 4: Aplicar muerte y resurrección ===
        edges_born = 0

        with torch.no_grad():
            # Matar conexiones malas
            if edges_killed > 0:
                # Poner peso a 0
                weights.data[to_kill] = 0.0
                # Modificar máscara (esto es lo nuevo - muerte real)
                mask.data[to_kill] = 0.0
                # Resetear contador
                self.consecutive_low_edges[zone_idx][to_kill] = 0

            # Resucitar en nuevas ubicaciones (Hebbian)
            for i, j, corr in hebbian_candidates:
                if edges_born >= edges_killed:
                    break  # No crear más de las que matamos

                # Verificar que sigue sin conexión
                if mask[i, j] > 0:
                    continue

                # Crear conexión con peso proporcional a correlación
                init_weight = corr * 0.1  # Peso inicial pequeño basado en correlación
                weights.data[i, j] = init_weight
                mask.data[i, j] = 1.0  # Activar en máscara
                edges_born += 1

        return edges_killed, edges_born

    def apply_edge_plasticity(
        self,
        zone_idx: int,
        zone: nn.Module,
        edge_utility: torch.Tensor,
    ) -> Tuple[int, int, int, int]:
        """
        Aplica plasticidad a los edges de una zona.

        Incluye:
        - Debilitar/reforzar edges existentes
        - Muerte y resurrección Hebbian de conexiones

        Returns:
            (edges_weakened, edges_strengthened, edges_killed, edges_born)
        """
        weights = zone.internal_weights
        mask = zone.connectivity_mask

        # Obtener utilidades de edges activos
        active_mask = mask > 0
        if active_mask.sum() == 0:
            return 0, 0, 0, 0

        active_utilities = edge_utility[active_mask]

        # Calcular umbrales por percentil
        low_threshold = torch.quantile(active_utilities, self.low_utility_percentile / 100)
        high_threshold = torch.quantile(active_utilities, self.high_utility_percentile / 100)

        # Identificar edges: percentil + límite absoluto
        # Debilitar: está en el bottom 10% Y tiene utilidad < 0.25
        low_utility_mask = (
            (edge_utility < low_threshold) &
            (edge_utility < self.min_utility_to_weaken) &
            active_mask
        )
        # Reforzar: está en el top 10% Y tiene utilidad > 0.30
        high_utility_mask = (
            (edge_utility > high_threshold) &
            (edge_utility > self.min_utility_to_strengthen) &
            active_mask
        )

        # Aplicar factores (sin gradiente)
        with torch.no_grad():
            # Debilitar edges de baja utilidad
            weights.data[low_utility_mask] *= self.edge_weaken_factor

            # Reforzar edges de alta utilidad (con límite)
            weights.data[high_utility_mask] *= self.edge_strengthen_factor
            weights.data.clamp_(-2.0, 2.0)  # Evitar explosión

        edges_weakened = low_utility_mask.sum().item()
        edges_strengthened = high_utility_mask.sum().item()

        # === Muerte y Resurrección Hebbian ===
        edges_killed, edges_born = self._apply_edge_death_resurrection(
            zone_idx, zone, edge_utility
        )

        return edges_weakened, edges_strengthened, edges_killed, edges_born

    def apply_neuron_plasticity(
        self,
        zone_idx: int,
        zone: nn.Module,
        neuron_utility: torch.Tensor,
    ) -> int:
        """
        Aplica plasticidad a las neuronas de una zona.

        "Reinicia" neuronas muertas re-inicializando sus pesos.

        Returns:
            neurons_refreshed
        """
        num_neurons = zone.num_neurons
        threshold = self.neuron_death_threshold

        # Actualizar contadores de bajo rendimiento
        low_utility = neuron_utility < threshold
        self.consecutive_low[zone_idx][low_utility] += 1
        self.consecutive_low[zone_idx][~low_utility] = 0

        # Identificar neuronas a reiniciar
        to_refresh = self.consecutive_low[zone_idx] >= self.neuron_grace_period

        # No reiniciar neuronas buffer (son importantes para comunicación)
        to_refresh[zone.buffer_start_idx:] = False

        neurons_refreshed = 0

        if to_refresh.any():
            with torch.no_grad():
                refresh_indices = to_refresh.nonzero(as_tuple=False).squeeze(-1)

                for idx in refresh_indices:
                    idx = idx.item()

                    # Reiniciar pesos entrantes y salientes
                    zone.internal_weights.data[idx, :] = torch.randn(
                        num_neurons, device=self.device
                    ) * 0.02
                    zone.internal_weights.data[:, idx] = torch.randn(
                        num_neurons, device=self.device
                    ) * 0.02

                    # Reiniciar bias
                    zone.biases.data[idx] = torch.zeros(
                        zone.neuron_dim, device=self.device
                    )

                    # Resetear contador
                    self.consecutive_low[zone_idx][idx] = 0

                    neurons_refreshed += 1

        return neurons_refreshed

    def step(self) -> PlasticityStats:
        """
        Ejecuta un paso de plasticidad en todo el modelo.

        Returns:
            Estadísticas del paso
        """
        self.total_steps += 1

        total_edges_weakened = 0
        total_edges_strengthened = 0
        total_edges_killed = 0
        total_edges_born = 0
        total_neurons_refreshed = 0
        edge_utilities = []
        neuron_utilities = []
        zone_stats = {}

        for i, zone in enumerate(self.model.zones):
            # Actualizar historial de activaciones
            zone.update_activation_history()

            # Calcular utilidades
            edge_utility = self.compute_edge_utility(i, zone)
            neuron_utility = self.compute_neuron_utility(i, zone, edge_utility)

            # Actualizar EMAs (o inicializar en el primer paso)
            if self._first_step:
                self.zone_edge_utilities[i] = edge_utility.clone()
                self.zone_neuron_utilities[i] = neuron_utility.clone()
            else:
                self.zone_edge_utilities[i] = (
                    self.ema_decay * self.zone_edge_utilities[i]
                    + (1 - self.ema_decay) * edge_utility
                )
                self.zone_neuron_utilities[i] = (
                    self.ema_decay * self.zone_neuron_utilities[i]
                    + (1 - self.ema_decay) * neuron_utility
                )

            # Aplicar plasticidad (ahora incluye muerte/resurrección Hebbian)
            weakened, strengthened, killed, born = self.apply_edge_plasticity(
                i, zone, self.zone_edge_utilities[i]
            )
            refreshed = self.apply_neuron_plasticity(
                i, zone, self.zone_neuron_utilities[i]
            )

            # Acumular estadísticas
            total_edges_weakened += weakened
            total_edges_strengthened += strengthened
            total_edges_killed += killed
            total_edges_born += born
            total_neurons_refreshed += refreshed

            active_mask = zone.connectivity_mask > 0
            if active_mask.any():
                edge_utilities.append(edge_utility[active_mask].mean().item())
            neuron_utilities.append(neuron_utility.mean().item())

            zone_stats[i] = {
                "edge_utility": edge_utilities[-1] if edge_utilities else 0,
                "neuron_utility": neuron_utilities[-1],
                "edges_weakened": weakened,
                "edges_strengthened": strengthened,
                "edges_killed": killed,
                "edges_born": born,
                "neurons_refreshed": refreshed,
            }

        stats = PlasticityStats(
            mean_edge_utility=sum(edge_utilities) / len(edge_utilities) if edge_utilities else 0,
            mean_neuron_utility=sum(neuron_utilities) / len(neuron_utilities) if neuron_utilities else 0,
            edges_weakened=total_edges_weakened,
            edges_strengthened=total_edges_strengthened,
            edges_killed=total_edges_killed,
            edges_born=total_edges_born,
            neurons_refreshed=total_neurons_refreshed,
            zone_stats=zone_stats,
        )

        self.stats_history.append(stats)

        # Ya no es el primer paso
        self._first_step = False

        return stats

    def get_summary(self) -> Dict:
        """Obtiene un resumen de las estadísticas de plasticidad."""
        if not self.stats_history:
            return {}

        recent = self.stats_history[-100:]  # Últimos 100 pasos

        return {
            "total_steps": self.total_steps,
            "avg_edge_utility": sum(s.mean_edge_utility for s in recent) / len(recent),
            "avg_neuron_utility": sum(s.mean_neuron_utility for s in recent) / len(recent),
            "total_edges_weakened": sum(s.edges_weakened for s in recent),
            "total_edges_strengthened": sum(s.edges_strengthened for s in recent),
            "total_edges_killed": sum(s.edges_killed for s in recent),
            "total_edges_born": sum(s.edges_born for s in recent),
            "total_neurons_refreshed": sum(s.neurons_refreshed for s in recent),
        }
