"""
Modelo principal FIP2: ZonedBrainModel

Arquitectura de zonas cerebrales con hub central obligatorio.
Diseñado para ser MPS-safe (Apple Silicon).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict

from .config import FIP2Config
from .zones import Zone, Hub
from .utils import get_device


class ZonedBrainModel(nn.Module):
    """
    Modelo de lenguaje basado en zonas cerebrales con hub central.

    Arquitectura:
        INPUT -> Z1 -> HUB -> Z2 -> HUB -> Z3 -> HUB -> Z4 -> HUB -> Z5 -> OUTPUT

    Todas las zonas se comunican a través del hub central (obligatorio).
    """

    def __init__(self, config: FIP2Config):
        super().__init__()
        self.config = config

        # Determinar dispositivo
        self.device = get_device(config.device)

        # === Embedding de entrada ===
        # Byte-level: 256 tokens posibles
        self.input_embedding = nn.Embedding(config.vocab_size, config.neuron_dim)

        # Embedding posicional
        self.pos_embedding = nn.Embedding(config.context_length, config.neuron_dim)

        # === Crear zonas ===
        self.zones = nn.ModuleList([
            Zone(
                zone_id=i,
                num_neurons=config.neurons_per_zone,
                neuron_dim=config.neuron_dim,
                buffer_ratio=config.buffer_ratio,
                connectivity=config.intra_zone_connectivity,
                decay=config.decay,
                activation=config.activation,
            )
            for i in range(config.num_zones)
        ])

        # === Inyección de contexto multi-token ===
        # Número de tokens del contexto a inyectar directamente
        # Más tokens = más contexto = mejor predicción
        self.num_inject_tokens = min(32, config.context_length)
        # Proyección: concatenación de últimos N tokens -> neuronas de input de Z1
        self.context_proj = nn.Linear(
            config.neuron_dim * self.num_inject_tokens,
            config.neuron_dim * self.zones[0].num_input_neurons
        )

        # === Crear hub central ===
        self.hub = Hub(
            num_neurons=config.hub_neurons,
            neuron_dim=config.neuron_dim,
            num_zones=config.num_zones,
            buffer_neurons_per_zone=config.buffer_neurons_per_zone,
            connectivity=0.15,  # Hub más denso que zonas
            decay=config.decay,
            activation=config.activation,
        )

        # === Attention para salida ===
        # Z5 output actúa como query, embeddings del contexto como key/value
        self.output_key = nn.Linear(config.neuron_dim, config.neuron_dim)
        self.output_value = nn.Linear(config.neuron_dim, config.neuron_dim)
        self.output_query = nn.Linear(config.neuron_dim, config.neuron_dim)

        # Combinar Z5 output + attention output antes de proyectar a vocab
        self.output_combine = nn.Linear(config.neuron_dim * 2, config.neuron_dim)

        # === Proyección de salida ===
        # De la última zona al vocabulario
        self.output_proj = nn.Linear(config.neuron_dim, config.vocab_size)

        # === Cache para embeddings del contexto ===
        self.context_embeddings: Optional[torch.Tensor] = None

        # === Estadísticas para pérdida de diversidad ===
        self.zone_outputs_cache: List[torch.Tensor] = []

        # Mover a dispositivo
        self.to(self.device)

    def reset_all_states(self, batch_size: int) -> None:
        """Reinicia todos los estados para un nuevo batch."""
        for zone in self.zones:
            zone.reset_states(batch_size, self.device)
        self.hub.reset_states(batch_size, self.device)
        self.zone_outputs_cache = []

    def forward_wave(self) -> None:
        """
        Ejecuta una wave de propagación completa:
        1. Propagar dentro de cada zona
        2. Enviar buffers al hub
        3. Procesar en hub
        4. Distribuir del hub a zonas
        """
        # 1. Propagación interna en cada zona
        for zone in self.zones:
            zone.internal_propagate()

        # 2. Recolectar buffers de todas las zonas
        zone_buffers = [zone.get_buffer_output() for zone in self.zones]

        # 3. Procesar en hub
        hub_output = self.hub.process(zone_buffers)

        # 4. Distribuir a todas las zonas
        for zone in self.zones:
            zone.receive_from_hub(hub_output)

    def inject_context(self, input_ids: torch.Tensor) -> None:
        """
        Inyecta el contexto de entrada en la primera zona usando multi-token projection.

        Toma los últimos N tokens y los proyecta directamente a las neuronas de input
        de Z1, preservando información secuencial.

        Args:
            input_ids: Tensor de shape (batch, seq_len) con IDs de tokens
        """
        batch_size, seq_len = input_ids.shape

        # Embeddings de tokens y posiciones
        positions = torch.arange(seq_len, device=self.device)
        token_emb = self.input_embedding(input_ids)  # (batch, seq, dim)
        pos_emb = self.pos_embedding(positions)  # (seq, dim)

        # Combinar embeddings
        embeddings = token_emb + pos_emb.unsqueeze(0)  # (batch, seq, dim)

        # Guardar embeddings para attention en la salida
        self.context_embeddings = embeddings

        # === Multi-token injection ===
        # Tomar los últimos num_inject_tokens embeddings (con padding si es necesario)
        if seq_len >= self.num_inject_tokens:
            last_tokens = embeddings[:, -self.num_inject_tokens:, :]  # (batch, N, dim)
        else:
            # Padding con zeros si el contexto es muy corto
            pad_len = self.num_inject_tokens - seq_len
            padding = torch.zeros(batch_size, pad_len, self.config.neuron_dim, device=self.device)
            last_tokens = torch.cat([padding, embeddings], dim=1)  # (batch, N, dim)

        # Concatenar: (batch, N, dim) -> (batch, N * dim)
        concat_tokens = last_tokens.reshape(batch_size, -1)

        # Proyectar a las neuronas de input de Z1
        # (batch, N * dim) -> (batch, num_input_neurons * dim)
        projected = self.context_proj(concat_tokens)

        # Reshape a (batch, num_input_neurons, dim)
        num_input = self.zones[0].num_input_neurons
        projected = projected.view(batch_size, num_input, self.config.neuron_dim)

        # Inyectar directamente en Z1 (sobrescribe las primeras neuronas)
        self.zones[0].states[:, :num_input, :] = projected

    def get_output_logits(self) -> torch.Tensor:
        """
        Obtiene los logits de salida usando attention sobre el contexto original.

        Z5 output actúa como query, los embeddings del contexto como key/value.
        Esto permite al modelo "mirar" el contexto al momento de predecir.

        Returns:
            Tensor de shape (batch, vocab_size)
        """
        # Obtener salida de la última zona
        z5_output = self.zones[-1].get_output()  # (batch, dim)

        # === Attention sobre el contexto original ===
        # Query: desde Z5 output
        query = self.output_query(z5_output).unsqueeze(1)  # (batch, 1, dim)

        # Key y Value: desde los embeddings guardados
        keys = self.output_key(self.context_embeddings)  # (batch, seq, dim)
        values = self.output_value(self.context_embeddings)  # (batch, seq, dim)

        # Attention scores
        scale = self.config.neuron_dim ** 0.5
        attn_scores = torch.bmm(query, keys.transpose(1, 2)) / scale  # (batch, 1, seq)
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Weighted sum de values
        context_output = torch.bmm(attn_weights, values).squeeze(1)  # (batch, dim)

        # Combinar Z5 output + context attention output
        combined = torch.cat([z5_output, context_output], dim=-1)  # (batch, dim*2)
        output_repr = self.output_combine(combined)  # (batch, dim)

        # Proyectar a vocabulario
        logits = self.output_proj(output_repr)  # (batch, vocab_size)

        return logits

    def collect_zone_outputs(self) -> List[torch.Tensor]:
        """Recolecta las salidas de cada zona para la pérdida de diversidad."""
        outputs = []
        for zone in self.zones:
            output = zone.get_output()  # (batch, dim)
            outputs.append(output)
        return outputs

    def forward(
        self,
        input_ids: torch.Tensor,
        return_zone_outputs: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass completo.

        Args:
            input_ids: Tensor de shape (batch, seq_len) con IDs de tokens
            return_zone_outputs: Si retornar salidas de zonas para diversidad

        Returns:
            logits: Tensor de shape (batch, vocab_size)
            zone_outputs: Lista de tensores (batch, dim) si return_zone_outputs=True
        """
        batch_size = input_ids.shape[0]

        # 1. Reiniciar estados
        self.reset_all_states(batch_size)

        # 2. Inyectar contexto en Z1
        self.inject_context(input_ids)

        # 3. Ejecutar waves de propagación
        for wave_idx in range(self.config.num_waves):
            self.forward_wave()

        # 4. Obtener logits
        logits = self.get_output_logits()

        # 5. Opcionalmente recolectar salidas de zonas
        zone_outputs = None
        if return_zone_outputs:
            zone_outputs = self.collect_zone_outputs()

        return logits, zone_outputs

    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """
        Genera texto autoregressivamente.

        Args:
            prompt_ids: Tensor de shape (1, prompt_len) con el prompt
            max_new_tokens: Número máximo de tokens a generar
            temperature: Temperatura para sampling
            top_k: Top-k filtering

        Returns:
            Tensor de shape (1, prompt_len + generated_len)
        """
        self.eval()
        prompt_len = prompt_ids.shape[1]

        # Pre-allocar tensor para evitar múltiples torch.cat (fuga de memoria)
        generated = torch.zeros(
            1, prompt_len + max_new_tokens,
            dtype=prompt_ids.dtype, device=self.device
        )
        generated[:, :prompt_len] = prompt_ids
        actual_len = prompt_len

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Tomar los últimos context_length tokens
                start_idx = max(0, actual_len - self.config.context_length)
                context = generated[:, start_idx:actual_len]

                # Forward pass
                logits, _ = self.forward(context)

                # Aplicar temperatura
                logits = logits / temperature

                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')

                # Sampling
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Añadir al resultado (sin torch.cat)
                generated[:, actual_len] = next_token.squeeze()
                actual_len += 1

                # Solo parar en byte 0 (null), no en newline
                if next_token.item() == 0:
                    break

        self.train()
        # Retornar solo la parte generada
        return generated[:, :actual_len]

    def get_num_parameters(self) -> Dict[str, int]:
        """Cuenta parámetros por componente."""
        counts = {
            "embedding": sum(p.numel() for p in self.input_embedding.parameters())
                       + sum(p.numel() for p in self.pos_embedding.parameters()),
            "zones": sum(sum(p.numel() for p in zone.parameters()) for zone in self.zones),
            "hub": sum(p.numel() for p in self.hub.parameters()),
            "output": sum(p.numel() for p in self.output_proj.parameters()),
        }
        counts["total"] = sum(counts.values())
        return counts

    def print_architecture(self) -> None:
        """Imprime un resumen de la arquitectura."""
        params = self.get_num_parameters()
        print(f"\n{'='*50}")
        print(f"FIP2 ZonedBrainModel Architecture")
        print(f"{'='*50}")
        print(f"Device: {self.device}")
        print(f"Zones: {self.config.num_zones}")
        print(f"Neurons per zone: {self.config.neurons_per_zone}")
        print(f"Hub neurons: {self.config.hub_neurons}")
        print(f"Total neurons: {self.config.total_neurons}")
        print(f"Neuron dim: {self.config.neuron_dim}")
        print(f"Vocab size: {self.config.vocab_size}")
        print(f"Context length: {self.config.context_length}")
        print(f"Waves per forward: {self.config.num_waves}")
        print(f"\nParameters:")
        for name, count in params.items():
            print(f"  {name}: {count:,}")
        print(f"{'='*50}\n")
