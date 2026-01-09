"""
Funciones de pérdida para FIP2.

Incluye:
- CrossEntropy para predicción de siguiente token
- Pérdida de diversidad para especialización de zonas
"""
import torch
import torch.nn.functional as F
from typing import List, Optional


def compute_cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Calcula la pérdida de cross-entropy para predicción de tokens.

    Args:
        logits: (batch, vocab_size) logits del modelo
        targets: (batch,) índices de tokens objetivo
        ignore_index: Índice a ignorar en el cálculo

    Returns:
        Pérdida escalar
    """
    return F.cross_entropy(logits, targets, ignore_index=ignore_index)


def diversity_loss(zone_outputs: List[torch.Tensor]) -> torch.Tensor:
    """
    Pérdida de diversidad para forzar especialización de zonas.

    Penaliza que las zonas tengan representaciones similares usando
    similitud coseno.

    Args:
        zone_outputs: Lista de tensores (batch, dim) - salida de cada zona

    Returns:
        Pérdida de diversidad (menor es mejor)
    """
    num_zones = len(zone_outputs)

    if num_zones < 2:
        return torch.tensor(0.0, device=zone_outputs[0].device)

    total_similarity = 0.0
    count = 0

    for i in range(num_zones):
        for j in range(i + 1, num_zones):
            # Similitud coseno entre zonas i y j
            # F.cosine_similarity retorna (batch,)
            similarity = F.cosine_similarity(
                zone_outputs[i], zone_outputs[j], dim=-1
            )
            # Queremos que sea baja, así que penalizamos valores altos
            total_similarity += similarity.abs().mean()
            count += 1

    return total_similarity / count if count > 0 else torch.tensor(0.0)


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    zone_outputs: Optional[List[torch.Tensor]] = None,
    diversity_weight: float = 0.1,
) -> torch.Tensor:
    """
    Calcula la pérdida total combinando CE y diversidad.

    Args:
        logits: (batch, vocab_size) logits del modelo
        targets: (batch,) tokens objetivo
        zone_outputs: Lista de salidas de zonas para diversidad
        diversity_weight: Peso de la pérdida de diversidad

    Returns:
        Pérdida total
    """
    # Pérdida principal: CrossEntropy
    ce_loss = compute_cross_entropy_loss(logits, targets)

    # Pérdida de diversidad
    div_loss = torch.tensor(0.0, device=logits.device)
    if zone_outputs is not None and diversity_weight > 0:
        div_loss = diversity_loss(zone_outputs)

    # Combinar
    total_loss = ce_loss + diversity_weight * div_loss

    return total_loss


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calcula la precisión de predicción.

    Args:
        logits: (batch, vocab_size) logits del modelo
        targets: (batch,) tokens objetivo

    Returns:
        Precisión como float [0, 1]
    """
    predictions = logits.argmax(dim=-1)
    correct = (predictions == targets).float()
    return correct.mean().item()


def compute_perplexity(loss: torch.Tensor) -> float:
    """
    Calcula la perplejidad desde la pérdida.

    Args:
        loss: Pérdida de cross-entropy

    Returns:
        Perplejidad
    """
    return torch.exp(loss).item()
