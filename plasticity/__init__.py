"""Plasticidad estructural: edges y neuronas."""
from .edge_utility import EdgeUtilityTracker
from .neuron_plasticity import NeuronPlasticityEngine
from .manager import PlasticityManager, PlasticityStats

__all__ = [
    "EdgeUtilityTracker",
    "NeuronPlasticityEngine",
    "PlasticityManager",
    "PlasticityStats",
]
