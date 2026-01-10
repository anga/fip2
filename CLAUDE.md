# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FIP2 (Flujo de Información con Presupuesto v2) is a neural network architecture inspired by brain zones with a central hub. It implements a language model using byte-level encoding (256 tokens) with features like structural plasticity (neuron birth/death) and sparse connectivity.

## Commands

```bash
# Train a model
python -m fip2.main train --data wikitext2.txt --config medium --epochs 10

# Generate text from a trained model
python -m fip2.main generate --model checkpoints/best_model.pt --prompt "Hello" --length 100

# Interactive mode
python -m fip2.main interactive --model checkpoints/best_model.pt

# Test forward pass (useful for verifying architecture changes)
python -m fip2.main test-forward

# Show system info (device, PyTorch version, model configs)
python -m fip2.main info
```

Training automatically resumes from the latest checkpoint in `checkpoints/`. Use Ctrl+C once for graceful shutdown with checkpoint save.

## Architecture

### Core Flow
```
INPUT → Z1 → HUB → Z2 → HUB → Z3 → HUB → Z4 → HUB → Z5 → OUTPUT
```

All zones communicate exclusively through the central hub (like a thalamus). Each forward pass runs multiple "waves" of propagation.

### Key Components

- **ZonedBrainModel** (`model.py`): Main model orchestrating zones and hub
- **Zone** (`zones/zone.py`): Brain zone with sparse internal connections and buffer neurons for hub communication
- **Hub** (`zones/hub.py`): Central integrator that receives from all zone buffers and broadcasts processed information back
- **PlasticityManager** (`plasticity/manager.py`): Handles structural changes - edge utility tracking and neuron health-based refresh

### Information Flow Detail

1. **Input**: Byte sequences embedded with position encoding, then multi-token context projected into Z1's input neurons
2. **Wave propagation**: For each wave:
   - Each zone propagates internally (sparse connections via fixed mask)
   - Buffer neurons from all zones collected by hub
   - Hub processes with weighted aggregation + internal propagation
   - Hub broadcasts output back to all zone buffers
3. **Output**: Z5 output used as attention query over original context embeddings, combined and projected to vocabulary

### Configurations

Three presets in `config.py`: `get_small_config()`, `get_medium_config()`, `get_large_config()`. Key parameters:
- `num_zones`: Number of brain zones (default 5)
- `neurons_per_zone`: Neurons in each zone (default 200)
- `hub_neurons`: Central hub size (default 400)
- `num_waves`: Propagation waves per forward (default 3-5)
- `buffer_ratio`: Fraction of zone neurons dedicated to hub communication (default 0.1)

## MPS/Apple Silicon Compatibility

The codebase is designed to be MPS-safe:
- Uses dense operations with masks instead of sparse ops
- `aggregate_by_target_mps_safe()` in `zones/connections.py` replaces scatter_add
- Device utilities in `utils/device.py` handle dtype conversions (no float64/int64 on MPS)

## Plasticity System

When `enable_plasticity=True`:
- **Edge utility** tracked via gradient magnitude, information flow, and stability (MUC metric)
- **Neuron health** based on activation patterns, variability, and connectivity
- Low-utility neurons get "refreshed" (reinitialized) while preserving connectivity
- Configurable intervals and thresholds in `FIP2Config`
