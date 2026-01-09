#!/usr/bin/env python3
"""
FIP2: Flujo de Información con Presupuesto v2

Entry point con CLI para entrenamiento y generación.

Uso:
    python -m fip2.main train --data path/to/data.txt
    python -m fip2.main generate --model path/to/model.pt --prompt "Hello"
    python -m fip2.main info
"""
import argparse
import sys
import os

# Añadir directorio padre al path para imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from fip2.config import FIP2Config, TrainingConfig, get_small_config, get_medium_config
from fip2.model import ZonedBrainModel
from fip2.training import Trainer
from fip2.utils import get_device, print_device_status


def cmd_train(args):
    """Comando de entrenamiento."""
    print("=" * 60)
    print("FIP2 Training")
    print("=" * 60)

    # Configuración
    if args.config == "small":
        config = get_small_config()
    elif args.config == "medium":
        config = get_medium_config()
    else:
        config = FIP2Config()

    # Override con argumentos CLI
    if args.neurons:
        config.neurons_per_zone = args.neurons
    if args.hub_neurons:
        config.hub_neurons = args.hub_neurons
    if args.zones:
        config.num_zones = args.zones
    if args.context:
        config.context_length = args.context
    if args.dim:
        config.neuron_dim = args.dim
    if args.waves:
        config.num_waves = args.waves
    if args.lr:
        config.learning_rate = args.lr
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.device:
        config.device = args.device

    # Training config
    train_config = TrainingConfig(
        data_path=args.data,
        num_epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        checkpoint_dir=args.checkpoint_dir,
    )

    if args.resume:
        train_config.resume_from = args.resume

    # Crear modelo
    print("\nCreating model...")
    model = ZonedBrainModel(config)
    model.print_architecture()

    # Crear trainer
    trainer = Trainer(model, config, train_config)

    # Cargar datos
    print(f"\nLoading data from: {args.data}")
    trainer.load_data(args.data)

    # Resume: primero explícito, luego automático
    if args.resume:
        trainer.load_checkpoint(args.resume)
    else:
        # Auto-resume desde el último checkpoint
        trainer.auto_resume()

    # Entrenar
    trainer.train()

    # Guardar modelo final
    final_path = os.path.join(args.checkpoint_dir, "final_model.pt")
    trainer.save_checkpoint(final_path)
    print(f"\nFinal model saved to: {final_path}")


def cmd_generate(args):
    """Comando de generación."""
    print("=" * 60)
    print("FIP2 Text Generation")
    print("=" * 60)

    # Cargar checkpoint (weights_only=False porque guardamos FIP2Config)
    print(f"\nLoading model from: {args.model}")
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    config = checkpoint["config"]

    # Crear modelo y cargar pesos
    model = ZonedBrainModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Convertir prompt a bytes
    prompt_bytes = args.prompt.encode("utf-8")
    prompt_ids = torch.tensor([list(prompt_bytes)], dtype=torch.long)
    prompt_ids = prompt_ids.to(model.device)

    print(f"\nPrompt: {args.prompt}")
    print(f"Generating {args.length} tokens...")
    print("-" * 40)

    # Generar
    with torch.no_grad():
        generated_ids = model.generate(
            prompt_ids,
            max_new_tokens=args.length,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    # Decodificar
    generated_bytes = bytes(generated_ids[0].tolist())
    try:
        generated_text = generated_bytes.decode("utf-8", errors="replace")
    except:
        generated_text = str(generated_bytes)

    print(generated_text)
    print("-" * 40)


def cmd_interactive(args):
    """Modo interactivo."""
    print("=" * 60)
    print("FIP2 Interactive Mode")
    print("=" * 60)

    # Cargar checkpoint (weights_only=False porque guardamos FIP2Config)
    print(f"\nLoading model from: {args.model}")
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    config = checkpoint["config"]

    model = ZonedBrainModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("\nModel loaded. Type your prompts (Ctrl+C to exit).")
    print("-" * 40)

    while True:
        try:
            prompt = input("\n> ")
            if not prompt.strip():
                continue

            prompt_bytes = prompt.encode("utf-8")
            prompt_ids = torch.tensor([list(prompt_bytes)], dtype=torch.long)
            prompt_ids = prompt_ids.to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    prompt_ids,
                    max_new_tokens=args.length,
                    temperature=args.temperature,
                    top_k=args.top_k,
                )

            generated_bytes = bytes(generated_ids[0].tolist())
            generated_text = generated_bytes.decode("utf-8", errors="replace")
            print(generated_text)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def cmd_info(args):
    """Muestra información del sistema."""
    print("=" * 60)
    print("FIP2 System Information")
    print("=" * 60)

    # Device info
    device = get_device("auto")
    print(f"\nDevice: {device}")
    print_device_status(device)

    # PyTorch info
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if hasattr(torch.backends, "mps"):
        print(f"MPS available: {torch.backends.mps.is_available()}")

    # Model info
    print("\nModel configurations:")
    for name, config_fn in [("small", get_small_config), ("medium", get_medium_config)]:
        cfg = config_fn()
        print(f"\n  {name}:")
        print(f"    Zones: {cfg.num_zones}")
        print(f"    Neurons/zone: {cfg.neurons_per_zone}")
        print(f"    Hub neurons: {cfg.hub_neurons}")
        print(f"    Total neurons: {cfg.total_neurons}")
        print(f"    Neuron dim: {cfg.neuron_dim}")


def cmd_test_forward(args):
    """Prueba un forward pass para verificar que funciona."""
    print("=" * 60)
    print("FIP2 Forward Pass Test")
    print("=" * 60)

    config = get_small_config()
    if args.device:
        config.device = args.device

    print(f"\nCreating model with config 'small'...")
    model = ZonedBrainModel(config)
    model.print_architecture()

    print("\nTesting forward pass...")

    # Crear input dummy
    batch_size = 2
    seq_len = config.context_length
    x = torch.randint(0, 256, (batch_size, seq_len), device=model.device)

    try:
        logits, zone_outputs = model(x, return_zone_outputs=True)
        print(f"  Input shape: {x.shape}")
        print(f"  Output logits shape: {logits.shape}")
        print(f"  Number of zone outputs: {len(zone_outputs)}")
        for i, zo in enumerate(zone_outputs):
            print(f"    Zone {i}: {zo.shape}")

        print("\nForward pass successful!")

        # Test backward
        print("\nTesting backward pass...")
        loss = logits.sum()
        loss.backward()
        print("Backward pass successful!")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="FIP2: Flujo de Información con Presupuesto v2"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--data", required=True, help="Path to training data")
    train_parser.add_argument("--config", choices=["small", "medium", "large"], default="medium")
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--steps-per-epoch", type=int, default=1000)
    train_parser.add_argument("--batch-size", type=int, default=None)
    train_parser.add_argument("--lr", type=float, default=None)
    train_parser.add_argument("--neurons", type=int, default=None)
    train_parser.add_argument("--hub-neurons", type=int, default=None)
    train_parser.add_argument("--zones", type=int, default=None)
    train_parser.add_argument("--context", type=int, default=None)
    train_parser.add_argument("--dim", type=int, default=None)
    train_parser.add_argument("--waves", type=int, default=None)
    train_parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default=None)
    train_parser.add_argument("--checkpoint-dir", default="checkpoints")
    train_parser.add_argument("--resume", default=None, help="Path to checkpoint to resume")
    train_parser.add_argument("--eval-interval", type=int, default=100)
    train_parser.add_argument("--save-interval", type=int, default=1000)
    train_parser.add_argument("--log-interval", type=int, default=10)

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate text")
    gen_parser.add_argument("--model", required=True, help="Path to model checkpoint")
    gen_parser.add_argument("--prompt", required=True, help="Prompt text")
    gen_parser.add_argument("--length", type=int, default=100, help="Max tokens to generate")
    gen_parser.add_argument("--temperature", type=float, default=1.0)
    gen_parser.add_argument("--top-k", type=int, default=50)

    # Interactive command
    int_parser = subparsers.add_parser("interactive", help="Interactive mode")
    int_parser.add_argument("--model", required=True, help="Path to model checkpoint")
    int_parser.add_argument("--length", type=int, default=100)
    int_parser.add_argument("--temperature", type=float, default=1.0)
    int_parser.add_argument("--top-k", type=int, default=50)

    # Info command
    info_parser = subparsers.add_parser("info", help="Show system info")

    # Test forward command
    test_parser = subparsers.add_parser("test-forward", help="Test forward pass")
    test_parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "interactive":
        cmd_interactive(args)
    elif args.command == "info":
        cmd_info(args)
    elif args.command == "test-forward":
        return cmd_test_forward(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    sys.exit(main() or 0)
