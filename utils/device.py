"""
Device handling optimizado para M1/MPS.

Este módulo proporciona funciones para detección y manejo de dispositivos,
con soporte especial para Apple Silicon (MPS).
"""
import torch


def is_mps_available() -> bool:
    """Verifica si MPS (Metal Performance Shaders) está disponible."""
    return (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )


def is_cuda_available() -> bool:
    """Verifica si CUDA está disponible."""
    return torch.cuda.is_available()


def get_device(preference: str = "auto") -> torch.device:
    """
    Obtiene el dispositivo óptimo para el entrenamiento.

    Args:
        preference: "auto", "mps", "cuda", o "cpu"

    Returns:
        torch.device configurado
    """
    if preference == "auto":
        if is_mps_available():
            return torch.device("mps")
        elif is_cuda_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    elif preference == "mps":
        if not is_mps_available():
            print("WARNING: MPS solicitado pero no disponible. Usando CPU.")
            return torch.device("cpu")
        return torch.device("mps")
    elif preference == "cuda":
        if not is_cuda_available():
            print("WARNING: CUDA solicitado pero no disponible. Usando CPU.")
            return torch.device("cpu")
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def get_device_info(device: torch.device) -> dict:
    """Obtiene información detallada del dispositivo."""
    info = {
        "device": str(device),
        "type": device.type,
    }

    if device.type == "cuda":
        info["name"] = torch.cuda.get_device_name(device)
        info["memory_total"] = torch.cuda.get_device_properties(device).total_memory
        info["memory_allocated"] = torch.cuda.memory_allocated(device)
    elif device.type == "mps":
        info["name"] = "Apple Silicon (MPS)"
        # MPS no tiene API para consultar memoria directamente
        info["memory_total"] = "N/A"
        info["memory_allocated"] = "N/A"
    else:
        info["name"] = "CPU"
        info["memory_total"] = "System RAM"

    return info


def to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Mueve un tensor al dispositivo de forma segura.

    Maneja casos especiales como MPS que no soporta ciertos dtypes.
    """
    if device.type == "mps":
        # MPS no soporta float64, convertir a float32
        if tensor.dtype == torch.float64:
            tensor = tensor.float()
        # MPS no soporta int64 en algunas operaciones
        if tensor.dtype == torch.int64:
            tensor = tensor.int()

    return tensor.to(device)


def empty_cache(device: torch.device) -> None:
    """Libera memoria del dispositivo si es posible."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        # MPS tiene su propio garbage collector, pero podemos forzar
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


def synchronize(device: torch.device) -> None:
    """Sincroniza operaciones del dispositivo."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        if hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()


def print_device_status(device: torch.device) -> None:
    """Imprime el estado del dispositivo."""
    info = get_device_info(device)
    print(f"Device: {info['name']} ({info['type']})")
    if info.get("memory_total") and info["memory_total"] != "N/A":
        print(f"  Memory Total: {info['memory_total'] / 1e9:.1f} GB")
    if info.get("memory_allocated") and info["memory_allocated"] != "N/A":
        print(f"  Memory Used: {info['memory_allocated'] / 1e9:.1f} GB")
