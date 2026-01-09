"""Utilidades generales."""
from .device import get_device, is_mps_available, print_device_status

__all__ = ["get_device", "is_mps_available", "print_device_status"]
