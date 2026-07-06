from __future__ import annotations

import torch


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_device() -> str:
    device = get_device()
    if torch.cuda.is_available():
        return f"Device: {device} - {torch.cuda.get_device_name(0)}"
    return "Device: CPU"


if __name__ == "__main__":
    print(describe_device())
