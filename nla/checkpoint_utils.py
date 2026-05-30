from pathlib import Path

import torch

_DEFAULT_CHUNK_MB = 95
_SPLIT_MARKER = "__split__"


def _storage_bytes(t: torch.Tensor) -> int:
    """Return the bytes torch.save will actually write (full backing storage)."""
    try:
        return t.untyped_storage().nbytes()
    except AttributeError:          # older PyTorch
        return t.storage().nbytes()


def _compact(t: torch.Tensor) -> torch.Tensor:
    """Give the tensor its own storage so nelement*element_size == saved bytes."""
    if _storage_bytes(t) > t.nelement() * t.element_size():
        return t.contiguous().clone()
    return t


def _expand(key: str, tensor: torch.Tensor, chunk_bytes: int) -> list[tuple[str, torch.Tensor]]:
    """Return one or more (key, tensor) pairs that each fit within chunk_bytes."""
    tensor = _compact(tensor)                           # own compact storage first
    data_bytes = tensor.nelement() * tensor.element_size()

    if data_bytes <= chunk_bytes or tensor.dim() == 0:
        return [(key, tensor)]

    rows = tensor.shape[0]
    rows_per_chunk = max(1, int(chunk_bytes / (data_bytes / rows)))
    return [
        (f"{key}{_SPLIT_MARKER}{i}", tensor[start : start + rows_per_chunk].clone())
        for i, start in enumerate(range(0, rows, rows_per_chunk))
    ]


def save_chunked(state_dict: dict, directory: Path, chunk_mb: int = _DEFAULT_CHUNK_MB) -> None:
    """Split a state dict into ≤chunk_mb MB files under directory/part_XXXX.pt."""
    chunk_bytes = chunk_mb * 1024 * 1024
    directory.mkdir(parents=True, exist_ok=True)

    expanded: list[tuple[str, torch.Tensor]] = []
    for key, tensor in state_dict.items():
        expanded.extend(_expand(key, tensor, chunk_bytes))

    parts: list[dict] = []
    current: dict = {}
    current_size = 0
    for key, tensor in expanded:
        size = tensor.nelement() * tensor.element_size()  # compact, so this is accurate
        if current and current_size + size > chunk_bytes:
            parts.append(current)
            current = {}
            current_size = 0
        current[key] = tensor
        current_size += size
    if current:
        parts.append(current)

    for i, part in enumerate(parts):
        torch.save(part, directory / f"part_{i:04d}.pt")

    for stale in sorted(directory.glob("part_*.pt"))[len(parts):]:
        stale.unlink()

    print(f"Saved {len(parts)} chunk(s) (~{chunk_mb} MB each) to {directory}/")


def load_chunked(directory: Path, **torch_load_kwargs) -> dict:
    """Merge all part_XXXX.pt files in directory/ into one state dict."""
    part_files = sorted(directory.glob("part_*.pt"))
    if not part_files:
        raise FileNotFoundError(f"No part_*.pt files found in {directory}")

    flat: dict = {}
    for p in part_files:
        flat.update(torch.load(p, **torch_load_kwargs))

    sliced: dict[str, list[tuple[int, torch.Tensor]]] = {}
    state_dict: dict = {}
    for key, tensor in flat.items():
        if _SPLIT_MARKER in key:
            base, idx = key.rsplit(_SPLIT_MARKER, 1)
            sliced.setdefault(base, []).append((int(idx), tensor))
        else:
            state_dict[key] = tensor

    for base, pieces in sliced.items():
        pieces.sort()
        state_dict[base] = torch.cat([t for _, t in pieces], dim=0)

    return state_dict
