import torch

_INJECTION_RANGE = range(0x3200, 0x3400)


def find_injection_token(tokenizer) -> tuple[str, int]:
    for cp in _INJECTION_RANGE:
        char = chr(cp)
        ids  = tokenizer.encode(char, add_special_tokens=False)
        if (
            len(ids) == 1
            and ids[0] != getattr(tokenizer, "unk_token_id", None)
            and ids[0] != getattr(tokenizer, "bos_token_id", None)
        ):
            return char, ids[0]
    raise ValueError(
        f"No single-token char found in U+3200..U+33FF for {type(tokenizer).__name__}."
    )


def find_injection_neighbors(
    input_ids: torch.Tensor,
    injection_token_id: int,
) -> tuple[int, int]:
    matches = (input_ids[0] == injection_token_id).nonzero(as_tuple=False)
    assert len(matches) == 1, (
        f"Expected exactly 1 injection token in prompt, found {len(matches)}."
    )
    p = int(matches[0, 0])
    return int(input_ids[0, p - 1]), int(input_ids[0, p + 1])


def inject_at_marked_positions(
    input_ids:  torch.Tensor,
    embeddings: torch.Tensor,
    v_scaled:   torch.Tensor,
    inj_id:     int,
    left_id:    int,
    right_id:   int,
) -> torch.Tensor:
    seq_len  = input_ids.shape[-1]
    out      = embeddings.clone()
    v_scaled = v_scaled.to(out.device, out.dtype)
    found    = 0
    for b, p in (input_ids == inj_id).nonzero().tolist():
        if p == 0 or p == seq_len - 1:
            continue
        if input_ids[b, p - 1] != left_id or input_ids[b, p + 1] != right_id:
            continue
        out[b, p] = v_scaled[0]
        found    += 1
    assert found == 1, (
        f"Expected 1 injection site with correct neighbors, found {found}."
    )
    return out
