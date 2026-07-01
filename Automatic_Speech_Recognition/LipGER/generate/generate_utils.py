def generate(
    model,
    idx: torch.Tensor,
    emb_diff: Optional[torch.Tensor] = None,
    visual_features: Optional[torch.Tensor] = None,
    max_returned_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 0,
    eos_id: int = 2,
):
    device = idx.device
    dtype = idx.dtype
    generated = idx.clone()

    if idx.numel() == 0 or idx.shape[-1] == 0:
        idx = torch.tensor([[eos_id]], device=device, dtype=dtype)
        generated = idx.clone()

    # Initialize KV cache safely
    kv_cache = None

    for step in range(max_returned_tokens):
        # Forward pass with AMP
        with torch.autocast(device_type="cuda", dtype=dtype):
            logits, kv_cache = model(
                idx,
                emb_diff=emb_diff,
                visual_features=visual_features,
                max_seq_length=min(model.config.block_size, generated.shape[1]+1),  # <-- Clip block_size
                input_pos=torch.tensor([generated.shape[1]-1], device=device)
                if generated.shape[1] > 0 else None,
                kv_cache=kv_cache
            )

        # Last token logits
        next_token_logits = logits[:, -1, :] / max(temperature, 1e-8)

        # Top-k filtering
        if top_k > 0:
            top_values, _ = torch.topk(next_token_logits, top_k, dim=-1)
            min_top_value = top_values[:, -1].unsqueeze(-1)
            next_token_logits[next_token_logits < min_top_value] = -float('Inf')

        # Sample next token
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Append token
        generated = torch.cat([generated, next_token], dim=-1)
        idx = next_token

        if (next_token == eos_id).all():
            break

    return generated[0]
