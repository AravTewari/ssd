from transformers import AutoTokenizer


# Infer model family based on model path name
def infer_model_family(model_path: str) -> str:
        """Infer if model is Llama or Qwen based on path name."""
        model_path_lower = model_path.lower()
        if "llama" in model_path_lower:
            return "llama"
        elif "qwen2.5" in model_path_lower or "qwen2_5" in model_path_lower or "qwen-2.5" in model_path_lower:
            return "qwen2"
        elif "qwen" in model_path_lower:
            return "qwen"
        elif "llada" in model_path_lower:
            return "llada"
        elif "dream" in model_path_lower:
            return "dream"
        else:
            return "unknown"


def decode_tokens(token_ids: list[int], tokenizer: AutoTokenizer) -> list[str]:
    decoded = []
    for token in token_ids:
        try:
            text = tokenizer.decode([token], skip_special_tokens=False)
            decoded.append(text)
        except Exception:
            decoded.append(f"<token_id:{token}>")
    return decoded
