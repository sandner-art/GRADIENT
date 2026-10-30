# tests/helpers.py
"""Test utilities: a tiny random-weight Stable Diffusion pipeline (no network needed)."""

import json
from pathlib import Path


def _write_tokenizer(folder: Path) -> None:
    """A minimal CLIP BPE tokenizer: printable ASCII characters as tokens, no merges."""
    folder.mkdir(parents=True, exist_ok=True)
    chars = [chr(c) for c in range(33, 127)]
    vocab: dict[str, int] = {}
    for ch in chars:
        vocab.setdefault(ch, len(vocab))
        vocab.setdefault(ch + "</w>", len(vocab))
    vocab["<|startoftext|>"] = len(vocab)
    vocab["<|endoftext|>"] = len(vocab)
    (folder / "vocab.json").write_text(json.dumps(vocab))
    (folder / "merges.txt").write_text("#version: 0.2\n")
    (folder / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "CLIPTokenizer",
                "model_max_length": 77,
                "bos_token": "<|startoftext|>",
                "eos_token": "<|endoftext|>",
                "unk_token": "<|endoftext|>",
                "pad_token": "<|endoftext|>",
            }
        )
    )
    (folder / "special_tokens_map.json").write_text(
        json.dumps(
            {
                "bos_token": "<|startoftext|>",
                "eos_token": "<|endoftext|>",
                "unk_token": "<|endoftext|>",
                "pad_token": "<|endoftext|>",
            }
        )
    )


def build_tiny_sd(path: Path, seed: int = 0) -> Path:
    """Create a ~1 MB random StableDiffusionPipeline in diffusers format at `path`."""
    import torch
    from diffusers import (
        AutoencoderKL,
        EulerDiscreteScheduler,
        StableDiffusionPipeline,
        UNet2DConditionModel,
    )
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

    torch.manual_seed(seed)
    unet = UNet2DConditionModel(
        block_out_channels=(8, 16),
        layers_per_block=1,
        sample_size=8,
        in_channels=4,
        out_channels=4,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=16,
        attention_head_dim=2,
        norm_num_groups=4,
    )
    vae = AutoencoderKL(
        block_out_channels=(8, 16),
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        latent_channels=4,
        norm_num_groups=4,
        sample_size=16,
    )
    text_encoder = CLIPTextModel(
        CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=16,
            intermediate_size=16,
            layer_norm_eps=1e-5,
            num_attention_heads=2,
            num_hidden_layers=1,
            pad_token_id=1,
            vocab_size=200,
            max_position_embeddings=77,
        )
    )
    tok_dir = path.parent / f".{path.name}-tokenizer-src"
    _write_tokenizer(tok_dir)
    tokenizer = CLIPTokenizer.from_pretrained(str(tok_dir))
    pipe = StableDiffusionPipeline(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=EulerDiscreteScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", steps_offset=1
        ),
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    pipe.save_pretrained(str(path))
    return path
