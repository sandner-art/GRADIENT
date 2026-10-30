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


# --------------------------------------------------------------------------- tiny DiT pipelines
WORDS = (
    "a an the of on in at with and red blue green dog cat lighthouse rocky coast sunset dramatic clouds "
    "sky sea rocks background tree grass castle hill photo painting user assistant"
).split()


def _word_tokenizer(specials: list[str], eos: str | None):
    """A word-level `tokenizers` tokenizer: known words, specials, <unk> for the rest."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    vocab = {tok: i for i, tok in enumerate(specials)}
    for w in WORDS:
        vocab.setdefault(w, len(vocab))
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    if eos:
        tok.post_processor = processors.TemplateProcessing(
            single=f"$A {eos}", special_tokens=[(eos, vocab[eos])]
        )
    return tok, len(vocab)


def _clip_pair(folder: Path, hidden: int, with_projection: bool = False, projection_dim: int = 16):
    from transformers import (
        CLIPTextConfig,
        CLIPTextModel,
        CLIPTextModelWithProjection,
        CLIPTokenizer,
    )

    _write_tokenizer(folder)
    config = CLIPTextConfig(
        bos_token_id=188,
        eos_token_id=189,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_attention_heads=2,
        num_hidden_layers=2,
        pad_token_id=189,
        vocab_size=200,
        max_position_embeddings=77,
        projection_dim=projection_dim,
    )
    model = CLIPTextModelWithProjection(config) if with_projection else CLIPTextModel(config)
    return model, CLIPTokenizer.from_pretrained(str(folder))


def _tiny_t5(d_model: int = 32):
    from transformers import PreTrainedTokenizerFast, T5Config, T5EncoderModel

    tok, vocab_size = _word_tokenizer(["<pad>", "</s>", "<unk>"], eos="</s>")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="<pad>", eos_token="</s>", unk_token="<unk>"
    )
    model = T5EncoderModel(
        T5Config(
            vocab_size=vocab_size,
            d_model=d_model,
            d_kv=8,
            d_ff=64,
            num_layers=2,
            num_heads=4,
            relative_attention_num_buckets=8,
            pad_token_id=0,
            eos_token_id=1,
            decoder_start_token_id=0,
        )
    )
    return model, tokenizer


def build_tiny_flux(path: Path, seed: int = 0) -> Path:
    """Tiny random FluxPipeline (FLUX.1-dev layout: 4 double + 1 single block, guidance embeds)."""
    import torch
    from diffusers import (
        AutoencoderKL,
        FlowMatchEulerDiscreteScheduler,
        FluxPipeline,
        FluxTransformer2DModel,
    )

    torch.manual_seed(seed)
    src = path.parent / f".{path.name}-src"
    text_encoder, tokenizer = _clip_pair(src / "clip", hidden=16)
    text_encoder_2, tokenizer_2 = _tiny_t5()
    transformer = FluxTransformer2DModel(
        patch_size=1,
        in_channels=4,
        num_layers=4,
        num_single_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=32,
        pooled_projection_dim=16,
        guidance_embeds=True,
        axes_dims_rope=(2, 2, 4),
    )
    vae = AutoencoderKL(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(8, 16),
        layers_per_block=1,
        latent_channels=1,
        norm_num_groups=4,
        use_quant_conv=False,
        use_post_quant_conv=False,
        shift_factor=0.0609,
        scaling_factor=1.5035,
    )
    scheduler = FlowMatchEulerDiscreteScheduler(
        use_dynamic_shifting=True,
        base_shift=0.5,
        max_shift=1.15,
        base_image_seq_len=256,
        max_image_seq_len=4096,
        shift=3.0,
    )
    FluxPipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        text_encoder_2=text_encoder_2,
        tokenizer_2=tokenizer_2,
        transformer=transformer,
    ).save_pretrained(str(path))
    return path


def build_tiny_sd3(path: Path, seed: int = 0) -> Path:
    """Tiny random StableDiffusion3Pipeline (SD3.5 layout: QK-norm, one dual-attention block)."""
    import torch
    from diffusers import (
        AutoencoderKL,
        FlowMatchEulerDiscreteScheduler,
        SD3Transformer2DModel,
        StableDiffusion3Pipeline,
    )

    torch.manual_seed(seed)
    src = path.parent / f".{path.name}-src"
    text_encoder, tokenizer = _clip_pair(src / "clip1", hidden=16, with_projection=True)
    text_encoder_2, tokenizer_2 = _clip_pair(src / "clip2", hidden=16, with_projection=True)
    text_encoder_3, tokenizer_3 = _tiny_t5()
    transformer = SD3Transformer2DModel(
        sample_size=32,
        patch_size=2,
        in_channels=4,
        num_layers=3,
        attention_head_dim=8,
        num_attention_heads=4,
        joint_attention_dim=32,
        caption_projection_dim=32,
        pooled_projection_dim=32,
        out_channels=4,
        qk_norm="rms_norm",
        dual_attention_layers=(0,),
    )
    vae = AutoencoderKL(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(8, 16),
        layers_per_block=1,
        latent_channels=4,
        norm_num_groups=4,
        use_quant_conv=False,
        use_post_quant_conv=False,
        shift_factor=0.0609,
        scaling_factor=1.5035,
    )
    StableDiffusion3Pipeline(
        transformer=transformer,
        scheduler=FlowMatchEulerDiscreteScheduler(shift=3.0),
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        text_encoder_2=text_encoder_2,
        tokenizer_2=tokenizer_2,
        text_encoder_3=text_encoder_3,
        tokenizer_3=tokenizer_3,
    ).save_pretrained(str(path))
    return path


QWEN_CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def build_tiny_flux2_klein(path: Path, seed: int = 0) -> Path:
    """Tiny random Flux2KleinPipeline (Qwen3 text encoder with a chat template)."""
    import torch
    from diffusers import (
        AutoencoderKLFlux2,
        FlowMatchEulerDiscreteScheduler,
        Flux2KleinPipeline,
        Flux2Transformer2DModel,
    )
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<unk>"]
    qwen_tok, vocab_size = _word_tokenizer(specials, eos=None)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=qwen_tok,
        pad_token="<|endoftext|>",
        eos_token="<|im_end|>",
        unk_token="<unk>",
        additional_special_tokens=["<|im_start|>", "<|im_end|>"],
    )
    tokenizer.chat_template = QWEN_CHAT_TEMPLATE
    text_encoder = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=vocab_size,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=28,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=1024,
            tie_word_embeddings=True,
        )
    )
    transformer = Flux2Transformer2DModel(
        patch_size=1,
        in_channels=16,
        num_layers=3,
        num_single_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=48,
        timestep_guidance_channels=32,
        axes_dims_rope=(2, 2, 2, 2),
        guidance_embeds=False,
    )
    vae = AutoencoderKLFlux2(
        sample_size=32,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(8, 16),
        layers_per_block=1,
        latent_channels=4,
        norm_num_groups=4,
    )
    Flux2KleinPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler(
            use_dynamic_shifting=True,
            base_shift=0.5,
            max_shift=1.15,
            base_image_seq_len=256,
            max_image_seq_len=4096,
            shift=3.0,
        ),
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        is_distilled=True,
    ).save_pretrained(str(path))
    return path
