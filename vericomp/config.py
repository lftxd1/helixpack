from dataclasses import dataclass


@dataclass
class VeriCompConfig:
    """Model and allocator settings.

    The default configuration is approximately 100M parameters with sparse
    Top-2 MoE FFN layers, a GeneZip-style Mamba2 encoder, and a 4096-token
    bottleneck.
    """

    vocab_size: int = 6
    output_vocab_size: int = 4
    shallow_dim: int = 256
    shallow_mamba_layers: int = 2
    shallow_mamba_state: int = 64
    shallow_mamba_conv: int = 4
    shallow_mamba_expand: int = 2
    shallow_mamba_head_dim: int = 64
    shallow_mamba_chunk_size: int = 256
    d_model: int = 512
    num_layers: int = 12
    nhead: int = 8
    dim_feedforward: int = 6656
    moe_intermediate_size: int = 272
    num_experts: int = 16
    num_experts_per_tok: int = 2
    expert_layer_period: int = 2
    expert_layer_offset: int = 1
    router_aux_loss_coef: float = 0.02
    max_tokens: int = 4096
    # Bases averaged into one bottleneck token when allocation_mode is "grid".
    # The other allocation modes ignore this field entirely.
    grid_stride: int = 32
    max_position_embeddings: int = 4096
    boundary_router_weight: float = 1.0
    allocation_mode: str = "dynamic"
    strict_uncompressed_short: bool = False
    # ``dual_independent`` uses the same two-strand evidence as ``dual`` but
    # normalizes each strand to the bottleneck budget before averaging.
    # ``rc_only`` is a single-strand path whose caller supplies the RC view as
    # ``input_ids_pos``; this keeps its output coordinates self-consistent.
    duplex_mode: str = "dual"
    dropout: float = 0.0
    force_flash_attention: bool = True
    causal_attention: bool = False
    activation_checkpointing: bool = True
    # Optionally keep the full-resolution shallow residual stream in BF16
    # under CUDA BF16 autocast to reduce sequence-dependent activation memory.
    bf16_residual_stream: bool = False
    dense_weight_max_elements: int = 256_000_000
    upsample_chunk_size: int = 8192
    pad_token_id: int = 5
    mask_token_id: int = 4
