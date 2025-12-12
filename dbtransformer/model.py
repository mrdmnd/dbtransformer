# An implementation of the relational transformer model.
# https://arxiv.org/abs/2510.06377
# Model sizing from the paper:
# L = 12
# d_text = 384
# d_model = 256
# d_ff = 1024
# num_heads = 8
# batch_size = 256 (we train on 32 to fit on non-datacenter hardware during compilation)
# seq_len = 1024


import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import TypedDict

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from einops._torch_specific import allow_ops_in_compiled_graph  # noqa: PLC2701
from jaxtyping import Bool, Float, Int, jaxtyped
from torch import Tensor, nn
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)

from dbtransformer.configurations import ModelConfig
from dbtransformer.sampler_types import SemanticType

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This model effectively requires CUDA.")

allow_ops_in_compiled_graph()
flex_attention = torch.compile(flex_attention)


# We have four kinds of masked attention blocks:
class AttentionType(Enum):
    COLUMN = 0
    FEATURE = 1
    NEIGHBOR = 2
    FULL = 3


# Maximum number of foreign-to-primary neighbors per cell.
# Only used for DummyBatchDataset in train.py.
MAX_F2P_NEIGHBORS = 5


@jaxtyped(typechecker=None)
class FlexAttentionBlock(nn.Module):
    """Attention using FlexAttention with sparse BlockMasks."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        attention_type: AttentionType,
    ) -> None:
        super().__init__()
        self.attention_type = attention_type
        self.num_heads = num_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: Float[Tensor, "b s d"],
        mask: BlockMask,
    ) -> Float[Tensor, "b s d"]:
        q: Float[Tensor, "b s d"] = self.wq(x)
        k: Float[Tensor, "b s d"] = self.wk(x)
        v: Float[Tensor, "b s d"] = self.wv(x)

        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

        # mypy doesn't know that flex_attention returns a Tensor, but it always
        # does unless return_lse=True, which we don't use.
        attn_out: Float[Tensor, "b h s d"] = flex_attention(q, k, v, block_mask=mask)  # type: ignore
        out: Float[Tensor, "b s d"] = rearrange(attn_out, "b h s d -> b s (h d)")
        return self.wo(out)


@jaxtyped(typechecker=None)
class DenseAttentionBlock(nn.Module):
    """Attention using F.scaled_dot_product_attention (FlashAttention backend).

    Uses dense boolean masks instead of FlexAttention's BlockMasks.
    Still respects the relational attention structure, but uses SDPA kernels
    which may be faster than FlexAttention for some mask patterns.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        attention_type: AttentionType,
    ) -> None:
        super().__init__()
        self.attention_type = attention_type
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: Float[Tensor, "b s d"],
        mask: Bool[Tensor, "b s s"],
    ) -> Float[Tensor, "b s d"]:
        q: Float[Tensor, "b s d"] = self.wq(x)
        k: Float[Tensor, "b s d"] = self.wk(x)
        v: Float[Tensor, "b s d"] = self.wv(x)

        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

        # SDPA expects mask where True = attend, False = mask out.
        # We pass the mask as attn_mask; SDPA handles the rest.
        # enable_flash=True requires the mask to be None or a specific format,
        # so we use the "efficient" backend which handles bool masks.
        attn_out: Float[Tensor, "b h s d"] = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask[:, None, :, :],  # broadcast over heads
            dropout_p=0.0,
            is_causal=False,
        )
        out: Float[Tensor, "b s d"] = rearrange(attn_out, "b h s d -> b s (h d)")
        return self.wo(out)


# Bog-standard FFN with no biases. Uses SwiGLU activation.
@jaxtyped(typechecker=None)
class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Float[Tensor, "b s d"]) -> Float[Tensor, "b s d"]:
        return self.down(F.silu(self.gate(x)) * self.up(x))


# Implements the "Relational Transformer Block" from the paper.
# Supports both FlexAttention (sparse BlockMasks) and FlashAttention SDPA (dense masks).
@jaxtyped(typechecker=None)
class RelationalBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        use_flash: bool = False,
    ) -> None:
        super().__init__()
        self.use_flash = use_flash

        self.column_norm = nn.RMSNorm(d_model)
        self.feature_norm = nn.RMSNorm(d_model)
        self.neighbor_norm = nn.RMSNorm(d_model)
        self.full_norm = nn.RMSNorm(d_model)
        self.ffn_norm = nn.RMSNorm(d_model)

        attn_cls = DenseAttentionBlock if use_flash else FlexAttentionBlock
        self.column_attn = attn_cls(d_model, num_heads, AttentionType.COLUMN)
        self.feature_attn = attn_cls(d_model, num_heads, AttentionType.FEATURE)
        self.neighbor_attn = attn_cls(d_model, num_heads, AttentionType.NEIGHBOR)
        self.full_attn = attn_cls(d_model, num_heads, AttentionType.FULL)

        self.ffn = FFN(d_model, d_ff)

    def forward(
        self,
        x: Float[Tensor, "b s d"],
        col_attn_mask: BlockMask | Bool[Tensor, "b s s"],
        feature_attn_mask: BlockMask | Bool[Tensor, "b s s"],
        neighbor_attn_mask: BlockMask | Bool[Tensor, "b s s"],
        full_attn_mask: BlockMask | Bool[Tensor, "b s s"],
    ) -> Float[Tensor, "b s d"]:
        # Don't use += operations to avoid autograd issues
        x = x + self.column_attn(self.column_norm(x), col_attn_mask)  # noqa: PLR6104
        x = x + self.feature_attn(self.feature_norm(x), feature_attn_mask)  # noqa: PLR6104
        x = x + self.neighbor_attn(self.neighbor_norm(x), neighbor_attn_mask)  # noqa: PLR6104
        x = x + self.full_attn(self.full_norm(x), full_attn_mask)  # noqa: PLR6104
        x = x + self.ffn(self.ffn_norm(x))  # noqa: PLR6104
        return x  # noqa: RET504


# This is torch.flex_attention's BlockMask.
# https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html
# https://github.com/pytorch/pytorch/blob/v2.9.1/torch/nn/attention/flex_attention.py
# The mask_mod signature is Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]
# but the input tensors are 1d scalars, basically.
@jaxtyped(typechecker=None)
def _generate_block_mask(
    mask: Bool[Tensor, "b s s"],
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> BlockMask:
    return create_block_mask(
        mask_mod=lambda b, _, q, kv: mask[b, q, kv],
        B=batch_size,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device,
        _compile=True,
    )


# A "batch" of data for the training loops.
# The sequence is a flattened list of cells from multiple rows (nodes) sampled
# via BFS traversal of the relational graph starting from a seed row in a task
# table.
# We will do some "pre-work" in the data loader to compute some specific sparse
# attention masks so that we don't have to do it in the forward pass.
@dataclass
class Batch:
    # Numeric cell values, z-score normalized per column: (val - mean) / std.
    # NaN values are skipped during preprocessing. Val/test splits use
    # statistics computed from the training set for consistency.
    # When the cell at the position is not a number, the value is 0.0
    number_values: Float[Tensor, "b s 1"]

    # Datetime cell values (converted to seconds since epoch), z-score normalized
    # using *global* statistics computed across ALL datetime columns in the entire
    # database, not per-column. This allows cross-table temporal reasoning.
    # When the cell at the position is not a datetime, the value is 0.0
    datetime_values: Float[Tensor, "b s 1"]

    # Boolean cell values, converted to 0.0/1.0 then z-score normalized
    # per column (same as number_values). Not raw 0/1!
    # When the cell at the position is not a boolean, the value is 0.0
    boolean_values: Float[Tensor, "b s 1"]

    # Pre-computed text embeddings from some SentenceTransformer (MiniLM).
    # During preprocessing, all unique strings are embedded and stored;
    # at runtime these could looked up by index from a memory-mapped file.
    # When the cell at the position is not a text, the value vector is all zeros.
    text_values: Float[Tensor, "b s d_text"]

    # Pre-computed embeddings for column names, formatted as
    # "<column_name> of <table_name>" (e.g., "price of products").
    # TODO(mrdmnd): expand to include column descriptions!
    # Added to every cell's representation as positional context.
    # This is always present, because every cell comes from a column.
    column_name_values: Float[Tensor, "b s d_text"]

    # Semantic type determining which type is present at each position.
    # 0=number, 1=text, 2=datetime, 3=boolean (see SemanticType enum).
    semantic_types: Int[Tensor, "b s"]

    # Positions to HIDE from the model (replaced with learned mask embedding)
    # In current impl, set identically to is_targets (only mask the target).
    masks: Bool[Tensor, "b s"]

    # Whether this cell belongs to a task table row (train/val/test split)
    # vs a regular database table row. Task tables contain the prediction
    # targets; DB tables provide relational context.
    is_task_node: Bool[Tensor, "b s"]

    # Whether this position is padding (sequence shorter than seq_len).
    # Padding positions are excluded from all attention masks and losses.
    is_padding: Bool[Tensor, "b s"]

    # Precomputed attention masks (computed by DataLoader workers on CPU).
    # These avoid expensive 4D tensor expansions in the forward pass.
    # Note: full_attn_mask is derived from is_padding in forward pass (cheap).
    column_attn_mask: Bool[Tensor, "b s s"]
    feature_attn_mask: Bool[Tensor, "b s s"]
    neighbor_attn_mask: Bool[Tensor, "b s s"]

    def pin_memory(self) -> "Batch":
        """Pin all tensors to enable fast async CPU->GPU transfer.

        Called automatically by DataLoader when pin_memory=True.
        Returns a new Batch with pinned tensors (required by DataLoader API).
        """
        pinned_fields = {}
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            pinned_fields[field.name] = value.pin_memory()
        return Batch(**pinned_fields)

    def to_device(
        self,
        device: torch.device,
        float_dtype: torch.dtype | None = None,
    ) -> None:
        """Move all tensors to device in-place.

        Use with pin_memory=True on DataLoader for async transfers via DMA.
        Float tensors are optionally converted to float_dtype (e.g. bfloat16).
        """
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor):
                if float_dtype is not None and value.is_floating_point():
                    setattr(
                        self,
                        field.name,
                        value.to(device, dtype=float_dtype, non_blocking=True),
                    )
                else:
                    setattr(
                        self,
                        field.name,
                        value.to(device, non_blocking=True),
                    )

    @staticmethod
    def compute_attention_masks(
        node_indices: Int[Tensor, "b s"],
        table_name_indices: Int[Tensor, "b s"],
        column_name_indices: Int[Tensor, "b s"],
        f2p_neighbor_indices: Int[Tensor, "b s max_f2p"],
        is_padding: Bool[Tensor, "b s"],
    ) -> tuple[
        Bool[Tensor, "b s s"],
        Bool[Tensor, "b s s"],
        Bool[Tensor, "b s s"],
    ]:
        """Compute the 3 relational attention masks from index tensors.

        Returns (column_mask, feature_mask, neighbor_mask).
        full_mask is trivially derived from is_padding in forward pass.

        Call this in the DataLoader's collate_fn or dataset __getitem__
        to offload mask computation to CPU workers.
        """
        # Active mask: both positions must be non-padding
        active: Bool[Tensor, "b s s"] = ~is_padding[:, :, None] & ~is_padding[:, None, :]

        # Same node (row) - for feature attention
        same_node: Bool[Tensor, "b s s"] = node_indices[:, :, None] == node_indices[:, None, :]

        # KV is in Q's foreign-to-primary (parent) neighborhood
        # (b, s, s, max_f2p) -> (b, s, s)
        kv_in_f2p: Bool[Tensor, "b s s"] = (node_indices[:, None, :, None] == f2p_neighbor_indices[:, :, None, :]).any(dim=-1)

        # Q is in KV's primary-to-foreign (child) neighborhood
        # (b, s, s, max_f2p) -> (b, s, s)
        q_in_p2f: Bool[Tensor, "b s s"] = (node_indices[:, :, None, None] == f2p_neighbor_indices[:, None, :, :]).any(dim=-1)

        # Same column AND same table
        same_column = column_name_indices[:, :, None] == column_name_indices[:, None, :]
        same_table = table_name_indices[:, :, None] == table_name_indices[:, None, :]
        same_col_table: Bool[Tensor, "b s s"] = same_column & same_table

        # Final masks (full_mask = active, computed in forward pass)
        column_mask = same_col_table & active
        feature_mask = (same_node | kv_in_f2p) & active
        neighbor_mask = q_in_p2f & active
        # full_mask = active

        return column_mask, feature_mask, neighbor_mask


@jaxtyped(typechecker=None)
class ModelOutput(TypedDict):
    # The loss averaged over the full batch
    loss: Float[Tensor, ""]
    yhat_number: Float[Tensor, "b s 1"] | None
    yhat_datetime: Float[Tensor, "b s 1"] | None
    yhat_boolean: Float[Tensor, "b s 1"] | None
    yhat_text: Float[Tensor, "b s d_text"] | None


@jaxtyped(typechecker=None)
class RelationalTransformer(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.use_flash = config.attention_mode == "flash"

        # Set up initial embedding layers
        self.column_name_encoder = nn.Linear(config.d_text, config.d_model, bias=True)
        self.number_encoder = nn.Linear(1, config.d_model, bias=True)
        self.text_encoder = nn.Linear(config.d_text, config.d_model, bias=True)
        self.datetime_encoder = nn.Linear(1, config.d_model, bias=True)
        self.boolean_encoder = nn.Linear(1, config.d_model, bias=True)

        # Norms
        self.column_name_norm = nn.RMSNorm(config.d_model)
        self.number_norm = nn.RMSNorm(config.d_model)
        self.text_norm = nn.RMSNorm(config.d_model)
        self.datetime_norm = nn.RMSNorm(config.d_model)
        self.boolean_norm = nn.RMSNorm(config.d_model)

        # Mask Embeddings
        # (number, text, datetime, boolean)
        self.mask_embeddings = nn.Parameter(torch.randn(4, config.d_model))

        # Transformer Blocks - use flash (SDPA) or flex attention based on config
        self.blocks = nn.ModuleList(
            [
                RelationalBlock(
                    config.d_model,
                    config.num_heads,
                    config.d_ff,
                    use_flash=self.use_flash,
                )
                for _ in range(config.num_blocks)
            ]
        )

        # Output Norm
        self.out_norm = nn.RMSNorm(config.d_model)

        # Set up decoder layers
        self.number_decoder = nn.Linear(config.d_model, 1, bias=True)
        self.datetime_decoder = nn.Linear(config.d_model, 1, bias=True)
        self.boolean_decoder = nn.Linear(config.d_model, 1, bias=True)
        self.text_decoder = nn.Linear(config.d_model, config.d_text, bias=True)

    def forward(self, batch: Batch) -> ModelOutput:
        number_values: Float[Tensor, "b s 1"] = batch.number_values
        text_values: Float[Tensor, "b s d_text"] = batch.text_values
        datetime_values: Float[Tensor, "b s 1"] = batch.datetime_values
        boolean_values: Float[Tensor, "b s 1"] = batch.boolean_values
        column_name_values: Float[Tensor, "b s d_text"] = batch.column_name_values
        masks: Bool[Tensor, "b s"] = batch.masks
        is_padding: Bool[Tensor, "b s"] = batch.is_padding

        # Semantics types: [0, 1, 2, 3] -> [number, text, datetime, boolean]
        semantic_type: Int[Tensor, "b s"] = batch.semantic_types
        is_number: Bool[Tensor, "b s"] = semantic_type == SemanticType.NUMBER.value
        is_text: Bool[Tensor, "b s"] = semantic_type == SemanticType.TEXT.value
        is_datetime: Bool[Tensor, "b s"] = semantic_type == SemanticType.DATETIME.value
        is_boolean: Bool[Tensor, "b s"] = semantic_type == SemanticType.BOOLEAN.value

        # Don't do python control flow in the forward pass unless we're debugging.
        # if (masks & is_text).any():
        #     raise ValueError("Masked text positions not supported yet.")

        b: int = is_padding.shape[0]  # Batch Size
        s: int = is_padding.shape[1]  # Sequence Length
        device = is_padding.device

        # =======================================================
        #  INPUT EMBEDDING STEP
        # =======================================================
        with torch.autograd.profiler.record_function("input_embedding"):
            encoded: Float[Tensor, "b s d"] = (
                self.number_norm(self.number_encoder(number_values)) * is_number[..., None]
                + self.text_norm(self.text_encoder(text_values)) * is_text[..., None]
                + self.datetime_norm(self.datetime_encoder(datetime_values)) * is_datetime[..., None]
                + self.boolean_norm(self.boolean_encoder(boolean_values)) * is_boolean[..., None]
            )

            mask_embedded: Float[Tensor, "b s d"] = self.mask_embeddings[semantic_type]
            visible = (~masks & ~is_padding)[..., None]
            hidden = (masks & ~is_padding)[..., None]

            # Input to the model starts as the column name embedding, plus the encoded
            # values, plus the embeddings for whatever is masked.
            x: Float[Tensor, "b s d"] = self.column_name_norm(self.column_name_encoder(column_name_values)) * (~is_padding)[..., None]
            x = x + encoded * visible + mask_embedded * hidden

        # =======================================================
        #  ATTENTION MASKS
        # =======================================================
        # Relational masks are precomputed by DataLoader workers on CPU.
        column_mask = batch.column_attn_mask
        feature_mask = batch.feature_attn_mask
        neighbor_mask = batch.neighbor_attn_mask
        full_mask: Bool[Tensor, "b s s"] = ~is_padding[:, :, None] & ~is_padding[:, None, :]

        # =======================================================
        # Pass input through the blocks!
        # =======================================================
        if self.use_flash:
            # Flash/SDPA mode: use dense bool masks directly
            with torch.autograd.profiler.record_function("transformer_blocks_flash"):
                for block in self.blocks:
                    x = block(x, column_mask, feature_mask, neighbor_mask, full_mask)
        else:
            # Flex attention mode: convert to sparse BlockMasks
            with torch.autograd.profiler.record_function("block_mask_creation"):
                col_attn_mask = _generate_block_mask(column_mask.contiguous(), b, s, device)
                feature_attn_mask = _generate_block_mask(feature_mask.contiguous(), b, s, device)
                neighbor_attn_mask = _generate_block_mask(neighbor_mask.contiguous(), b, s, device)
                full_attn_mask = _generate_block_mask(full_mask.contiguous(), b, s, device)

            with torch.autograd.profiler.record_function("transformer_blocks_flex"):
                for block in self.blocks:
                    x = block(
                        x,
                        col_attn_mask,
                        feature_attn_mask,
                        neighbor_attn_mask,
                        full_attn_mask,
                    )

        with torch.autograd.profiler.record_function("output_norm"):
            x = self.out_norm(x)

        # =======================================================
        # OUTPUT DECODING & LOSS
        # =======================================================
        # Run all decoders unconditionally on full tensor to avoid graph breaks.
        # Boolean indexing uses nonzero internally, which torch.compile can't
        # handle. Instead, we run decoders on all positions and mask the output.
        # This is an intentional tradeoff (extra compute to avoid graph breaks).
        with torch.autograd.profiler.record_function("output_decoding"):
            yhat_number: Float[Tensor, "b s 1"] = self.number_decoder(x) * is_number[..., None]
            yhat_datetime: Float[Tensor, "b s 1"] = self.datetime_decoder(x) * is_datetime[..., None]
            yhat_boolean: Float[Tensor, "b s 1"] = self.boolean_decoder(x) * is_boolean[..., None]
            yhat_text: Float[Tensor, "b s d_text"] = self.text_decoder(x) * is_text[..., None]

        with torch.autograd.profiler.record_function("loss_computation"):
            # Compute per-position losses (before masking)
            loss_number: Float[Tensor, "b s"] = F.huber_loss(yhat_number, number_values, reduction="none").mean(-1)
            loss_datetime: Float[Tensor, "b s"] = F.huber_loss(yhat_datetime, datetime_values, reduction="none").mean(-1)
            loss_boolean: Float[Tensor, "b s"] = F.binary_cross_entropy_with_logits(
                yhat_boolean, (boolean_values > 0).float(), reduction="none"
            ).mean(-1)

            # Select the right loss per position based on semantic type
            combined_loss: Float[Tensor, "b s"] = loss_number * is_number + loss_datetime * is_datetime + loss_boolean * is_boolean

            # Single masked sum and division
            # By fiat, we've decided that we're not allowed to mask any text, so although
            # we aren't computing loss on text positions, we're not going to get any
            # numerator contribution from the masks and so this is fine as written.
            # Dummy term touches text_decoder params for DDP gradient sync.
            loss_out: Float[Tensor, ""] = (combined_loss * masks).sum() / masks.sum() + 0.0 * yhat_text.sum()

        return ModelOutput(
            loss=loss_out,
            yhat_number=yhat_number,
            yhat_datetime=yhat_datetime,
            yhat_boolean=yhat_boolean,
            yhat_text=yhat_text,
        )
