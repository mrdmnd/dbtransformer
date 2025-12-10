# An implementation of the relational transformer model.
# https://arxiv.org/abs/2510.06377
# Model sizing from the paper:
# L = 12
# d_text = 384
# d_model = 256
# d_ff = 1024
# num_heads = 8
# batch_size = 128
# seq_len = 1024


from enum import Enum
from typing import TypedDict

import torch
import torch.nn.functional as F  # noqa: N812
from beartype import beartype
from einops import rearrange
from einops._torch_specific import allow_ops_in_compiled_graph  # noqa: PLC2701
from jaxtyping import Bool, Float, Int, jaxtyped
from loguru import logger
from torch import Tensor, nn
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)

from dbtransformer.hardware_abstraction_layer import HardwareConfig

logger.info("Initializing model.py...")
allow_ops_in_compiled_graph()

# TODO(mrdmnd): only do this if we're using flex attention; this will break on MPS
logger.info("Compiling flex_attention...")
flex_attention = torch.compile(flex_attention)


# We have four kinds of masked attention blocks:
class AttentionType(Enum):
    COLUMN = 0
    FEATURE = 1
    NEIGHBOR = 2
    FULL = 3


# Different cells are treated differently by the architecture.
class SemanticsType(Enum):
    NUMBER = 0
    TEXT = 1
    DATETIME = 2
    BOOLEAN = 3
    # TODO(mrdmnd): categorical / enum? If you know it has to be one of a few things?
    # TODO(mrdmnd): learn this heuristically?


# Maximum number of foreign-to-primary neighbors per cell.
MAX_F2P_NEIGHBORS = 5


# This is a generic masked attention block that can be used for any attention type.
# We save the attention type as a class variable for debugging.
@jaxtyped(typechecker=beartype)
class GenericMaskedAttentionBlock(nn.Module):
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
        mask: BlockMask | Bool[Tensor, "b s s"] | None,
    ) -> Float[Tensor, "b s d"]:
        q: Float[Tensor, "b s d"] = self.wq(x)
        k: Float[Tensor, "b s d"] = self.wk(x)
        v: Float[Tensor, "b s d"] = self.wv(x)

        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

        if isinstance(mask, BlockMask):
            # flex_attention returns Tensor when return_lse=False (default)
            # we know this, but mypy doesn't - no cast, but just ignoring the warning.
            attn_out: Float[Tensor, "b h s d"] = flex_attention(q, k, v, block_mask=mask)  # type: ignore[assignment]
        elif isinstance(mask, Tensor):
            # MPS/CPU fallback: use SDPA with dense bool mask
            # Expand (b, s, s) -> (b, num_heads, s, s)
            attn_mask = mask[:, None, :, :].expand(-1, self.num_heads, -1, -1)
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            # No mask: full attention
            attn_out = F.scaled_dot_product_attention(q, k, v)

        out: Float[Tensor, "b s d"] = rearrange(attn_out, "b h s d -> b s (h d)")
        return self.wo(out)


# Bog-standard FFN with no biases. Uses SwiGLU activation.
@jaxtyped(typechecker=beartype)
class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Float[Tensor, "b s d"]) -> Float[Tensor, "b s d"]:
        return self.down(F.silu(self.gate(x)) * self.up(x))


# Implements the "Relational Transformer Block" from the paper.
@jaxtyped(typechecker=beartype)
class RelationalBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
    ) -> None:
        super().__init__()
        self.column_norm = nn.RMSNorm(d_model)
        self.feature_norm = nn.RMSNorm(d_model)
        self.neighbor_norm = nn.RMSNorm(d_model)
        self.full_norm = nn.RMSNorm(d_model)
        self.ffn_norm = nn.RMSNorm(d_model)

        self.column_attn = GenericMaskedAttentionBlock(d_model, num_heads, AttentionType.COLUMN)
        self.feature_attn = GenericMaskedAttentionBlock(d_model, num_heads, AttentionType.FEATURE)
        self.neighbor_attn = GenericMaskedAttentionBlock(d_model, num_heads, AttentionType.NEIGHBOR)
        self.full_attn = GenericMaskedAttentionBlock(d_model, num_heads, AttentionType.FULL)

        self.ffn = FFN(d_model, d_ff)

    def forward(
        self,
        x: Float[Tensor, "b s d"],
        masks: dict[AttentionType, BlockMask | Bool[Tensor, "b s s"] | None],
    ) -> Float[Tensor, "b s d"]:
        # Don't use += operations to avoid autograd issues
        x = x + self.column_attn(self.column_norm(x), masks[AttentionType.COLUMN])  # noqa: PLR6104
        x = x + self.feature_attn(self.feature_norm(x), masks[AttentionType.FEATURE])  # noqa: PLR6104
        x = x + self.neighbor_attn(self.neighbor_norm(x), masks[AttentionType.NEIGHBOR])  # noqa: PLR6104
        x = x + self.full_attn(self.full_norm(x), masks[AttentionType.FULL])  # noqa: PLR6104
        x = x + self.ffn(self.ffn_norm(x))  # noqa: PLR6104
        return x  # noqa: RET504


# This is torch.flex_attention's BlockMask.
# https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html
# https://github.com/pytorch/pytorch/blob/v2.9.1/torch/nn/attention/flex_attention.py
# The mask_mod signature is Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]
# but the input tensors are 1d scalars, basically.
@jaxtyped(typechecker=beartype)
def _generate_block_mask(
    mask: Bool[Tensor, "b s s"],
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> BlockMask:
    # This is a "mask modifier" function that flex attention uses to determine
    # if a query position can attend to a key-value position. The parameters are
    # - b: batch index
    # - h: head index
    # - q_idx: query index
    # - kv_idx: key-value index

    def mask_mod(
        b: Int[Tensor, " b"],  # Batch size
        _h: Int[Tensor, " h"],  # Num heads; unused. All heads share same mask.
        q_idx: Int[Tensor, " q_idx"],  # Query index
        kv_idx: Int[Tensor, " kv_idx"],  # Key-value index
    ) -> Bool[Tensor, "b h q_idx kv_idx"]:
        return mask[b, q_idx, kv_idx]

    return create_block_mask(
        mask_mod=mask_mod,
        B=batch_size,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device,
        _compile=True,
    )


# A "batch" of data for the training loops.
# The sequence is a flattened list of cells from multiple rows (nodes) sampled
# via BFS traversal of the relational graph starting from a seed row in a task table.
@jaxtyped(typechecker=beartype)
class Batch(TypedDict):
    # Row index for each cell. Multiple cells share the same node_index when
    # they belong to the same row. Used to compute attention masks:
    # - "same_node" for feature attention (cells in the same row attend)
    # - "kv_in_f2p" / "q_in_p2f" for neighbor attention (parent/child rows)
    node_indices: Int[Tensor, "b s"]

    # Unique integer ID for this cell's table. Used with column_name_indices
    # for equality comparison to compute the "same column AND same table"
    # attention mask for column attention. Not used for embedding lookup.
    table_name_indices: Int[Tensor, "b s"]

    # Unique integer ID for this cell's column (globally unique across all
    # tables). Used with table_name_indices for equality comparison to
    # compute column attention mask. Not used for embedding lookup -
    # the actual column name embeddings come via column_name_values.
    column_name_indices: Int[Tensor, "b s"]

    # Foreign-to-primary (parent) neighbor indices for each cell's row.
    # If a row has a foreign key referencing another row, that parent row's
    # index appears here. Padded with -1 if fewer than MAX_F2P_NEIGHBORS.
    # Used for feature attention (attend to parent rows) and neighbor
    # attention (attend to child rows via the reverse relationship).

    # The real shape is (b, s, MAX_F2P_NEIGHBORS), but jaxtyping is going to want a
    # constant, so here's the magic number five.
    f2p_neighbor_indices: Int[Tensor, "b s 5"]

    # Numeric cell values, z-score normalized per column: (val - mean) / std.
    # NaN values are skipped during preprocessing. Val/test splits use
    # statistics computed from the training set for consistency.
    number_values: Float[Tensor, "b s 1"]

    # Datetime cell values (converted to nanoseconds), z-score normalized using
    # *global* statistics computed across ALL datetime columns in the entire
    # database, not per-column. This allows cross-table temporal reasoning.
    datetime_values: Float[Tensor, "b s 1"]

    # Boolean cell values, converted to 0.0/1.0 then z-score normalized
    # per column (same as number_values). Not raw 0/1!
    boolean_values: Float[Tensor, "b s 1"]

    # Pre-computed text embeddings from some SentenceTransformer (e.g. MiniLM).
    # During preprocessing, all unique strings are embedded and stored;
    # at runtime these could looked up by index from a memory-mapped file.
    text_values: Float[Tensor, "b s d_text"]

    # Pre-computed embeddings for column names, formatted as
    # "<column_name> of <table_name>" (e.g., "price of products").
    # TODO(mrdmnd): expand to include column descriptions!
    # Added to every cell's representation as positional context.
    column_name_values: Float[Tensor, "b s d_text"]

    # Semantic type determining which encoder/decoder head to use:
    # 0=number, 1=text, 2=datetime, 3=boolean (see SemanticsType enum).
    semantic_types: Int[Tensor, "b s"]

    # Positions to HIDE from the model (replaced with learned mask embedding)
    # and compute loss on. This is the MLM-style training signal.
    # In current impl, set identically to is_targets (only mask the target).
    masks: Bool[Tensor, "b s"]

    # Positions that are actual prediction targets for evaluation metrics.
    # NOT used by the model internally - only used in the training loop to
    # extract predictions (yhat[is_targets]) for computing AUC/R² scores.
    # Separated from `masks` to support pretraining with additional random
    # masking while still tracking the true task targets.
    is_targets: Bool[Tensor, "b s"]

    # Whether this cell belongs to a task table row (train/val/test split)
    # vs a regular database table row. Task tables contain the prediction
    # targets; DB tables provide relational context.
    is_task_nodes: Bool[Tensor, "b s"]

    # Whether this position is padding (sequence shorter than seq_len).
    # Padding positions are excluded from all attention masks and losses.
    is_padding: Bool[Tensor, "b s"]

    # For categorical/string cell values, an integer ID for the string.
    # Set to -1 for non-categorical types. Currently unused in the model
    # but could enable a categorical prediction head in the future
    # (predict which category from a vocabulary).
    class_value_indices: Int[Tensor, "b s"]

    # Actual number of valid samples in this batch (not padded duplicates).
    # The last batch may have fewer samples; positions >= true_batch_size
    # are duplicates that should be ignored during evaluation.
    true_batch_size: int


@jaxtyped(typechecker=beartype)
class ModelOutput(TypedDict):
    # The loss averaged over the full batch
    loss: Float[Tensor, ""]
    yhat_number: Float[Tensor, "b s 1"] | None
    yhat_datetime: Float[Tensor, "b s 1"] | None
    yhat_boolean: Float[Tensor, "b s 1"] | None
    yhat_text: Float[Tensor, "b s d_text"] | None


@jaxtyped(typechecker=beartype)
class RelationalTransformer(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        d_model: int,
        d_text: int,
        num_heads: int,
        d_ff: int,
        hardware_config: HardwareConfig,
    ) -> None:
        super().__init__()
        self.hardware_config = hardware_config
        self.d_model = d_model

        # Set up initial embedding layers
        self.column_name_encoder = nn.Linear(d_text, d_model, bias=True)
        self.number_encoder = nn.Linear(1, d_model, bias=True)
        self.text_encoder = nn.Linear(d_text, d_model, bias=True)
        self.datetime_encoder = nn.Linear(1, d_model, bias=True)
        self.boolean_encoder = nn.Linear(1, d_model, bias=True)

        # Norms
        self.column_name_norm = nn.RMSNorm(d_model)
        self.number_norm = nn.RMSNorm(d_model)
        self.text_norm = nn.RMSNorm(d_model)
        self.datetime_norm = nn.RMSNorm(d_model)
        self.boolean_norm = nn.RMSNorm(d_model)

        # Mask Embeddings
        self.number_mask_embedding = nn.Parameter(torch.randn(d_model))
        self.text_mask_embedding = nn.Parameter(torch.randn(d_model))
        self.datetime_mask_embedding = nn.Parameter(torch.randn(d_model))
        self.boolean_mask_embedding = nn.Parameter(torch.randn(d_model))

        # Transformer Blocks
        self.blocks = nn.ModuleList([RelationalBlock(d_model, num_heads, d_ff) for _ in range(num_blocks)])

        # Output Norm
        self.out_norm = nn.RMSNorm(d_model)

        # Set up decoder layers
        self.number_decoder = nn.Linear(d_model, 1, bias=True)
        self.datetime_decoder = nn.Linear(d_model, 1, bias=True)
        self.boolean_decoder = nn.Linear(d_model, 1, bias=True)
        self.text_decoder = nn.Linear(d_model, d_text, bias=True)

    def forward(self, batch: Batch) -> ModelOutput:  # noqa: PLR0915
        node_indices: Int[Tensor, "b s"] = batch["node_indices"]
        f2p_neighbor_indices: Int[Tensor, "b s 5"] = batch["f2p_neighbor_indices"]

        number_values: Float[Tensor, "b s d"] = batch["number_values"]
        text_values: Float[Tensor, "b s d"] = batch["text_values"]
        datetime_values: Float[Tensor, "b s d"] = batch["datetime_values"]
        boolean_values: Float[Tensor, "b s d"] = batch["boolean_values"]
        column_name_values: Float[Tensor, "b s d"] = batch["column_name_values"]

        column_name_indices: Int[Tensor, "b s"] = batch["column_name_indices"]
        table_name_indices: Int[Tensor, "b s"] = batch["table_name_indices"]
        is_padding: Bool[Tensor, "b s"] = batch["is_padding"]
        masks: Bool[Tensor, "b s"] = batch["masks"]

        # Semantics types: [0, 1, 2, 3] -> [number, text, datetime, boolean]
        semantic_type: Int[Tensor, "b s"] = batch["semantic_types"]

        # Disallow masked text positions for now - we don't predict text yet.
        # NOTE: This check is expensive (forces GPU sync) so only uncomment / run in debug mode.
        # masked_text = masks & (semantic_type == SemanticsType.TEXT.value)
        # if masked_text.any():
        #   raise ValueError("Masked text positions not supported yet.")

        b: int = node_indices.shape[0]  # Batch Size
        s: int = node_indices.shape[1]  # Sequence Length
        device = node_indices.device

        # Padding mask for attention pairs - only non-pad can attend to non-pad
        pad: Bool[Tensor, "b s s"] = ~is_padding[:, :, None] & ~is_padding[:, None, :]

        # Cells in the same node can attend to each other
        same_node: Bool[Tensor, "b s s"] = node_indices[:, :, None] == node_indices[:, None, :]

        # If the KV index is in A's foreign-to-primary (parent) neighbohood set
        kv_in_f2p: Bool[Tensor, "b s s"] = (node_indices[:, None, :, None] == f2p_neighbor_indices[:, :, None, :]).any(dim=-1)

        # If the Q index is in KV's primary-to-foreign (child) neighborhood set
        q_in_p2f: Bool[Tensor, "b s s"] = (node_indices[:, :, None, None] == f2p_neighbor_indices[:, None, :, :]).any(dim=-1)

        # If this is the same column AND same table
        same_column_and_table: Bool[Tensor, "b s s"] = (column_name_indices[:, :, None] == column_name_indices[:, None, :]) & (
            table_name_indices[:, :, None] == table_name_indices[:, None, :]
        )

        # Final attention masks, per-type (dense bool tensors)
        column_mask: Bool[Tensor, "b s s"] = same_column_and_table & pad
        feature_mask: Bool[Tensor, "b s s"] = (same_node | kv_in_f2p) & pad
        neighbor_mask: Bool[Tensor, "b s s"] = q_in_p2f & pad
        full_mask: Bool[Tensor, "b s s"] = pad

        # Build attention masks - use BlockMask on CUDA, dense bool on MPS/CPU
        if self.hardware_config.use_flex_attention:
            # CUDA: convert to BlockMask for flex_attention
            attn_masks: dict[AttentionType, BlockMask | Bool[Tensor, "b s s"] | None] = {
                AttentionType.COLUMN: _generate_block_mask(
                    column_mask.contiguous(),
                    b,
                    s,
                    device,
                ),
                AttentionType.FEATURE: _generate_block_mask(
                    feature_mask.contiguous(),
                    b,
                    s,
                    device,
                ),
                AttentionType.NEIGHBOR: _generate_block_mask(
                    neighbor_mask.contiguous(),
                    b,
                    s,
                    device,
                ),
                AttentionType.FULL: _generate_block_mask(
                    full_mask.contiguous(),
                    b,
                    s,
                    device,
                ),
            }
        else:
            # MPS/CPU: pass dense bool masks directly for SDPA
            attn_masks = {
                AttentionType.COLUMN: column_mask.contiguous(),
                AttentionType.FEATURE: feature_mask.contiguous(),
                AttentionType.NEIGHBOR: neighbor_mask.contiguous(),
                AttentionType.FULL: full_mask.contiguous(),
            }

        # The forward pass!

        # =======================================================
        #  INPUT EMBEDDING STEP
        # =======================================================
        # Stack all encoded values
        encoded_stack: Float[Tensor, "b s 4 d"] = torch.stack(
            [
                self.number_norm(self.number_encoder(number_values)),
                self.text_norm(self.text_encoder(text_values)),
                self.datetime_norm(self.datetime_encoder(datetime_values)),
                self.boolean_norm(self.boolean_encoder(boolean_values)),
            ],
            dim=2,
        )

        # Stack the (learned parameter) mask embeddings
        mask_emb_stack: Float[Tensor, "4 d"] = torch.stack(
            [
                self.number_mask_embedding,
                self.text_mask_embedding,
                self.datetime_mask_embedding,
                self.boolean_mask_embedding,
            ],
            dim=0,
        )

        # Gather the appropriate encoded value per-position based on semantic type
        idx = semantic_type[..., None, None].expand(-1, -1, 1, self.d_model)
        selected_encoded: Float[Tensor, "b s d"] = encoded_stack.gather(dim=2, index=idx).squeeze(2)
        selected_mask_emb: Float[Tensor, "b s d"] = mask_emb_stack[semantic_type]

        visible = (~masks & ~is_padding)[..., None]
        hidden = (masks & ~is_padding)[..., None]

        # The starting point is the column name embedding, plus the encoded values, plus the mask embeddings.
        x: Float[Tensor, "b s d"] = (
            self.column_name_norm(self.column_name_encoder(column_name_values)) * (~is_padding)[..., None]
            + (selected_encoded * visible)
            + (selected_mask_emb * hidden)
        )

        # =======================================================
        #  RUN THE BLOCKS
        # =======================================================
        for block in self.blocks:
            x = block(x, attn_masks)

        x = self.out_norm(x)

        # OUTPUT DECODING & LOSS
        # Compute all predictions at once
        yhat_number: Float[Tensor, "b s 1"] = self.number_decoder(x)
        yhat_datetime: Float[Tensor, "b s 1"] = self.datetime_decoder(x)
        yhat_boolean: Float[Tensor, "b s 1"] = self.boolean_decoder(x)
        yhat_text: Float[Tensor, "b s d_text"] = self.text_decoder(x)

        # Compute per-position losses (before masking)
        loss_number: Float[Tensor, "b s"] = F.huber_loss(yhat_number, number_values, reduction="none").mean(-1)
        loss_datetime: Float[Tensor, "b s"] = F.huber_loss(yhat_datetime, datetime_values, reduction="none").mean(-1)
        loss_boolean: Float[Tensor, "b s"] = F.binary_cross_entropy_with_logits(yhat_boolean, (boolean_values > 0).float(), reduction="none").mean(-1)

        # Build type selector masks
        is_number: Bool[Tensor, "b s"] = semantic_type == SemanticsType.NUMBER.value
        is_datetime: Bool[Tensor, "b s"] = semantic_type == SemanticsType.DATETIME.value
        is_boolean: Bool[Tensor, "b s"] = semantic_type == SemanticsType.BOOLEAN.value

        # After computing combined_loss, before the final loss computation:
        # Touch all decoder params to avoid DDP unused parameter errors
        # in the case where we have, say, no masked boolean positions, so there are
        # no boolean gradients computed.
        dummy = yhat_number.sum() * 0.0 + yhat_datetime.sum() * 0.0 + yhat_boolean.sum() * 0.0 + yhat_text.sum() * 0.0

        # Select the right loss per position based on semantic type
        combined_loss: Float[Tensor, "b s"] = loss_number * is_number + loss_datetime * is_datetime + loss_boolean * is_boolean

        # Single masked sum and division
        loss_out: Float[Tensor, ""] = (combined_loss * masks).sum() / masks.sum() + dummy
        return ModelOutput(
            loss=loss_out,
            yhat_number=yhat_number,
            yhat_datetime=yhat_datetime,
            yhat_boolean=yhat_boolean,
            yhat_text=yhat_text,
        )
