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

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This model requires CUDA.")

allow_ops_in_compiled_graph()
flex_attention = torch.compile(flex_attention)


# We have four kinds of masked attention blocks:
class AttentionType(Enum):
    COLUMN = 0
    FEATURE = 1
    NEIGHBOR = 2
    FULL = 3


class SemanticType(Enum):
    """Semantic type of a cell value.

    Determines encoding strategy and loss function:
    - NUMERICAL: Z-score normalized scalars, regression loss (Huber).
                 Includes floats and ints.
    - CATEGORICAL: Pre-embedded as "<col_name> is <value>" text, cosine loss.
                   Includes string categories, integer codes, and booleans.
                   Enables zero-shot transfer to new databases/categories.
    - TEXT: Pre-embedded via frozen text encoder, contrastive loss (InfoNCE).
            Predicts embeddings; at inference, use nearest-neighbor retrieval.
    - TIMESTAMP: Pre-decomposed into cyclical (sin/cos) and linear components.
                 11 dimensions: 5 cyclical (minute, hour, dow, doy, month) x 2 + epoch.
                 Predicts z-scored epoch seconds via Huber loss.

    Note: Booleans are treated as categoricals (e.g., "is_active is true").
    """

    NUMERICAL = 0
    CATEGORICAL = 1
    TEXT = 2
    TIMESTAMP = 3


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
        attn_out: Float[Tensor, "b h s d"] = flex_attention(q, k, v, block_mask=mask, kernel_options={"USE_TMA": True})  # type: ignore
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
@jaxtyped(typechecker=None)
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

        self.column_attn = FlexAttentionBlock(d_model, num_heads, AttentionType.COLUMN)
        self.feature_attn = FlexAttentionBlock(d_model, num_heads, AttentionType.FEATURE)
        self.neighbor_attn = FlexAttentionBlock(d_model, num_heads, AttentionType.NEIGHBOR)
        self.full_attn = FlexAttentionBlock(d_model, num_heads, AttentionType.FULL)

        self.ffn = FFN(d_model, d_ff)

    def forward(
        self,
        x: Float[Tensor, "b s d"],
        col_block_mask: BlockMask,
        feature_block_mask: BlockMask,
        neighbor_block_mask: BlockMask,
        full_block_mask: BlockMask,
    ) -> Float[Tensor, "b s d"]:
        # Don't use += operations to avoid autograd issues
        x = x + self.column_attn(self.column_norm(x), col_block_mask)  # noqa: PLR6104
        x = x + self.feature_attn(self.feature_norm(x), feature_block_mask)  # noqa: PLR6104
        x = x + self.neighbor_attn(self.neighbor_norm(x), neighbor_block_mask)  # noqa: PLR6104
        x = x + self.full_attn(self.full_norm(x), full_block_mask)  # noqa: PLR6104
        x = x + self.ffn(self.ffn_norm(x))  # noqa: PLR6104
        return x  # noqa: RET504


# BlockMask generation for flex_attention.
# https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html
#
# We create BlockMasks in the forward pass from dense boolean masks.
# This ensures the mask tensor is a traced input to the compiled function,
# not a captured external variable (which causes Inductor lowering errors).


@jaxtyped(typechecker=None)
def generate_block_mask(
    mask: Bool[Tensor, "b s s"],
    batch_size: int,
    seq_len: int,
) -> BlockMask:
    """Generate a BlockMask from a boolean attention mask tensor.

    Must be called inside the forward pass (within torch.compile scope) so that
    the mask tensor is properly traced as an input, not captured as a closure.
    """
    return create_block_mask(
        mask_mod=lambda b, _h, q, kv: mask[b, q, kv],
        B=batch_size,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=mask.device,
    )


# A "batch" of data for the training loops.
# The sequence is a flattened list of cells from multiple rows (nodes) sampled
# via BFS traversal of the relational graph starting from a seed row in a task
# table.
# We do some "pre-work" in the data loader to compute the specific sparse
# attention masks so that we don't have to do it in the forward pass on the GPU.
@dataclass
class Batch:
    # Numeric cell values, z-score normalized.
    # Includes floats and ints (timestamps have their own semantic type).
    # NaN values are skipped during preprocessing. Val/test splits use
    # statistics computed from the training set for consistency.
    # Normalization strategy (per-column vs global) is handled in preprocessing.
    # When the cell at the position is not numerical, the value is irrelevant (masked)
    numerical_values: Float[Tensor, "b s 1"]

    # Pre-computed text embeddings for categorical values.
    # Each categorical column is embedded as "<column_name> is <value>" via
    # the frozen text encoder. Examples: "color is red", "is_active is true", "state is 3"
    # This enables zero-shot transfer to new databases/categories.
    # When the cell is not categorical, this should be irrelevant (masked).
    categorical_values: Float[Tensor, "b s d_text"]

    # Pre-computed text embeddings for textual values.
    # During preprocessing, all unique strings are embedded and stored;
    # at runtime these could looked up by index from a memory-mapped file.
    # When the cell at the position is not a text, the value vector is irrelevant (masked).
    text_values: Float[Tensor, "b s d_text"]

    # Pre-decomposed timestamp features (11 dimensions) for INPUT encoding:
    # - 5 cyclical components encoded as sin/cos pairs (10 values):
    #   [0-1] minute_of_hour, [2-3] hour_of_day, [4-5] day_of_week,
    #   [6-7] day_of_year, [8-9] month
    # - 1 linear component (z-score normalized):
    #   [10] epoch_seconds (used as prediction target)
    # When the cell is not a timestamp, values are 0.0.
    # Note: The model encodes all 9-d but decodes only epoch (1-d) for consistency.
    timestamp_values: Float[Tensor, "b s d_time"]

    # Pre-computed embeddings for column names, formatted as
    # "<column_name> of <table_name>" (e.g., "price of products").
    # TODO(mrdmnd): expand to include column descriptions!
    # Added to every cell's representation as positional context.
    # This is always present, because every cell comes from a column.
    column_name_values: Float[Tensor, "b s d_text"]

    # Semantic type determining which type is present at each position.
    # 0=numerical, 1=categorical, 2=text, 3=timestamp (see SemanticType enum).
    # Note: Booleans are encoded as categoricals (type=1).
    semantic_types: Int[Tensor, "b s"]

    # Positions to HIDE from the model (replaced with learned mask embedding)
    masks: Bool[Tensor, "b s"]

    # Whether this position is padding (sequence shorter than seq_len).
    # Padding positions are excluded from all attention masks and losses.
    is_padding: Bool[Tensor, "b s"]

    # Dense boolean attention masks (b, s, s). BlockMasks are created from these
    # in the forward pass to ensure proper torch.compile tracing.
    # Note: full_attn_mask is computed in the forward pass from is_padding.
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
            if isinstance(value, torch.Tensor):
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
        Note: full_attn_mask is just ~is_padding broadcasted, computed in forward pass.

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

        # Final masks (full_mask computed in forward pass from is_padding)
        column_mask = same_col_table & active
        feature_mask = (same_node | kv_in_f2p) & active
        neighbor_mask = q_in_p2f & active

        return column_mask, feature_mask, neighbor_mask


@jaxtyped(typechecker=None)
class ModelOutput(TypedDict):
    # The loss averaged over the full batch
    loss: Float[Tensor, ""]
    yhat_numerical: Float[Tensor, "b s 1"] | None
    yhat_categorical: Float[Tensor, "b s d_text"] | None  # Predicted embedding, use NN for class
    yhat_text: Float[Tensor, "b s d_text"] | None
    yhat_timestamp: Float[Tensor, "b s 1"] | None  # Predicted z-scored epoch seconds


@jaxtyped(typechecker=None)
class RelationalTransformer(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
    ) -> None:
        """Initialize the Relational Transformer.

        Args:
            config: Model architecture hyperparameters.
        """
        super().__init__()
        self.d_model = config.d_model
        self.d_text = config.d_text
        self.text_contrastive_temperature = config.text_contrastive_temperature

        # Set up initial embedding layers
        self.column_name_encoder = nn.Linear(config.d_text, config.d_model, bias=True)
        self.numerical_encoder = nn.Linear(1, config.d_model, bias=True)
        self.text_encoder = nn.Linear(config.d_text, config.d_model, bias=True)
        self.timestamp_encoder = nn.Linear(config.d_time, config.d_model, bias=True)

        # Categorical projection: learns to separate category embeddings that may be
        # too close in the frozen text encoder's space (e.g., "enabled is true" and
        # "enabled is false" have ~0.93 cosine similarity in BGE).
        # This projection is applied to BOTH input and target, so the loss becomes
        # more discriminative as the projection learns to push apart categories.
        self.category_projection = nn.Linear(config.d_text, config.d_text, bias=True)

        # Categorical encoder: projects the (now separated) embeddings to model dim.
        self.categorical_encoder = nn.Linear(config.d_text, config.d_model, bias=True)

        # Norms
        self.column_name_norm = nn.RMSNorm(config.d_model)
        self.numerical_norm = nn.RMSNorm(config.d_model)
        self.categorical_norm = nn.RMSNorm(config.d_model)
        self.text_norm = nn.RMSNorm(config.d_model)
        self.timestamp_norm = nn.RMSNorm(config.d_model)

        # Mask Embeddings - one per semantic type
        # Index: 0=numerical, 1=categorical, 2=text, 3=timestamp
        self.mask_embeddings = nn.Parameter(torch.randn(4, config.d_model))

        # Transformer Blocks
        self.blocks = nn.ModuleList([RelationalBlock(config.d_model, config.num_heads, config.d_ff) for _ in range(config.num_blocks)])

        # Output Norm
        self.out_norm = nn.RMSNorm(config.d_model)

        # Set up decoder layers
        self.numerical_decoder = nn.Linear(config.d_model, 1, bias=True)
        self.text_decoder = nn.Linear(config.d_model, config.d_text, bias=True)

        # Categorical decoder: predicts a d_text embedding. At inference, use
        # nearest neighbor search against precomputed category embeddings.
        self.categorical_decoder = nn.Linear(config.d_model, config.d_text, bias=True)

        # Timestamp decoder: predicts z-scored epoch seconds (1-d, like numerical).
        # Input uses full 12-d cyclical encoding; output is single epoch value.
        self.timestamp_decoder = nn.Linear(config.d_model, 1, bias=True)

    def forward(self, batch: Batch) -> ModelOutput:
        numerical_values: Float[Tensor, "b s 1"] = batch.numerical_values
        categorical_values: Float[Tensor, "b s d_text"] = batch.categorical_values
        text_values: Float[Tensor, "b s d_text"] = batch.text_values
        timestamp_values: Float[Tensor, "b s d_time"] = batch.timestamp_values
        column_name_values: Float[Tensor, "b s d_text"] = batch.column_name_values
        masks: Bool[Tensor, "b s"] = batch.masks
        is_padding: Bool[Tensor, "b s"] = batch.is_padding

        # Semantic types: [0, 1, 2, 3] -> [numerical, categorical, text, timestamp]
        semantic_type: Int[Tensor, "b s"] = batch.semantic_types
        is_numerical: Bool[Tensor, "b s"] = semantic_type == SemanticType.NUMERICAL.value
        is_categorical: Bool[Tensor, "b s"] = semantic_type == SemanticType.CATEGORICAL.value
        is_text: Bool[Tensor, "b s"] = semantic_type == SemanticType.TEXT.value
        is_timestamp: Bool[Tensor, "b s"] = semantic_type == SemanticType.TIMESTAMP.value

        # Note: Text masking is now supported via contrastive loss (InfoNCE).
        # The text loss computation uses boolean indexing which may cause a
        # torch.compile graph break, but this is acceptable for the text path.

        # =======================================================
        #  INPUT EMBEDDING STEP
        # =======================================================
        with torch.autograd.profiler.record_function("input_embedding"):
            # Project categorical embeddings to a space with better separation.
            # This same projection is used for targets in the loss computation.
            projected_categorical: Float[Tensor, "b s d_text"] = self.category_projection(categorical_values)

            encoded: Float[Tensor, "b s d"] = (
                self.numerical_norm(self.numerical_encoder(numerical_values)) * is_numerical[..., None]
                + self.categorical_norm(self.categorical_encoder(projected_categorical)) * is_categorical[..., None]
                + self.text_norm(self.text_encoder(text_values)) * is_text[..., None]
                + self.timestamp_norm(self.timestamp_encoder(timestamp_values)) * is_timestamp[..., None]
            )

            mask_embedded: Float[Tensor, "b s d"] = self.mask_embeddings[semantic_type]
            visible = (~masks & ~is_padding)[..., None]
            hidden = (masks & ~is_padding)[..., None]

            # Input to the model starts as the column name embedding, plus the encoded
            # values, plus the embeddings for whatever is masked.
            x: Float[Tensor, "b s d"] = self.column_name_norm(self.column_name_encoder(column_name_values)) * (~is_padding)[..., None]
            x = x + encoded * visible + mask_embedded * hidden

        # =======================================================
        # Create BlockMasks from dense boolean masks
        # =======================================================
        # This must happen inside the forward pass (within torch.compile scope)
        # so that the mask tensors are traced as inputs, not captured as closures.
        with torch.autograd.profiler.record_function("create_block_masks"):
            batch_size, seq_len = numerical_values.shape[:2]
            col_block_mask = generate_block_mask(batch.column_attn_mask, batch_size, seq_len)
            feature_block_mask = generate_block_mask(batch.feature_attn_mask, batch_size, seq_len)
            neighbor_block_mask = generate_block_mask(batch.neighbor_attn_mask, batch_size, seq_len)
            # full_attn_mask is just the active mask (non-padding positions), computed here
            full_attn_mask: Bool[Tensor, "b s s"] = ~is_padding[:, :, None] & ~is_padding[:, None, :]
            full_block_mask = generate_block_mask(full_attn_mask, batch_size, seq_len)

        # =======================================================
        # Pass input through the blocks!
        # =======================================================
        with torch.autograd.profiler.record_function("transformer_blocks"):
            for block in self.blocks:
                x = block(
                    x,
                    col_block_mask,
                    feature_block_mask,
                    neighbor_block_mask,
                    full_block_mask,
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
            yhat_numerical: Float[Tensor, "b s 1"] = self.numerical_decoder(x) * is_numerical[..., None]
            yhat_text: Float[Tensor, "b s d_text"] = self.text_decoder(x) * is_text[..., None]
            yhat_timestamp: Float[Tensor, "b s 1"] = self.timestamp_decoder(x) * is_timestamp[..., None]

            # Categorical decoder predicts a d_text embedding.
            # At inference, use nearest neighbor search against category embeddings.
            yhat_categorical: Float[Tensor, "b s d_text"] = self.categorical_decoder(x)

        with torch.autograd.profiler.record_function("loss_computation"):
            # Compute per-position losses (before masking)
            loss_numerical: Float[Tensor, "b s"] = F.huber_loss(yhat_numerical, numerical_values, reduction="none").mean(-1)

            # Categorical loss: cosine embedding loss in the PROJECTED space.
            # We compare predicted embedding to the PROJECTED target (not raw).
            # This allows the projection to learn to separate categories that are
            # too close in the original text embedding space.
            # Note: projected_categorical was computed earlier in input embedding.
            yhat_cat_norm = F.normalize(yhat_categorical, p=2, dim=-1)
            target_cat_norm = F.normalize(projected_categorical, p=2, dim=-1)
            # Cosine similarity: higher is better, so loss = 1 - cos_sim
            cos_sim: Float[Tensor, "b s"] = (yhat_cat_norm * target_cat_norm).sum(dim=-1)
            loss_categorical: Float[Tensor, "b s"] = 1.0 - cos_sim

            # Timestamp loss: Huber on z-scored epoch seconds (last component of input).
            # Input uses full 12-d for cyclical awareness; output predicts epoch only.
            timestamp_epoch_target: Float[Tensor, "b s 1"] = timestamp_values[..., -1:]
            loss_timestamp: Float[Tensor, "b s"] = F.huber_loss(yhat_timestamp, timestamp_epoch_target, reduction="none").mean(-1)

            # Select the right loss per position based on semantic type
            # (numerical, categorical, timestamp are per-position; text handled separately)
            combined_loss: Float[Tensor, "b s"] = loss_numerical * is_numerical + loss_categorical * is_categorical + loss_timestamp * is_timestamp

            # Compute masked loss for numerical, categorical, and timestamp
            num_cat_time_mask = masks & (is_numerical | is_categorical | is_timestamp)
            num_cat_time_count = num_cat_time_mask.sum().clamp(min=1)
            loss_num_cat_time: Float[Tensor, ""] = (combined_loss * num_cat_time_mask).sum() / num_cat_time_count

            # Text contrastive loss (InfoNCE)
            # Note: This uses boolean indexing which may cause a torch.compile graph break.
            # We accept this tradeoff since text masking is expected to be less frequent
            # than numerical/categorical masking, and contrastive loss requires gathering.
            text_mask: Bool[Tensor, "b s"] = is_text & masks
            num_text_targets: int = int(text_mask.sum().item())

            if num_text_targets > 1:
                # Gather masked text predictions and targets
                pred_text_flat: Float[Tensor, "n d_text"] = yhat_text[text_mask]
                target_text_flat: Float[Tensor, "n d_text"] = text_values[text_mask]

                # Normalize for cosine similarity
                pred_norm = F.normalize(pred_text_flat, dim=-1)
                target_norm = F.normalize(target_text_flat, dim=-1)

                # Compute similarity matrix: (N, d) @ (d, N) -> (N, N)
                # Each row i contains similarities between prediction i and all targets
                logits: Float[Tensor, "n n"] = pred_norm @ target_norm.T / self.text_contrastive_temperature

                # Labels: diagonal (prediction i should match target i)
                labels = torch.arange(num_text_targets, device=logits.device)
                loss_text: Float[Tensor, ""] = F.cross_entropy(logits, labels)

            elif num_text_targets == 1:
                # Single text target: fall back to cosine loss (can't do contrastive)
                pred_text_flat = yhat_text[text_mask]
                target_text_flat = text_values[text_mask]
                cos_sim_text = F.cosine_similarity(pred_text_flat, target_text_flat, dim=-1)
                loss_text = (1.0 - cos_sim_text).mean()

            else:
                # No text targets: dummy term to touch text_decoder params for DDP gradient sync
                loss_text = yhat_text.sum() * 0.0

            # Combine losses
            # Weight text loss equally with num/cat/timestamp loss (both contribute to total)
            loss_out: Float[Tensor, ""] = loss_num_cat_time + loss_text

        return ModelOutput(
            loss=loss_out,
            yhat_numerical=yhat_numerical,
            yhat_categorical=yhat_categorical,
            yhat_text=yhat_text,
            yhat_timestamp=yhat_timestamp,
        )
