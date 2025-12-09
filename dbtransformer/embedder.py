"""
A module for embedding strings into a fixed-size vector space.

Uses a fixed embedding model to place strings into a vector space.

The paper uses MiniLM-L12-V2 w/ 384 dims; I want to try a more modern model.

https://huggingface.co/Qwen/Qwen3-Embedding-0.6B

Embedding dimension is 1024 here.
"""

import random
import time

import torch
import torch.nn.functional as F  # noqa: N812
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from loguru import logger
from torch import Tensor
from transformers import AutoModel, AutoTokenizer
from transformers.utils import is_flash_attn_2_available


@jaxtyped(typechecker=beartype)
def last_token_pooling(
    hidden_states: Float[Tensor, "b s d"],
    attention_mask: Int[Tensor, "b s"],
) -> Float[Tensor, "b d"]:
    """Pool by selecting the last non-padding token's hidden state."""
    # Fast path: left-padded sequences always have valid last token
    if attention_mask[:, -1].all():
        return hidden_states[:, -1]

    # General case: find each sequence's last valid token
    last_idx = attention_mask.sum(dim=1) - 1
    batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
    return hidden_states[batch_idx, last_idx]


class FrozenEmbedder:
    def __init__(self, mrl_dimension: int = 1024) -> None:
        logger.info("Initializing FrozenEmbedder...")
        self.mrl_dimension = min(mrl_dimension, 1024)
        if self.mrl_dimension != mrl_dimension:
            logger.warning(f"Truncating MRL dimension from {mrl_dimension} to {self.mrl_dimension} because it's too large")
        self.model_name = "Qwen/Qwen3-Embedding-0.6B"
        # This might be too long when the texts are like "<col_name> of <table_name>" etc
        self.max_length = 1024
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side="left",
            truncation_side="left",
        )

        logger.info("Loading model...")

        # Determine best available device and attention implementation
        if is_flash_attn_2_available():
            logger.info("Flash Attention 2 is available. Using CUDA.")
            self.model = AutoModel.from_pretrained(
                self.model_name,
                attn_implementation="flash_attention_2",
                dtype=torch.bfloat16,
            ).cuda()
        elif torch.backends.mps.is_available():
            logger.info("Using MPS (Apple Silicon GPU).")
            self.model = AutoModel.from_pretrained(
                self.model_name,
                attn_implementation="sdpa",
                dtype=torch.float16,
            ).to("mps")
        else:
            logger.warning("No GPU available. Using CPU.")
            self.model = AutoModel.from_pretrained(
                self.model_name,
                attn_implementation="sdpa",
                dtype=torch.float16,
            ).cpu()

    @torch.no_grad()
    def embed(self, texts: list[str]) -> Float[Tensor, "b d"]:
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.model.device)
        outputs = self.model(**batch)
        embeddings = last_token_pooling(
            outputs.last_hidden_state,
            batch["attention_mask"],
        )
        # Matryoshka: truncate and renormalize
        embeddings = embeddings[:, : self.mrl_dimension]
        return F.normalize(embeddings, p=2, dim=1)


# Microbenchmarks
# These are not part of the public API.


def _generate_synthetic_strings(n: int, seed: int = 42) -> list[str]:
    """Generate n synthetic strings for benchmarking."""
    random.seed(seed)

    # Word pools for generating varied synthetic text
    nouns = [
        "table",
        "column",
        "database",
        "user",
        "product",
        "order",
        "customer",
        "invoice",
        "payment",
        "transaction",
        "account",
        "record",
        "field",
        "index",
        "query",
        "schema",
        "view",
        "constraint",
        "key",
        "relation",
    ]
    adjectives = [
        "primary",
        "foreign",
        "unique",
        "indexed",
        "nullable",
        "required",
        "active",
        "pending",
        "archived",
        "deleted",
        "updated",
        "created",
    ]
    verbs = [
        "contains",
        "references",
        "links",
        "stores",
        "tracks",
        "manages",
        "aggregates",
        "filters",
        "joins",
        "groups",
        "orders",
        "limits",
    ]
    prepositions = ["of", "for", "in", "from", "to", "with", "by", "on"]

    strings = []
    for _ in range(n):
        # Vary string length and structure
        pattern = random.choice(
            [
                # Short: "column_name of table_name"
                lambda: f"{random.choice(adjectives)}_{random.choice(nouns)} {random.choice(prepositions)} {random.choice(nouns)}",
                # Medium: "The X column contains Y data"
                lambda: f"The {random.choice(adjectives)} {random.choice(nouns)} {random.choice(verbs)} {random.choice(nouns)} data",
                # Longer: descriptive sentence
                lambda: f"This {random.choice(adjectives)} {random.choice(nouns)} "
                f"{random.choice(verbs)} {random.choice(adjectives)} "
                f"{random.choice(nouns)} {random.choice(prepositions)} "
                f"the {random.choice(nouns)}",
            ]
        )
        strings.append(pattern())
    return strings


def _run_benchmark(
    batch_size: int = 256,
    num_batches: int = 10,
    warmup_batches: int = 2,
) -> None:
    """Benchmark embedding throughput with synthetic strings."""
    total_strings = (warmup_batches + num_batches) * batch_size
    logger.info(f"Generating {total_strings} synthetic strings...")
    all_strings = _generate_synthetic_strings(total_strings)

    logger.info("Initializing embedder...")
    embedder = FrozenEmbedder(mrl_dimension=1024)

    # Count tokens per string for throughput calculation
    logger.info("Tokenizing all strings...")
    token_counts = [len(embedder.tokenizer.encode(s)) for s in all_strings]
    total_tokens = sum(token_counts)
    logger.info(f"Total tokens: {total_tokens}")

    # Warmup
    logger.info(f"Running {warmup_batches} warmup batches...")
    for i in range(warmup_batches):
        start_idx = i * batch_size
        batch = all_strings[start_idx : start_idx + batch_size]
        _ = embedder.embed(batch)

    # Synchronize before timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    logger.info(f"Running {num_batches} timed batches of {batch_size}...")
    batch_times = []
    batch_tokens = []
    for i in range(num_batches):
        start_idx = (warmup_batches + i) * batch_size
        end_idx = start_idx + batch_size
        batch = all_strings[start_idx:end_idx]
        tokens_in_batch = sum(token_counts[start_idx:end_idx])

        start = time.perf_counter()
        _ = embedder.embed(batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        batch_times.append(elapsed)
        batch_tokens.append(tokens_in_batch)
        logger.info(f"  Batch {i + 1}/{num_batches}: {elapsed:.3f}s ({tokens_in_batch / elapsed:.1f} tokens/s)")

    # Report summary
    total_time = sum(batch_times)
    total_strings_timed = num_batches * batch_size
    total_tokens_timed = sum(batch_tokens)
    avg_time = total_time / num_batches
    throughput = total_tokens_timed / total_time

    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Batch size:        {batch_size}")
    print(f"Batches timed:     {num_batches}")
    print(f"Total strings:     {total_strings_timed}")
    print(f"Total tokens:      {total_tokens_timed}")
    print(f"Total time:        {total_time:.3f}s")
    print(f"Avg time/batch:    {avg_time:.3f}s")
    print(f"Throughput:        {throughput:.1f} tokens/s")
    print(f"Min batch time:    {min(batch_times):.3f}s")
    print(f"Max batch time:    {max(batch_times):.3f}s")
    print("=" * 50)


if __name__ == "__main__":
    _run_benchmark(batch_size=128, num_batches=10, warmup_batches=2)
