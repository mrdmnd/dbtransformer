//! Integration test for the Sampler with a real preprocessed database.
//!
//! This test loads the rel-event database and generates batches,
//! printing information about each cell in the sequence.
//!
//! Run with: cargo test --release --test sampler_integration -- --nocapture
//!
//! Note: mimalloc is used via the library's #[global_allocator]

use std::path::Path;
use std::time::Instant;
use tributary::{Database, MaskingStrategy, Sampler, SamplerConfig};

const DB_PATH: &str = "/tmp/databases_preprocessed/rel-event.rkyv";

#[test]
fn test_load_and_sample_rel_event() {
    // Check if the file exists
    let path = Path::new(DB_PATH);
    if !path.exists() {
        eprintln!("Skipping test: {} does not exist", DB_PATH);
        eprintln!("Run the preprocessor first to generate this file.");
        return;
    }

    // Load the database directly to print stats
    println!("\n=== Loading Database ===");
    let db = Database::load(path).expect("Failed to load database");
    println!("Number of tables: {}", db.num_tables());
    println!("Number of columns: {}", db.num_columns());
    println!("Number of rows: {}", db.num_rows());
    println!("Number of edges: {}", db.num_edges());
    println!("Vocab size: {}", db.vocab_size());
    println!("Embedding dimension: {}", db.embed_dim);

    // Print table info
    println!("\n=== Tables ===");
    for table in &db.tables {
        println!(
            "  {}: {} columns, {} rows (cols {}-{}, rows {}-{})",
            table.name,
            table.num_columns(),
            table.num_rows(),
            table.column_range.0.0,
            table.column_range.1.0,
            table.row_range.0.0,
            table.row_range.1.0
        );
    }

    // Print column info
    println!("\n=== Columns ===");
    for col in &db.columns {
        let table_name = &db.tables[col.table_idx.0 as usize].name;
        println!(
            "  {}.{}: {:?}{}{}",
            table_name,
            col.name,
            col.stype,
            if col.is_primary_key { " [PK]" } else { "" },
            if col.fk_target_column.is_some() {
                " [FK]"
            } else {
                ""
            }
        );
    }

    // Create a sampler with batch_size=1, small seq_len for testing
    println!("\n=== Creating Sampler ===");
    let config = SamplerConfig {
        batch_size: 1,
        seq_len: 64, // Small for testing
        max_bfs_width: 32,
        masking_strategy: MaskingStrategy::Random { mask_rate: 0.15 },
        seed: 42,
    };

    let sampler = Sampler::from_path(path, config).expect("Failed to create sampler");
    println!("Number of batches: {}", sampler.num_batches());
    println!("Embed dim: {}", sampler.database().embed_dim);

    // Generate a batch
    println!("\n=== Generating Batch 0 ===");
    let batch = sampler.get_batch(0);
    let seq_len = sampler.config().seq_len;
    let d_text = sampler.database().embed_dim as usize;

    // Count non-padding cells
    let non_padding_count = batch.is_padding().iter().filter(|&&p| !p).count();
    println!("Non-padding cells: {} / {}", non_padding_count, seq_len);

    // Count masked cells
    let masked_count = batch.masks().iter().filter(|&&m| m).count();
    println!("Masked cells: {}", masked_count);

    // Print info for each cell in the sequence
    println!("\n=== Cell Details (first 20 non-padding cells) ===");
    let mut printed = 0;
    for i in 0..seq_len {
        if batch.is_padding()[i] {
            continue;
        }

        let stype = batch.semantic_types()[i];
        let masked = batch.masks()[i];

        let stype_name = match stype {
            0 => "Numerical",
            1 => "Categorical",
            2 => "Text",
            3 => "Timestamp",
            _ => "Unknown",
        };

        // Get value info based on semantic type
        let value_info = match stype {
            0 => {
                // Numerical - get the z-scored value
                let val = batch.numerical_values()[i];
                format!("value={:.4}", val)
            }
            1 => {
                // Categorical - show first few embedding values
                let start = i * d_text;
                let emb = &batch.categorical_values()[start..start + d_text.min(4)];
                let emb_vals: Vec<String> = emb.iter().map(|v| format!("{:.3}", v)).collect();
                format!("emb=[{}...]", emb_vals.join(", "))
            }
            2 => {
                // Text - show first few embedding values
                let start = i * d_text;
                let emb = &batch.text_values()[start..start + d_text.min(4)];
                let emb_vals: Vec<String> = emb.iter().map(|v| format!("{:.3}", v)).collect();
                format!("emb=[{}...]", emb_vals.join(", "))
            }
            3 => {
                // Timestamp - show the expanded timestamp features
                let start = i * 11; // TIMESTAMP_DIM = 11
                let ts = &batch.timestamp_values()[start..start + 11];
                format!(
                    "sin_min={:.3}, cos_min={:.3}, zscore={:.3}",
                    ts[0], ts[1], ts[10]
                )
            }
            _ => "unknown".to_string(),
        };

        println!(
            "  Cell {}: {} | {} | masked={}",
            i, stype_name, value_info, masked
        );

        printed += 1;
        if printed >= 20 {
            break;
        }
    }

    // Show semantic type distribution
    println!("\n=== Semantic Type Distribution ===");
    let mut type_counts = [0usize; 4];
    for i in 0..seq_len {
        if !batch.is_padding()[i] {
            let stype = batch.semantic_types()[i] as usize;
            if stype < 4 {
                type_counts[stype] += 1;
            }
        }
    }
    println!("  Numerical: {}", type_counts[0]);
    println!("  Categorical: {}", type_counts[1]);
    println!("  Text: {}", type_counts[2]);
    println!("  Timestamp: {}", type_counts[3]);

    // Generate a few more batches to verify consistency
    println!("\n=== Batch Statistics (first 5 batches) ===");
    for batch_idx in 0..5.min(sampler.num_batches()) {
        let batch = sampler.get_batch(batch_idx);
        let non_padding = batch.is_padding().iter().filter(|&&p| !p).count();
        let masked = batch.masks().iter().filter(|&&m| m).count();
        let mask_rate = if non_padding > 0 {
            masked as f64 / non_padding as f64
        } else {
            0.0
        };
        println!(
            "  Batch {}: {} cells, {} masked ({:.1}%)",
            batch_idx,
            non_padding,
            masked,
            mask_rate * 100.0
        );
    }

    println!("\n=== Test Complete ===\n");
}

#[test]
fn test_sampler_shuffle() {
    let path = Path::new(DB_PATH);
    if !path.exists() {
        eprintln!("Skipping test: {} does not exist", DB_PATH);
        return;
    }

    let config = SamplerConfig {
        batch_size: 1,
        seq_len: 32,
        max_bfs_width: 16,
        masking_strategy: MaskingStrategy::Random { mask_rate: 0.15 },
        seed: 42,
    };

    let mut sampler = Sampler::from_path(path, config).expect("Failed to create sampler");

    // Get batch 0 before shuffle
    let batch_before = sampler.get_batch(0);
    let types_before: Vec<i32> = batch_before.semantic_types().to_vec();

    // Shuffle for epoch 1
    sampler.shuffle(1);

    // Get batch 0 after shuffle - should be different (different seed row)
    let batch_after = sampler.get_batch(0);
    let types_after: Vec<i32> = batch_after.semantic_types().to_vec();

    // They should typically be different (different starting row)
    // This isn't guaranteed but is extremely likely with a large database
    println!(
        "Before shuffle: first 10 types = {:?}",
        &types_before[..10.min(types_before.len())]
    );
    println!(
        "After shuffle: first 10 types = {:?}",
        &types_after[..10.min(types_after.len())]
    );

    println!("Shuffle test complete");
}

#[test]
fn test_load_performance() {
    let path = Path::new(DB_PATH);
    if !path.exists() {
        eprintln!("Skipping test: {} does not exist", DB_PATH);
        eprintln!("Run the preprocessor first to generate this file.");
        return;
    }

    // Get file size
    let file_size = std::fs::metadata(path)
        .expect("Failed to get file metadata")
        .len();
    let file_size_mb = file_size as f64 / (1024.0 * 1024.0);

    println!("\n=== Load Performance Test ===");
    println!("File: {}", DB_PATH);
    println!("File size: {:.2} MB", file_size_mb);

    // Warm up (first load might have cold cache effects)
    let _ = Database::load(path).expect("Failed to load database");

    // Benchmark multiple loads
    const NUM_ITERATIONS: usize = 5;
    let mut load_times = Vec::with_capacity(NUM_ITERATIONS);

    for i in 0..NUM_ITERATIONS {
        let start = Instant::now();
        let db = Database::load(path).expect("Failed to load database");
        let elapsed = start.elapsed();
        load_times.push(elapsed);

        // Prevent the compiler from optimizing away the load
        std::hint::black_box(&db);

        println!(
            "  Load {}: {:.3}s ({:.2} MB/s)",
            i + 1,
            elapsed.as_secs_f64(),
            file_size_mb / elapsed.as_secs_f64()
        );
    }

    // Calculate statistics
    let total_time: f64 = load_times.iter().map(|d| d.as_secs_f64()).sum();
    let avg_time = total_time / NUM_ITERATIONS as f64;
    let min_time = load_times
        .iter()
        .map(|d| d.as_secs_f64())
        .fold(f64::INFINITY, f64::min);
    let max_time = load_times
        .iter()
        .map(|d| d.as_secs_f64())
        .fold(0.0, f64::max);

    println!("\n=== Load Performance Summary ===");
    println!("  Iterations: {}", NUM_ITERATIONS);
    println!(
        "  Average: {:.3}s ({:.2} MB/s)",
        avg_time,
        file_size_mb / avg_time
    );
    println!(
        "  Min: {:.3}s ({:.2} MB/s)",
        min_time,
        file_size_mb / min_time
    );
    println!(
        "  Max: {:.3}s ({:.2} MB/s)",
        max_time,
        file_size_mb / max_time
    );

    // Basic sanity check - loading shouldn't take more than 30 seconds for reasonable file sizes
    assert!(
        avg_time < 30.0,
        "Average load time ({:.3}s) exceeded 30s threshold",
        avg_time
    );
}

#[test]
fn test_batch_generation_performance() {
    let path = Path::new(DB_PATH);
    if !path.exists() {
        eprintln!("Skipping test: {} does not exist", DB_PATH);
        eprintln!("Run the preprocessor first to generate this file.");
        return;
    }

    println!("\n=== Sampler Throughput Test ===\n");

    // Test configurations: (batch_size, seq_len, num_batches, description)
    let configs = [
        (1, 512, 500, "Single sample, short seq"),
        (8, 512, 200, "Small batch, short seq"),
        (32, 512, 100, "Medium batch, short seq"),
        (32, 1024, 100, "Medium batch, medium seq"),
        (32, 2048, 50, "Medium batch, long seq"),
        (64, 1024, 50, "Large batch, medium seq"),
        (128, 1024, 25, "XL batch, medium seq"),
    ];

    println!(
        "{:>6} {:>6} {:>8} {:>10} {:>12} {:>14} {:>12}",
        "batch", "seq", "batches", "time(s)", "batch/s", "sample/s", "Mcell/s"
    );
    println!("{}", "-".repeat(80));

    for (batch_size, seq_len, num_batches, desc) in configs {
        let config = SamplerConfig {
            batch_size,
            seq_len,
            max_bfs_width: 256,
            masking_strategy: MaskingStrategy::Random { mask_rate: 0.15 },
            seed: 42,
        };

        let sampler = Sampler::from_path(path, config).expect("Failed to create sampler");

        // Create reusable buffer
        let mut batch_buffer = sampler.create_batch_buffer();

        // Warm up (3 batches)
        for i in 0..3 {
            sampler.fill_batch_into(i, &mut batch_buffer);
        }

        // Benchmark with buffer reuse
        let start = Instant::now();
        for batch_idx in 0..num_batches {
            sampler.fill_batch_into(batch_idx + 3, &mut batch_buffer); // Skip warmup batches
            std::hint::black_box(&batch_buffer);
        }
        let elapsed = start.elapsed();

        let secs = elapsed.as_secs_f64();
        let batches_per_sec = num_batches as f64 / secs;
        let samples_per_sec = batches_per_sec * batch_size as f64;
        let cells_per_sec = samples_per_sec * seq_len as f64;
        let mcells_per_sec = cells_per_sec / 1_000_000.0;

        println!(
            "{:>6} {:>6} {:>8} {:>10.3} {:>12.1} {:>14.1} {:>12.2}",
            batch_size,
            seq_len,
            num_batches,
            secs,
            batches_per_sec,
            samples_per_sec,
            mcells_per_sec
        );

        // Sanity check
        assert!(
            batches_per_sec > 1.0,
            "{}: Throughput ({:.2} batch/s) is too low",
            desc,
            batches_per_sec
        );
    }

    println!("\n=== Sustained Throughput Test (10 seconds) ===\n");

    // Test sustained throughput for 10 seconds with typical training config
    let config = SamplerConfig {
        batch_size: 32,
        seq_len: 1024,
        max_bfs_width: 256,
        masking_strategy: MaskingStrategy::Random { mask_rate: 0.15 },
        seed: 42,
    };

    let mut sampler = Sampler::from_path(path, config).expect("Failed to create sampler");

    // Create reusable buffer
    let mut batch_buffer = sampler.create_batch_buffer();

    // Warm up
    for i in 0..5 {
        sampler.fill_batch_into(i, &mut batch_buffer);
    }

    let test_duration = std::time::Duration::from_secs(10);
    let start = Instant::now();
    let mut total_batches = 0usize;
    let mut batch_idx = 0usize;
    let mut epoch = 0u64;

    while start.elapsed() < test_duration {
        sampler.fill_batch_into(batch_idx, &mut batch_buffer);
        std::hint::black_box(&batch_buffer);
        total_batches += 1;
        batch_idx += 1;

        // Simulate epoch boundary every 1000 batches
        if batch_idx >= 1000 {
            epoch += 1;
            sampler.shuffle(epoch);
            batch_idx = 0;
        }
    }

    let elapsed = start.elapsed();
    let secs = elapsed.as_secs_f64();
    let batches_per_sec = total_batches as f64 / secs;
    let samples_per_sec = batches_per_sec * sampler.config().batch_size as f64;
    let cells_per_sec = samples_per_sec * sampler.config().seq_len as f64;

    println!(
        "Config: batch_size={}, seq_len={}",
        sampler.config().batch_size,
        sampler.config().seq_len
    );
    println!("Duration: {:.2}s", secs);
    println!("Total batches: {}", total_batches);
    println!("Epochs completed: {}", epoch);
    println!("Throughput:");
    println!("  {:.1} batches/sec", batches_per_sec);
    println!("  {:.1} samples/sec", samples_per_sec);
    println!("  {:.2} M cells/sec", cells_per_sec / 1_000_000.0);

    // For context: estimate how long an epoch would take
    let batches_per_epoch = sampler.num_batches();
    let estimated_epoch_time = batches_per_epoch as f64 / batches_per_sec;
    println!(
        "\nEstimated epoch time: {:.1} hours ({} batches)",
        estimated_epoch_time / 3600.0,
        batches_per_epoch
    );

    // Sanity check - should maintain reasonable throughput
    assert!(
        batches_per_sec > 10.0,
        "Sustained throughput ({:.2} batch/s) is too low",
        batches_per_sec
    );

    // Memory bandwidth analysis
    println!("\n=== Memory Bandwidth Analysis ===");
    let batch_size = sampler.config().batch_size;
    let seq_len = sampler.config().seq_len;
    let d_text = sampler.database().embed_dim as usize;

    // Calculate bytes written per batch
    let numerical_bytes = batch_size * seq_len * 4; // f32
    let categorical_bytes = batch_size * seq_len * d_text * 2; // f16
    let text_bytes = batch_size * seq_len * d_text * 2; // f16
    let timestamp_bytes = batch_size * seq_len * 11 * 4; // 11 f32s
    let colname_bytes = batch_size * seq_len * d_text * 2; // f16
    let semantic_bytes = batch_size * seq_len * 4; // i32
    let mask_bytes = batch_size * seq_len * 1; // bool
    let padding_bytes = batch_size * seq_len * 1; // bool

    // Bitpacked attention masks: 3 masks, each is batch_size * seq_len rows * ceil(seq_len/64) u64s
    let words_per_row = (seq_len + 63) / 64;
    let packed_attn_mask_bytes = batch_size * seq_len * words_per_row * 8 * 3; // 3 masks, 8 bytes per u64
    let unpacked_attn_mask_bytes = batch_size * seq_len * seq_len * 3; // for comparison

    let total_output_bytes = numerical_bytes
        + categorical_bytes
        + text_bytes
        + timestamp_bytes
        + colname_bytes
        + semantic_bytes
        + mask_bytes
        + padding_bytes
        + packed_attn_mask_bytes;

    let output_mb_per_batch = total_output_bytes as f64 / (1024.0 * 1024.0);
    let output_gb_per_sec = (output_mb_per_batch * batches_per_sec) / 1024.0;

    println!(
        "Output per batch: {:.2} MB (with bitpacked masks)",
        output_mb_per_batch
    );
    println!(
        "  - Embeddings (cat+text+colname): {:.2} MB",
        (categorical_bytes + text_bytes + colname_bytes) as f64 / (1024.0 * 1024.0)
    );
    println!(
        "  - Attention masks (packed): {:.2} MB (was {:.2} MB unpacked, {:.1}x smaller)",
        packed_attn_mask_bytes as f64 / (1024.0 * 1024.0),
        unpacked_attn_mask_bytes as f64 / (1024.0 * 1024.0),
        unpacked_attn_mask_bytes as f64 / packed_attn_mask_bytes as f64
    );
    println!(
        "  - Other (num+ts+types+masks): {:.2} MB",
        (numerical_bytes + timestamp_bytes + semantic_bytes + mask_bytes + padding_bytes) as f64
            / (1024.0 * 1024.0)
    );
    println!("Memory write throughput: {:.2} GB/s", output_gb_per_sec);
}

#[test]
fn test_profile_scaling() {
    let path = Path::new(DB_PATH);
    if !path.exists() {
        eprintln!("Skipping test: {} does not exist", DB_PATH);
        return;
    }

    println!("\n=== Profiling: Seq Length Scaling (batch_size=32) ===\n");
    println!(
        "{:>8} {:>10} {:>12} {:>12} {:>12}",
        "seq_len", "time(ms)", "cells/ms", "masks_MB", "masks_time%"
    );
    println!("{}", "-".repeat(60));

    // Test how time scales with seq_len (O(seq_len) vs O(seq_len^2))
    let seq_lens = [128, 256, 512, 1024, 2048];
    let batch_size = 32;
    let num_batches = 50;

    for seq_len in seq_lens {
        let config = SamplerConfig {
            batch_size,
            seq_len,
            max_bfs_width: 256,
            masking_strategy: MaskingStrategy::Random { mask_rate: 0.15 },
            seed: 42,
        };

        let sampler = Sampler::from_path(path, config).expect("Failed to create sampler");

        // Warmup
        for i in 0..3 {
            let _ = sampler.get_batch(i);
        }

        let start = Instant::now();
        for batch_idx in 0..num_batches {
            let batch = sampler.get_batch(batch_idx + 3);
            std::hint::black_box(&batch);
        }
        let elapsed = start.elapsed();

        let ms_per_batch = elapsed.as_secs_f64() * 1000.0 / num_batches as f64;
        let cells_per_batch = (batch_size * seq_len) as f64;
        let cells_per_ms = cells_per_batch / ms_per_batch;

        // Attention masks are O(seq_len^2)
        let mask_bytes = batch_size * seq_len * seq_len * 3;
        let mask_mb = mask_bytes as f64 / (1024.0 * 1024.0);

        // Estimate mask time based on O(seq_len^2) scaling
        // At seq_len=1024, masks are ~40% of time
        let base_seq = 1024.0;
        let base_mask_pct = 0.40;
        let scale = (seq_len as f64 / base_seq).powi(2);
        let estimated_mask_pct = (base_mask_pct * scale) / (1.0 + base_mask_pct * (scale - 1.0));

        println!(
            "{:>8} {:>10.2} {:>12.0} {:>12.1} {:>11.0}%",
            seq_len,
            ms_per_batch,
            cells_per_ms,
            mask_mb,
            estimated_mask_pct * 100.0
        );
    }

    println!("\n=== Profiling: Batch Size Scaling (seq_len=1024) ===\n");
    println!(
        "{:>8} {:>10} {:>12} {:>12}",
        "batch", "time(ms)", "samples/ms", "parallel_eff%"
    );
    println!("{}", "-".repeat(50));

    let batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128];
    let seq_len = 1024;

    // First measure single-sample baseline
    let config = SamplerConfig {
        batch_size: 1,
        seq_len,
        max_bfs_width: 256,
        masking_strategy: MaskingStrategy::Random { mask_rate: 0.15 },
        seed: 42,
    };
    let sampler = Sampler::from_path(path, config).expect("Failed");
    for i in 0..3 {
        let _ = sampler.get_batch(i);
    }
    let start = Instant::now();
    for batch_idx in 0..200 {
        let batch = sampler.get_batch(batch_idx + 3);
        std::hint::black_box(&batch);
    }
    let baseline_ms_per_sample = start.elapsed().as_secs_f64() * 1000.0 / 200.0;

    for batch_size in batch_sizes {
        let config = SamplerConfig {
            batch_size,
            seq_len,
            max_bfs_width: 256,
            masking_strategy: MaskingStrategy::Random { mask_rate: 0.15 },
            seed: 42,
        };

        let sampler = Sampler::from_path(path, config).expect("Failed to create sampler");

        // Warmup
        for i in 0..3 {
            let _ = sampler.get_batch(i);
        }

        let iters = (200 / batch_size).max(10);
        let start = Instant::now();
        for batch_idx in 0..iters {
            let batch = sampler.get_batch(batch_idx + 3);
            std::hint::black_box(&batch);
        }
        let elapsed = start.elapsed();

        let ms_per_batch = elapsed.as_secs_f64() * 1000.0 / iters as f64;
        let samples_per_ms = batch_size as f64 / ms_per_batch;

        // Parallel efficiency: ideal would be batch_size * baseline = actual
        let ideal_ms = baseline_ms_per_sample * batch_size as f64;
        let parallel_eff = (ideal_ms / ms_per_batch) * 100.0;

        println!(
            "{:>8} {:>10.2} {:>12.2} {:>11.0}%",
            batch_size, ms_per_batch, samples_per_ms, parallel_eff
        );
    }

    println!("\n=== Profiling: Component Timing Estimation ===\n");

    // Measure BFS-only cost by varying max_bfs_width
    println!("BFS width impact (batch=32, seq=1024):");
    println!("{:>12} {:>10} {:>12}", "max_width", "time(ms)", "delta%");
    println!("{}", "-".repeat(40));

    let widths = [16, 64, 256, 1024];
    let mut base_time = 0.0;

    for (i, width) in widths.iter().enumerate() {
        let config = SamplerConfig {
            batch_size: 32,
            seq_len: 1024,
            max_bfs_width: *width,
            masking_strategy: MaskingStrategy::Random { mask_rate: 0.15 },
            seed: 42,
        };

        let sampler = Sampler::from_path(path, config).expect("Failed");

        for j in 0..3 {
            let _ = sampler.get_batch(j);
        }

        let start = Instant::now();
        for batch_idx in 0..50 {
            let batch = sampler.get_batch(batch_idx + 3);
            std::hint::black_box(&batch);
        }
        let ms_per_batch = start.elapsed().as_secs_f64() * 1000.0 / 50.0;

        if i == 0 {
            base_time = ms_per_batch;
        }

        let delta = ((ms_per_batch - base_time) / base_time) * 100.0;
        println!("{:>12} {:>10.2} {:>+11.1}%", width, ms_per_batch, delta);
    }

    println!("\nNote: Small delta suggests BFS traversal is NOT the bottleneck.");
    println!("Large delta would suggest graph traversal costs dominate.\n");
}
