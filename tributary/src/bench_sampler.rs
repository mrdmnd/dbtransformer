//! Benchmark binary for profiling the sampler.
//!
//! Usage:
//!   cargo build --release
//!   cargo flamegraph --bin bench_sampler -- /tmp/databases_preprocessed/rel-event.rkyv
//!
//! Note: mimalloc is used via the library's #[global_allocator]

use std::env;
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

use tributary::{Database, MaskingStrategy, Sampler, SamplerConfig};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <path-to-rkyv-file>", args[0]);
        eprintln!(
            "Example: {} /tmp/databases_preprocessed/rel-event.rkyv",
            args[0]
        );
        std::process::exit(1);
    }

    let db_path = Path::new(&args[1]);
    if !db_path.exists() {
        eprintln!("Error: File not found: {}", args[1]);
        std::process::exit(1);
    }

    println!("Loading database from: {}", args[1]);
    let start = Instant::now();
    let db = Database::load(db_path).expect("Failed to load database");
    println!(
        "Loaded in {:.2}s: {} rows, {} tables",
        start.elapsed().as_secs_f64(),
        db.num_rows(),
        db.num_tables()
    );

    // Create sampler with typical training config
    let config = SamplerConfig {
        batch_size: 32,
        seq_len: 1024,
        max_bfs_width: 256,
        masking_strategy: MaskingStrategy::Random { mask_rate: 0.15 },
        seed: 42,
    };

    println!(
        "Creating sampler: batch_size={}, seq_len={}",
        config.batch_size, config.seq_len
    );
    let mut sampler = Sampler::from_path(db_path, config).expect("Failed to create sampler");

    // Create reusable buffer - this avoids allocation per batch!
    let mut batch_buffer = sampler.create_batch_buffer();
    println!("Created reusable batch buffer");

    // Warmup
    println!("Warming up...");
    for i in 0..10 {
        sampler.fill_batch_into(i, &mut batch_buffer);
        black_box(&batch_buffer);
    }

    // Benchmark loop - run for ~20 seconds
    println!("Benchmarking for 20 seconds...");
    let duration = std::time::Duration::from_secs(20);
    let start = Instant::now();
    let mut total_batches = 0usize;
    let mut batch_idx = 10usize;
    let mut epoch = 0u64;

    while start.elapsed() < duration {
        sampler.fill_batch_into(batch_idx, &mut batch_buffer);
        black_box(&batch_buffer);
        total_batches += 1;
        batch_idx += 1;

        // Simulate epoch boundary
        if batch_idx >= 10000 {
            epoch += 1;
            sampler.shuffle(epoch);
            batch_idx = 0;
        }
    }

    let elapsed = start.elapsed();
    let batches_per_sec = total_batches as f64 / elapsed.as_secs_f64();
    let samples_per_sec = batches_per_sec * sampler.config().batch_size as f64;
    let cells_per_sec = samples_per_sec * sampler.config().seq_len as f64;

    println!("\n=== Results ===");
    println!("Total batches: {}", total_batches);
    println!("Throughput: {:.1} batches/sec", batches_per_sec);
    println!("Throughput: {:.1} samples/sec", samples_per_sec);
    println!("Throughput: {:.2} M cells/sec", cells_per_sec / 1_000_000.0);
}
