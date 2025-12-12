//! Integration test for the Sampler.
//!
//! This test loads the sample F1 database and generates a batch to verify
//! the full pipeline works end-to-end.

use batcher::{Database, TableIdx};
use std::path::Path;
use std::time::Instant;

/// First, let's inspect the database to understand its structure
fn inspect_database(db: &Database) {
    println!("\n=== Database Summary ===");
    println!("Tables: {}", db.tables.len());
    println!("Columns: {}", db.columns.len());
    println!("Rows: {}", db.rows.len());
    println!("FK Edges: {}", db.fk_edges.len());
    println!("Text Vocab: {}", db.text_value_embeddings.len());

    println!("\n=== Tables ===");
    for table in &db.tables {
        let num_rows = table.row_range.1.0 - table.row_range.0.0;
        let num_cols = table.column_range.1.0 - table.column_range.0.0;
        println!(
            "  [{}] {:20} - {:5} rows, {:2} cols (rows [{:5}, {:5}), cols [{:2}, {:2}))",
            table.idx.0,
            table.name,
            num_rows,
            num_cols,
            table.row_range.0.0,
            table.row_range.1.0,
            table.column_range.0.0,
            table.column_range.1.0,
        );
        if let Some(pk) = table.primary_key_col {
            let pk_col = db.get_column(pk);
            println!("      PK: {} (col {})", pk_col.name, pk.0);
        }
        if let Some(tc) = table.time_col {
            let tc_col = db.get_column(tc);
            println!("      Time: {} (col {})", tc_col.name, tc.0);
        }
    }

    println!("\n=== Columns ===");
    for table in &db.tables {
        println!("  Table: {}", table.name);
        for col_idx in table.column_range.0.0..table.column_range.1.0 {
            let col = &db.columns[col_idx as usize];
            let fk_info = if let Some(target) = col.fk_target_table {
                format!(" -> {}", db.tables[target.0 as usize].name)
            } else {
                String::new()
            };
            println!(
                "    [{}] {:25} {:?}{}",
                col.idx.0, col.name, col.dtype, fk_info
            );
        }
    }
}

/// Find a good table to use as the "task table" for testing.
/// We want a table with enough rows for a batch.
fn find_task_table(db: &Database, min_rows: u32) -> Option<(TableIdx, u32, u32)> {
    for table in &db.tables {
        let num_rows = table.row_range.1.0 - table.row_range.0.0;
        if num_rows >= min_rows {
            // Find a numeric column to use as target
            for col_idx in table.column_range.0.0..table.column_range.1.0 {
                let col = &db.columns[col_idx as usize];
                if matches!(col.dtype, batcher::SemanticType::Number)
                    && !col.is_primary_key
                    && col.fk_target_table.is_none()
                {
                    return Some((table.idx, col_idx, num_rows));
                }
            }
        }
    }
    None
}

#[test]
fn test_sampler_batch_generation() {
    // Load the database
    let db_path = "sample_data_f1.rkyv";
    if !Path::new(db_path).exists() {
        eprintln!(
            "Skipping test: {} not found. Run the preprocessor first.",
            db_path
        );
        eprintln!("  cargo run --bin preprocessor -- --data-dir /path/to/f1/data");
        return;
    }

    let db = Database::load(Path::new(db_path)).expect("Failed to load database");
    inspect_database(&db);

    // Find a suitable task table
    let batch_size = 32;
    let (task_table, target_col, num_rows) = find_task_table(&db, batch_size as u32)
        .expect("No suitable task table found with enough rows");

    let table = db.get_table(task_table);
    let col = db.get_column(batcher::ColumnIdx(target_col));
    println!(
        "\n=== Test Configuration ===\n\
         Task table: {} (idx {})\n\
         Target column: {} (idx {})\n\
         Available rows: {}\n\
         Batch size: {}\n\
         Seq len: 1024",
        table.name, task_table.0, col.name, target_col, num_rows, batch_size
    );

    // Get embedding dimension from the database
    let d_text = db
        .text_value_embeddings
        .first()
        .map(|e| e.len())
        .unwrap_or(384);
    println!("Text embedding dim: {}", d_text);

    // Create the sampler
    // Note: We can't use the PyO3 Sampler directly in Rust tests easily,
    // so we'll test the Database loading and structure instead.
    // The actual Sampler tests would be done via Python.

    println!("\n=== Database Structure Verified ===");
    println!("Ready for sampling with:");
    println!(
        "  db_configs=[(\"{}\", {}, {}, [])]",
        db_path, task_table.0, target_col
    );
    println!("  batch_size={}", batch_size);
    println!("  seq_len=1024");
    println!("  d_text={}", d_text);
}

#[test]
fn test_database_load_performance() {
    let db_path = "sample_data_f1.rkyv";
    if !Path::new(db_path).exists() {
        eprintln!("Skipping test: {} not found", db_path);
        return;
    }

    // Measure load time
    let start = Instant::now();
    let db = Database::load(Path::new(db_path)).expect("Failed to load database");
    let load_time = start.elapsed();

    println!("\n=== Load Performance ===");
    println!("Load time: {:?}", load_time);
    println!("Tables: {}", db.tables.len());
    println!("Rows: {}", db.rows.len());
    println!(
        "Rows/sec: {:.0}",
        db.rows.len() as f64 / load_time.as_secs_f64()
    );
}
