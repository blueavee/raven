#!/bin/bash

# Get the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
BENCH_DIR="$SCRIPT_DIR"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." &> /dev/null && pwd)"

echo "Running benchmarks from directory: $BENCH_DIR"

# First run OCaml benchmark
echo "Running OCaml benchmark..."
cd "$BENCH_DIR/ocaml_tokenizer"
OCAML_OUTPUT=$(OCAMLRUNPARAM=b dune exec ./bench_tokenizer.exe 2>/dev/null)
echo "$OCAML_OUTPUT" > "$BENCH_DIR/ocaml_results.txt"

# Then run Rust benchmark
echo "Running Rust benchmark..."
cd "$BENCH_DIR/rust_baseline"
RUST_OUTPUT=$(cargo bench 2>/dev/null)
echo "$RUST_OUTPUT" > "$BENCH_DIR/rust_results.txt"

# Print both results
echo -e "\nOCaml Results:"
echo "=============="
cat "$BENCH_DIR/ocaml_results.txt"

echo -e "\nRust Results:"
echo "============="
cat "$BENCH_DIR/rust_results.txt"

# Cleanup
rm -f "$BENCH_DIR/ocaml_results.txt" "$BENCH_DIR/rust_results.txt"
