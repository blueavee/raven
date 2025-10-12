use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::fs::File;
use std::hint::black_box;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};
use tokenizers::models::wordpiece::WordPiece;
use tokenizers::normalizers::BertNormalizer;
use tokenizers::pre_tokenizers::bert::BertPreTokenizer;

fn calculate_percentile(mut latencies: Vec<f64>, percentile: f64) -> f64 {
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let index = (latencies.len() as f64 * percentile / 100.0) as usize;
    latencies[index]
}
use tokenizers::processors::bert::BertProcessing;
use tokenizers::{Decoder, EncodeInput, Normalizer, PostProcessor, PreTokenizer, decoders};
use tokenizers::{Model, TokenizerImpl};

type BertTokenizer = TokenizerImpl<
    WordPiece,
    BertNormalizer,
    BertPreTokenizer,
    BertProcessing,
    decoders::wordpiece::WordPiece,
>;
static BATCH_SIZE: usize = 1_000;

pub fn iter_bench_encode<M, N, PT, PP, D>(
    iters: u64,
    tokenizer: &TokenizerImpl<M, N, PT, PP, D>,
    lines: &[EncodeInput],
) -> (Duration, Vec<f64>, usize)
// Returns (total_duration, individual_latencies, total_tokens)
where
    M: Model,
    N: Normalizer,
    PT: PreTokenizer,
    PP: PostProcessor,
    D: Decoder,
{
    let mut duration = Duration::new(0, 0);
    let mut latencies = Vec::with_capacity((iters as usize) * lines.len());
    let mut total_tokens = 0;

    for _i in 0..iters {
        for line in lines {
            let input = line.clone();
            let start = Instant::now();
            let result = black_box(tokenizer.encode(input, false));
            let latency = start.elapsed();
            latencies.push(latency.as_secs_f64() * 1000.0); // Convert to milliseconds
            duration = duration.checked_add(latency).unwrap();

            // Count tokens
            if let Ok(encoding) = result {
                total_tokens += encoding.get_tokens().len();
            }
        }
    }
    (duration, latencies, total_tokens)
}
pub fn iter_bench_encode_batch<M, N, PT, PP, D>(
    iters: u64,
    tokenizer: &TokenizerImpl<M, N, PT, PP, D>,
    batches: &[Vec<EncodeInput>],
) -> (Duration, Vec<f64>, usize)
// Returns (total_duration, individual_latencies, total_tokens)
where
    M: Model + Send + Sync,
    N: Normalizer + Send + Sync,
    PT: PreTokenizer + Send + Sync,
    PP: PostProcessor + Send + Sync,
    D: Decoder + Send + Sync,
{
    let mut duration = Duration::new(0, 0);
    let mut latencies = Vec::with_capacity((iters as usize) * batches.len());
    let mut total_tokens = 0;

    for _i in 0..iters {
        for batch in batches {
            let batch = batch.clone();
            let start = Instant::now();
            let result = black_box(tokenizer.encode_batch(batch, false));
            let latency = start.elapsed();
            latencies.push(latency.as_secs_f64() * 1000.0); // Convert to milliseconds
            duration = duration.checked_add(latency).unwrap();

            // Count tokens
            if let Ok(encodings) = result {
                total_tokens += encodings
                    .iter()
                    .map(|e| e.get_tokens().len())
                    .sum::<usize>();
            }
        }
    }
    (duration, latencies, total_tokens)
}

fn create_bert_tokenizer(wp: WordPiece) -> BertTokenizer {
    let sep_id = *wp.get_vocab().get("[SEP]").unwrap();
    let cls_id = *wp.get_vocab().get("[CLS]").unwrap();
    let mut tokenizer = TokenizerImpl::new(wp);
    tokenizer.with_pre_tokenizer(Some(BertPreTokenizer));
    tokenizer.with_normalizer(Some(BertNormalizer::default()));
    tokenizer.with_decoder(Some(decoders::wordpiece::WordPiece::default()));
    tokenizer.with_post_processor(Some(BertProcessing::new(
        ("[SEP]".into(), sep_id),
        ("[CLS]".into(), cls_id),
    )));
    tokenizer
}

pub fn bench_bert(c: &mut Criterion) {
    let wp = WordPiece::from_file("data/bert-base-uncased-vocab.txt")
        .build()
        .unwrap();
    let tokenizer = create_bert_tokenizer(wp);
    let mut group = c.benchmark_group("bert-encode");
    let data = std::fs::read_to_string("data/big.txt").unwrap();
    let data_size_mb = data.len() as f64 / (1024.0 * 1024.0);
    group.throughput(Throughput::Bytes(data.len() as u64));

    let mut lines: Vec<EncodeInput> = vec![];
    let mut batches: Vec<Vec<EncodeInput>> = vec![vec![]];
    for line in BufReader::new(File::open(Path::new("data/big.txt")).unwrap()).lines() {
        let line: EncodeInput = line.unwrap().into();
        lines.push(line.clone());
        if batches.last().unwrap().len() >= BATCH_SIZE {
            batches.push(vec![]);
        }
        batches.last_mut().unwrap().push(line);
    }

    // Collect metrics for single encode
    let mut single_metrics = None;
    group.bench_function("WordPiece BERT encode", |b| {
        b.iter_custom(|iters| {
            let (duration, latencies, total_tokens) = iter_bench_encode(iters, &tokenizer, &lines);
            let total_time_secs = duration.as_secs_f64();
            single_metrics = Some((
                total_tokens as f64 / total_time_secs,
                data_size_mb / total_time_secs,
                calculate_percentile(latencies.clone(), 50.0),
                calculate_percentile(latencies.clone(), 95.0),
                calculate_percentile(latencies, 99.0),
            ));
            duration
        })
    });

    // Collect metrics for batch encode
    let mut batch_metrics = None;
    group.bench_function("WordPiece BERT encode batch", |b| {
        b.iter_custom(|iters| {
            let (duration, latencies, total_tokens) =
                iter_bench_encode_batch(iters, &tokenizer, &batches);
            let total_time_secs = duration.as_secs_f64();
            batch_metrics = Some((
                total_tokens as f64 / total_time_secs,
                data_size_mb / total_time_secs,
                calculate_percentile(latencies.clone(), 50.0),
                calculate_percentile(latencies.clone(), 95.0),
                calculate_percentile(latencies, 99.0),
            ));
            duration
        })
    });

    // Get peak RSS using ps command
    let peak_rss = Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .and_then(|s| s.trim().parse::<f64>().ok())
        .map(|rss| rss / 1024.0) // Convert KB to MB
        .unwrap_or(0.0);

    // Print all metrics at the end
    println!("\nFinal Benchmark Results");
    println!("=====================");

    if let Some((tokens_per_sec, mb_per_sec, p50, p95, p99)) = single_metrics {
        println!("\nSingle Encode Metrics:");
        println!("---------------------");
        println!("Throughput:");
        println!("  - Tokens/second: {:.2}", tokens_per_sec);
        println!("  - MB/second: {:.2}", mb_per_sec);
        println!("Latency (ms):");
        println!("  - p50: {:.2}", p50);
        println!("  - p95: {:.2}", p95);
        println!("  - p99: {:.2}", p99);
    }

    if let Some((tokens_per_sec, mb_per_sec, p50, p95, p99)) = batch_metrics {
        println!("\nBatch Encode Metrics:");
        println!("---------------------");
        println!("Throughput:");
        println!("  - Tokens/second: {:.2}", tokens_per_sec);
        println!("  - MB/second: {:.2}", mb_per_sec);
        println!("Latency (ms):");
        println!("  - p50: {:.2}", p50);
        println!("  - p95: {:.2}", p95);
        println!("  - p99: {:.2}", p99);
    }

    println!("\nMemory Usage:");
    println!("-------------");
    println!("Peak RSS: {:.2} MB", peak_rss);
}

criterion_group! {
    name = bert_benches;
    config = Criterion::default().sample_size(20).measurement_time(Duration::from_secs(15));
    targets = bench_bert
}
criterion_main!(bert_benches);
