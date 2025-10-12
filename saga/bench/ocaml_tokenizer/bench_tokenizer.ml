module M = Saga_tokenizers.Models
module W = Saga_tokenizers.Wordpiece

let batch_size = 1_000

let calculate_percentile latencies percentile =
  let sorted = Array.copy latencies in
  Array.sort Float.compare sorted;
  let index = int_of_float (float_of_int (Array.length sorted) *. percentile /. 100.0) in
  sorted.(index)

let get_peak_rss () =
  let cmd = Printf.sprintf "ps -o rss= -p %d" (Unix.getpid ()) in
  let ic = Unix.open_process_in cmd in
  let rss = float_of_string (input_line ic) /. 1024.0 in (* Convert KB to MB *)
  let _ = Unix.close_process_in ic in
  rss

let read_lines file =
  let ic = open_in file in
  let rec read_all acc =
    try
      let line = input_line ic in
      read_all (line :: acc)
    with End_of_file -> List.rev acc
  in
  let lines = read_all [] in
  close_in ic;
  lines

let make_batches lines batch_size =
  let rec split acc current = function
    | [] -> if current <> [] then List.rev (List.rev current :: acc) else List.rev acc
    | x :: xs when List.length current >= batch_size -> split (List.rev current :: acc) [x] xs
    | x :: xs -> split acc (x :: current) xs
  in
  split [] [] lines

let bench_encode_single tokenizer lines n =
  let total_tokens = ref 0 in
  let latencies = Array.make (n * List.length lines) 0.0 in
  let idx = ref 0 in
  
  let start_total = Unix.gettimeofday () in
  for _i = 0 to n - 1 do
    List.iter (fun line ->
      let start_time = Unix.gettimeofday () in
      let tokens = W.tokenize tokenizer line in
      let end_time = Unix.gettimeofday () in
      latencies.(!idx) <- (end_time -. start_time) *. 1000.0; (* Convert to ms *)
      total_tokens := !total_tokens + List.length tokens;
      incr idx
    ) lines
  done;
  let total_time = (Unix.gettimeofday () -. start_total) *. 1000.0 in (* Convert to ms *)

  let total_time_secs = total_time /. 1000.0 in (* Convert ms to seconds *)
  let data_size_mb = 
    let total_chars = List.fold_left (fun acc s -> acc + String.length s) 0 lines in
    float_of_int total_chars /. (1024.0 *. 1024.0)
  in

  (* Calculate metrics *)
  let tokens_per_sec = float_of_int !total_tokens /. total_time_secs in
  let mb_per_sec = data_size_mb /. total_time_secs in
  let p50 = calculate_percentile latencies 50.0 in
  let p95 = calculate_percentile latencies 95.0 in
  let p99 = calculate_percentile latencies 99.0 in

  (tokens_per_sec, mb_per_sec, p50, p95, p99)

let bench_encode_batch tokenizer batches n =
  let total_tokens = ref 0 in
  let latencies = Array.make (n * List.length batches) 0.0 in
  let idx = ref 0 in
  
  let start_total = Unix.gettimeofday () in
  for _i = 0 to n - 1 do
    List.iter (fun batch ->
      let start_time = Unix.gettimeofday () in
      let token_lists = List.map (W.tokenize tokenizer) batch in
      let end_time = Unix.gettimeofday () in
      latencies.(!idx) <- (end_time -. start_time) *. 1000.0;
      total_tokens := !total_tokens + List.fold_left (fun acc l -> acc + List.length l) 0 token_lists;
      incr idx
    ) batches
  done;
  let total_time = (Unix.gettimeofday () -. start_total) *. 1000.0 in (* Convert to ms *)

  let total_time_secs = total_time /. 1000.0 in
  let data_size_mb = 
    let total_chars = List.fold_left (fun acc batch -> 
      acc + List.fold_left (fun acc s -> acc + String.length s) 0 batch
    ) 0 batches in
    float_of_int total_chars /. (1024.0 *. 1024.0)
  in

  (* Calculate metrics *)
  let tokens_per_sec = float_of_int !total_tokens /. total_time_secs in
  let mb_per_sec = data_size_mb /. total_time_secs in
  let p50 = calculate_percentile latencies 50.0 in
  let p95 = calculate_percentile latencies 95.0 in
  let p99 = calculate_percentile latencies 99.0 in

  (tokens_per_sec, mb_per_sec, p50, p95, p99)

let () =
  (* Create the tokenizer *)
  let tokenizer = W.from_file 
    ~vocab_file:"../rust_baseline/data/bert-base-uncased-vocab.txt" in

  (* Read input data *)
  let lines = read_lines "../rust_baseline/data/big.txt" in
  let batches = make_batches lines batch_size in
  
  (* Number of iterations *)
  let n = 20 in

  Printf.printf "\nFinal Benchmark Results\n";
  Printf.printf "=====================\n";

  (* Single encode benchmark *)
  let (tokens_per_sec, mb_per_sec, p50, p95, p99) = bench_encode_single tokenizer lines n in
  Printf.printf "\nSingle Encode Metrics:\n";
  Printf.printf "---------------------\n";
  Printf.printf "Throughput:\n";
  Printf.printf "  - Tokens/second: %.2f\n" tokens_per_sec;
  Printf.printf "  - MB/second: %.2f\n" mb_per_sec;
  Printf.printf "Latency (ms):\n";
  Printf.printf "  - p50: %.2f\n" p50;
  Printf.printf "  - p95: %.2f\n" p95;
  Printf.printf "  - p99: %.2f\n" p99;

  (* Batch encode benchmark *)
  let (tokens_per_sec, mb_per_sec, p50, p95, p99) = bench_encode_batch tokenizer batches n in
  Printf.printf "\nBatch Encode Metrics:\n";
  Printf.printf "---------------------\n";
  Printf.printf "Throughput:\n";
  Printf.printf "  - Tokens/second: %.2f\n" tokens_per_sec;
  Printf.printf "  - MB/second: %.2f\n" mb_per_sec;
  Printf.printf "Latency (ms):\n";
  Printf.printf "  - p50: %.2f\n" p50;
  Printf.printf "  - p95: %.2f\n" p95;
  Printf.printf "  - p99: %.2f\n" p99;

  (* Memory usage *)
  let peak_rss = get_peak_rss () in
  Printf.printf "\nMemory Usage:\n";
  Printf.printf "-------------\n";
  Printf.printf "Peak RSS: %.2f MB\n" peak_rss
