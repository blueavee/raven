open Ubench
open Saga

let batch_size = 1000

let create_bert_tokenizer () =
  let tokenizer = Tokenizer.create ~model:(Models.wordpiece ()) in
  (* Add special tokens like BERT *)
  let _ = Tokenizer.add_special_tokens tokenizer [
    Either.Right (Added_token.create ~content:"[UNK]" ~special:true ());
    Either.Right (Added_token.create ~content:"[SEP]" ~special:true ());
    Either.Right (Added_token.create ~content:"[PAD]" ~special:true ());
    Either.Right (Added_token.create ~content:"[CLS]" ~special:true ());
    Either.Right (Added_token.create ~content:"[MASK]" ~special:true ());
  ] in
  (* Set up normalizer and pre-tokenizer like BERT *)
  Tokenizer.set_normalizer tokenizer (Some (Normalizers.bert ()));
  Tokenizer.set_pre_tokenizer tokenizer (Some (Pre_tokenizers.bert ()));
  (* Set up post-processor like BERT *)
  let sep_id = Tokenizer.token_to_id tokenizer "[SEP]" |> Option.get in
  let cls_id = Tokenizer.token_to_id tokenizer "[CLS]" |> Option.get in
  Tokenizer.set_post_processor tokenizer (Some (Processors.bert ~sep:("[SEP]", sep_id) ~cls:("[CLS]", cls_id) ()));
  tokenizer

let bench_bert_encode () =
  let tokenizer = create_bert_tokenizer () in
  let data = In_channel.with_open_text "../../../saga/bench/data/test.txt" In_channel.input_all in
  let lines = String.split_on_char '\n' data |> List.filter (fun s -> String.length s > 0) in
  let lines = List.map (fun line -> Either.Left line) lines in

  let bench_encode_single () =
    List.iter (fun line ->
      let _ = Tokenizer.encode tokenizer ~sequence:line () in
      ()
    ) lines
  in

  let bench_encode_batch () =
    let rec process_batches remaining =
      match remaining with
      | [] -> ()
      | _ ->
          let batch_size = min batch_size (List.length remaining) in
          let batch = List.take batch_size remaining in
          let batch_input = List.map (fun x -> Either.Left x) batch in
          let _ = Tokenizer.encode_batch tokenizer ~input:batch_input () in
          process_batches (List.drop batch_size remaining)
    in
    process_batches lines
  in

  let suite = [
    bench "WordPiece BERT encode" bench_encode_single;
    bench "WordPiece BERT encode batch" bench_encode_batch;
  ] in

  let config = Config.default |> Config.time_limit 2.0 |> Config.warmup 3 |> Config.build in
  let results = run ~config suite in

  Printf.printf "\n=== BERT Tokenizer Benchmarks ===\n";
  List.iter (fun result ->
    Printf.printf "%s: %.2f ns/op\n" result.name result.time_stats.avg
  ) results


let () =
  bench_bert_encode ();
  (* bench_train_small (); *)
  (* bench_train_big () *)