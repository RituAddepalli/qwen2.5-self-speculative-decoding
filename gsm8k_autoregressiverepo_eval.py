#!/usr/bin/env python3

import os
import re
import json
import time
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2TokenizerFast
from datasets import load_dataset


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "/home/ziroh/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

DATASET_NAME = "openai/gsm8k"
DATASET_CONFIG = "main"
DATASET_SPLIT = "test"

MAX_GEN_TOKS = 1024
WARMUP_EXAMPLES = 3

TORCH_THREADS = 96
TORCH_INTEROP_THREADS = 96

DEVICE = "cpu"
DTYPE = torch.bfloat16

OUTPUT_FILE = os.path.join(
    "/home/ziroh/ritu/ritu-ssd/",
    "gsm8k_autoregressiverepo_log_full_dataset.json",
)

LOG_FILE = os.path.join(
    "/home/ziroh/ritu/ritu-ssd/",
    "gsm8k_autoregressiverepo_log_full_dataset.jsonl",
)


# ============================================================
# THREAD CONFIGURATION
# ============================================================

torch.set_num_threads(TORCH_THREADS)
torch.set_num_interop_threads(TORCH_INTEROP_THREADS)


# ============================================================
# GSM8K ANSWER EXTRACTION
# ============================================================

def extract_gsm8k_answer(text):
    """
    Extract the final GSM8K answer.

    Expected model format:
        #### <number>
    """

    matches = re.findall(r"####\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)", text)

    if matches:
        return matches[-1].replace(",", "").strip()

    # Fallback: last number in generated text
    numbers = re.findall(
        r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?",
        text
    )

    if numbers:
        return numbers[-1].replace(",", "").strip()

    return None


def normalize_answer(answer):
    if answer is None:
        return None

    answer = str(answer).strip()
    answer = answer.replace(",", "")

    try:
        value = float(answer)

        if value.is_integer():
            return str(int(value))

        return str(value)

    except Exception:
        return answer


# ============================================================
# OFFICIAL-STYLE AUTOREGRESSIVE BASELINE
# ============================================================

@torch.no_grad()
def base_generate(
    model,
    input_ids,
    max_new_tokens,
    eos_token_id=None,
):
    """
    Manual autoregressive generation following the structure
    of the official self-speculative-decoding repository's
    base_generate().

    This is intentionally NOT model.generate().
    """

    batch_size = input_ids.shape[0]

    current_input_ids = input_ids
    past_key_values = None

    # Preallocate generated tokens.
    generate_ids = torch.empty(
        (batch_size, max_new_tokens),
        dtype=torch.long,
        device=input_ids.device,
    )

    generated_count = 0

    for step in range(max_new_tokens):

        output = model(
            input_ids=current_input_ids,
            past_key_values=past_key_values,
            return_dict=True,
            use_cache=True,
        )

        # Last-token logits.
        logits = output.logits[:, -1, :]

        # Greedy decoding.
        output_ids = torch.argmax(
            logits,
            dim=-1,
        )

        generate_ids[:, step] = output_ids

        generated_count += 1

        # Update cache.
        past_key_values = output.past_key_values

        # Next iteration only needs the newly generated token.
        current_input_ids = output_ids.unsqueeze(-1)

        # Optional EOS stopping.
        if eos_token_id is not None:
            if torch.all(output_ids == eos_token_id):
                break

    generate_ids = generate_ids[:, :generated_count]

    return generate_ids


# ============================================================
# PROMPT
# ============================================================

def build_prompt(question):
    return (
        "Solve the following math word problem carefully. "
        "Show your reasoning step by step and give the final answer "
        "on a separate line in the exact format:\n"
        "#### <number>\n\n"
        f"Question:\n{question}"
    )


# ============================================================
# MAIN
# ============================================================
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of GSM8K examples to evaluate. Default: all 1319 examples.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_FILE,
    )

    parser.add_argument(
        "--log",
        type=str,
        default=LOG_FILE,
    )

    args = parser.parse_args()

    with open(args.log, "w", encoding="utf-8") as f:
        f.write("")

    print("=" * 70)
    print("GSM8K FULL AUTOREGRESSIVE BASELINE")
    print("=" * 70)

    print(f"Model       : {MODEL_NAME}")
    print(f"Device      : {DEVICE}")
    print(f"Dtype       : {DTYPE}")
    print(f"Torch threads: {TORCH_THREADS}")
    print(f"Max tokens  : {MAX_GEN_TOKS}")
    print(f"Log         : {args.log}")

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = Qwen2TokenizerFast.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=DTYPE,
        device_map=None,
        local_files_only=True,
        trust_remote_code=True,
    )

    model = model.to(DEVICE)
    model.eval()    

    print("Model loaded.")

    # --------------------------------------------------------
    # EOS token
    # --------------------------------------------------------

    eos_token_id = tokenizer.eos_token_id

    print(f"EOS token ID: {eos_token_id}")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print("\nLoading GSM8K...")

    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split=DATASET_SPLIT,
    )

    if args.limit is not None:
        dataset = dataset.select(
            range(min(args.limit, len(dataset)))
        )

    print(f"Examples: {len(dataset)}")

    # ========================================================
    # WARMUP
    # ========================================================

    print("\n" + "=" * 70)
    print("WARMUP")
    print("=" * 70)

    for i in range(min(WARMUP_EXAMPLES, len(dataset))):

        question = dataset[i]["question"]

        prompt = build_prompt(question)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are Qwen, created by Alibaba Cloud. "
                    "You are a helpful assistant."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        input_ids = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(DEVICE)

        _ = base_generate(
            model=model,
            input_ids=input_ids,
            max_new_tokens=MAX_GEN_TOKS,
            eos_token_id=eos_token_id,
        )

        print(f"Warmup {i + 1}/{min(WARMUP_EXAMPLES, len(dataset))}")

    # ========================================================
    # EVALUATION
    # ========================================================

    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    total_latency = 0.0
    total_generated_tokens = 0

    correct_count = 0
    processed_count = 0

    all_records = []

    for idx, example in enumerate(dataset):

        question = example["question"]
        target = normalize_answer(
            extract_gsm8k_answer(
                example["answer"]
            )
        )

        # ----------------------------------------------------
        # Build EXACT same prompt used by SSD
        # ----------------------------------------------------

        prompt = build_prompt(question)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are Qwen, created by Alibaba Cloud. "
                    "You are a helpful assistant."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenization is intentionally OUTSIDE the timer.
        input_ids = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(DEVICE)

        prompt_tokens = input_ids.shape[1]

        # ----------------------------------------------------
        # TIMED GENERATION
        # ----------------------------------------------------

        start_time = time.perf_counter()

        output_ids = base_generate(
            model=model,
            input_ids=input_ids,
            max_new_tokens=MAX_GEN_TOKS,
            eos_token_id=eos_token_id,
        )

        end_time = time.perf_counter()

        latency = end_time - start_time

        # ----------------------------------------------------
        # Decode OUTSIDE timer
        # ----------------------------------------------------

        generated_token_ids = output_ids[0].tolist()

        generated_text = tokenizer.decode(
            generated_token_ids,
            skip_special_tokens=True,
        )

        prediction = normalize_answer(
            extract_gsm8k_answer(
                generated_text
            )
        )

        is_correct = (
            prediction is not None
            and target is not None
            and prediction == target
        )

        generated_tokens = len(generated_token_ids)

        tokens_per_sec = (
            generated_tokens / latency
            if latency > 0
            else 0.0
        )

        # ----------------------------------------------------
        # Accumulate
        # ----------------------------------------------------

        total_latency += latency
        total_generated_tokens += generated_tokens

        if is_correct:
            correct_count += 1

        processed_count += 1

        # ----------------------------------------------------
        # Record
        # ----------------------------------------------------

        record = {
            "index": idx,
            "id": example.get("id"),

            "target": target,
            "prediction": prediction,
            "correct": is_correct,

            "latency": latency,
            "generated_tokens": generated_tokens,
            "tokens_per_sec": tokens_per_sec,

            "generated_token_ids": generated_token_ids,

            "prompt_tokens": prompt_tokens,

            "question": question,

            "raw_generation": generated_text,
        }

        all_records.append(record)

        # ----------------------------------------------------
        # JSONL
        # ----------------------------------------------------

        with open(args.log, "a") as f:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        print(
            f"[{idx + 1}/{len(dataset)}] "
            f"{'✓' if is_correct else '✗'} "
            f"latency={latency:.3f}s "
            f"tokens={generated_tokens} "
            f"tok/s={tokens_per_sec:.3f} "
            f"pred={prediction} "
            f"target={target}"
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    avg_latency = (
        total_latency / processed_count
        if processed_count
        else 0.0
    )

    overall_tokens_per_sec = (
        total_generated_tokens / total_latency
        if total_latency > 0
        else 0.0
    )

    accuracy = (
        correct_count / processed_count
        if processed_count
        else 0.0
    )

    summary = {
        "model": MODEL_NAME,

        "dataset": DATASET_NAME,
        "dataset_config": DATASET_CONFIG,
        "dataset_split": DATASET_SPLIT,

        "device": DEVICE,
        "dtype": str(DTYPE),

        "torch_threads": TORCH_THREADS,
        "torch_interop_threads": TORCH_INTEROP_THREADS,

        "max_new_tokens": MAX_GEN_TOKS,

        "examples": processed_count,
        "correct": correct_count,
        "accuracy": accuracy,

        "total_latency": total_latency,
        "avg_latency_per_example": avg_latency,

        "generated_tokens": total_generated_tokens,
        "tokens_per_sec": overall_tokens_per_sec,

        "generation_method": (
            "manual autoregressive generation "
            "with past_key_values and greedy argmax"
        ),

        "uses_hf_model_generate": False,

        "repetition_penalty": 1.0,
    }

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary_path = os.path.splitext(args.log)[0] + "_summary.json"

    with open(summary_path, "w") as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL BASELINE RESULTS")
    print("=" * 70)

    print(f"Examples              : {processed_count}")
    print(f"Correct               : {correct_count}")
    print(f"Accuracy              : {accuracy * 100:.2f}%")

    print(f"Total latency         : {total_latency:.3f} sec")
    print(
        f"Avg latency/example   : "
        f"{avg_latency:.3f} sec"
    )

    print(
        f"Generated tokens      : "
        f"{total_generated_tokens}"
    )

    print(
        f"Tokens/sec            : "
        f"{overall_tokens_per_sec:.3f}"
    )

    print(f"\nJSONL log             : {args.log}")
    print(f"Summary               : {summary_path}")

    print("=" * 70)


if __name__ == "__main__":
    main()



#             numactl --physcpubind=0-95 --membind=0 \
#  python3 /home/ziroh/ritu/ritu-ssd/gsm8k_autoregressiverepo_eval.py --limit 50