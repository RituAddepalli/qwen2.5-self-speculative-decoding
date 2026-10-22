import argparse
import json
import os
import re
import time
from datetime import datetime

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from decoding import ssd_generate_tokens


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"

DEVICE = "cpu"
DTYPE = torch.bfloat16

# ------------------------------------------------------------
# Benchmark configuration
# ------------------------------------------------------------

DEFAULT_LIMIT = 1319

MAX_GEN_TOKS = 1024

MAX_DRAFT_TOKENS = 8

TORCH_THREADS = 96
TORCH_INTEROP_THREADS = 96

# ------------------------------------------------------------
# Your selected reasoning skip pattern
# ------------------------------------------------------------
MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"

SKIP_PATTERN_FILE = (
   "/home/ziroh/ritu/ritu-ssd/skip_layers_qwen7b_reasoning_1000.json"
)


OUTPUT_FILE = (
    "/home/ziroh/ritu/ritu-ssd/"  
    "gsm8k_ssd_results_fulldataset.json"
)

LOG_FILE = os.path.join(
    "/home/ziroh/ritu/ritu-ssd/"
    "gsm8k_ssd_log_fulldataset.jsonl",
)

SYSTEM_MESSAGE = (
    "You are Qwen, created by Alibaba Cloud. "
    "You are a helpful assistant."
)


# ============================================================
# SSD PARAMETERS
#
# These correspond to the adaptive-threshold configuration
# used by the official exact self-speculative decoder:
#
# max_step_draft = 8
# th_stop_draft = 0.8
# auto_parameters =
#     [1, 0.5, 0.9, 1e-2, 0.9]
# ============================================================

GAMMA_KWARGS = {

    "initial_gamma": 0.8,

    "alpha": 0.9,

    "beta1": 0.5,

    "beta2": 0.9,

    "epsilon": 0.02,
}


# ============================================================
# CPU CONFIGURATION
# ============================================================

torch.set_num_threads(
    TORCH_THREADS
)

torch.set_num_interop_threads(
    TORCH_INTEROP_THREADS
)


# ============================================================
# PROMPT
#
# MUST remain identical to Full Greedy.
# ============================================================

def build_gsm8k_prompt(
    example,
    tokenizer,
):

    question = example["question"].strip()

    user_message = (
        "Solve the following math word problem carefully. "
        "Show your reasoning step by step and give the final "
        "answer on a separate line in the exact format:\n"
        "#### <number>\n\n"
        f"Question:\n{question}"
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_MESSAGE,
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ============================================================
# NUMBER NORMALIZATION
#
# IDENTICAL TO FULL GREEDY.
# ============================================================

def normalize_number(value):

    if value is None:
        return None

    value = str(value).strip()

    value = value.replace(",", "")
    value = value.replace("$", "")
    value = value.rstrip(". ")

    try:

        number = float(value)

        if number.is_integer():
            return str(int(number))

        return str(number)

    except ValueError:

        return value.lower().strip()


# ============================================================
# GSM8K GOLD ANSWER
#
# IDENTICAL TO FULL GREEDY.
# ============================================================

def get_gsm8k_target(example):

    answer = str(
        example["answer"]
    ).strip()

    matches = re.findall(
        r"####\s*"
        r"([-+]?\$?\d+(?:,\d{3})*(?:\.\d+)?)",
        answer,
    )

    if matches:

        return normalize_number(
            matches[-1]
        )

    numbers = re.findall(
        r"[-+]?\$?\d+(?:,\d{3})*(?:\.\d+)?",
        answer,
    )

    if numbers:

        return normalize_number(
            numbers[-1]
        )

    return None


# ============================================================
# GSM8K ANSWER EXTRACTION
#
# IDENTICAL TO FULL GREEDY.
# ============================================================

def extract_final_answer(text):

    if not text:
        return None

    text = text.replace(
        "\r\n",
        "\n",
    ).strip()

    # --------------------------------------------------------
    # 1. \boxed{42}
    # --------------------------------------------------------

    matches = re.findall(
        r"\\boxed\s*\{\s*"
        r"([-+]?\$?\d+(?:,\d{3})*(?:\.\d+)?)"
        r"\s*\}",
        text,
        flags=re.IGNORECASE,
    )

    if matches:

        return normalize_number(
            matches[-1]
        )

    # --------------------------------------------------------
    # 2. #### 42
    # --------------------------------------------------------

    matches = re.findall(
        r"####\s*"
        r"([-+]?\$?\d+(?:,\d{3})*(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )

    if matches:

        return normalize_number(
            matches[-1]
        )

    # --------------------------------------------------------
    # 3. Final answer is 42
    # --------------------------------------------------------

    matches = re.findall(
        r"(?:the\s+)?final\s+answer\s+is\s*"
        r"[:\-]?\s*"
        r"(?:\*\*)?"
        r"([-+]?\$?\d+(?:,\d{3})*(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )

    if matches:

        return normalize_number(
            matches[-1]
        )

    # --------------------------------------------------------
    # 4. Answer is 42
    # --------------------------------------------------------

    matches = re.findall(
        r"(?:the\s+)?answer\s+is\s*"
        r"[:\-]?\s*"
        r"(?:\*\*)?"
        r"([-+]?\$?\d+(?:,\d{3})*(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )

    if matches:

        return normalize_number(
            matches[-1]
        )

    # --------------------------------------------------------
    # 5. Final numerical value
    # --------------------------------------------------------

    numbers = re.findall(
        r"(?<![\w.])"
        r"[-+]?\$?\d+(?:,\d{3})*(?:\.\d+)?",
        text,
    )

    if numbers:

        return normalize_number(
            numbers[-1]
        )

    return None


# ============================================================
# SCORE
#
# IDENTICAL TO FULL GREEDY.
# ============================================================

def score_answer(
    generated_text,
    target,
):

    prediction = extract_final_answer(
        generated_text
    )

    if prediction is None:

        return None, False

    correct = (
        prediction
        == normalize_number(target)
    )

    return prediction, correct


# ============================================================
# APPLY SSD SKIP PATTERN
# ============================================================

def apply_skip_pattern(
    model,
    pattern,
):

    attn_set = set(
        pattern.get(
            "attn",
            [],
        )
    )

    mlp_set = set(
        pattern.get(
            "mlp",
            [],
        )
    )

    model.config.skip_layer_ids = {

        "attn":
            sorted(attn_set),

        "mlp":
            sorted(mlp_set),
    }

    for i, layer in enumerate(
        model.model.layers
    ):

        # ----------------------------------------------------
        # Layer 0 remains full.
        #
        # This is required by your Qwen DynamicCache
        # adaptation because layer 0 acts as the cache/mask
        # sequence-length anchor.
        # ----------------------------------------------------

        if i == 0:

            layer.skip_attn = False
            layer.skip_mlp = False

        else:

            layer.skip_attn = (
                i in attn_set
            )

            layer.skip_mlp = (
                i in mlp_set
            )


# ============================================================
# SSD GENERATION
#
# IMPORTANT:
#
# The actual Draft & Verify algorithm lives in decoding.py.
#
# This wrapper only measures the end-to-end generation call.
# ============================================================

@torch.no_grad()
def ssd_generate(
    model,
    tokenizer,
    input_ids,
):

    start = time.perf_counter()

    output_ids, stats = (
        ssd_generate_tokens(

            model=model,

            input_ids=input_ids,

            max_new_tokens=MAX_GEN_TOKS,

            max_draft_tokens=MAX_DRAFT_TOKENS,

            eos_token_id=(
                tokenizer.eos_token_id
            ),

            **GAMMA_KWARGS,
        )
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    generated_ids = output_ids[
        :,
        input_ids.shape[1]:
    ]

    generated_text = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    )

    return (
        generated_text,
        elapsed,
        generated_ids,
        stats,
    )


# ============================================================
# SSD WARM-UP
# ============================================================

@torch.no_grad()
def warmup_model(
    model,
    tokenizer,
):

    print()
    print("Running SSD warm-up...")

    warmup_text = (
        "Solve this simple math problem: "
        "John has 2 apples and gets 3 more. "
        "How many apples does he have?"
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_MESSAGE,
        },
        {
            "role": "user",
            "content": warmup_text,
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    _ = ssd_generate_tokens(

        model=model,

        input_ids=inputs.input_ids,

        max_new_tokens=8,

        max_draft_tokens=MAX_DRAFT_TOKENS,

        eos_token_id=(
            tokenizer.eos_token_id
        ),

        **GAMMA_KWARGS,
    )

    print("SSD warm-up complete.")


# ============================================================
# JSONL
# ============================================================

def append_jsonl(
    filename,
    record,
):

    with open(
        filename,
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )

        f.flush()


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
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

    print()
    print("=" * 90)
    print("GSM8K — QWEN2.5-7B SSD BENCHMARK")
    print("=" * 90)

    print()
    print(f"Model          : {MODEL_PATH}")
    print(f"Dtype          : {DTYPE}")
    print(f"Device         : {DEVICE}")
    print(f"CPU threads    : {TORCH_THREADS}")
    print(f"Inter-op       : {TORCH_INTEROP_THREADS}")
    print(f"Max new tokens : {MAX_GEN_TOKS}")
    print(f"Max draft      : {MAX_DRAFT_TOKENS}")
    print(f"Examples       : {args.limit}")
    print("Few-shot       : 0")
    print("Draft sampling : Disabled")
    print("Verify sampling: Disabled")

    print()
    print("Adaptive threshold:")
    print(
        json.dumps(
            GAMMA_KWARGS,
            indent=2,
        )
    )

    print()
    print(
        f"Skip pattern   : "
        f"{SKIP_PATTERN_FILE}"
    )

    print(
        f"Log file       : "
        f"{os.path.abspath(args.log)}"
    )

    print(
        f"Result file    : "
        f"{os.path.abspath(args.output)}"
    )

    # ========================================================
    # TOKENIZER
    # ========================================================

    print()
    print("Loading tokenizer...")

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_PATH
        )
    )

    # ========================================================
    # MODEL
    # ========================================================

    print()
    print("Loading Qwen2.5-7B-Instruct...")

    tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    )



    model = (
        AutoModelForCausalLM.from_pretrained(

            MODEL_PATH,

            dtype=DTYPE,

            device_map={
                "": DEVICE
            },
        )
    )

    model.eval()

    print(
        "Model device:",
        model.device,
    )

    print(
        "Model dtype:",
        next(model.parameters()).dtype,
    )

    print(
        "Model layers:",
        model.config.num_hidden_layers,
    )

    # ========================================================
    # LOAD SKIP PATTERN
    # ========================================================

    if not os.path.exists(
        SKIP_PATTERN_FILE
    ):

        raise FileNotFoundError(
            f"Missing skip pattern:\n"
            f"{SKIP_PATTERN_FILE}"
        )

    with open(
        SKIP_PATTERN_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        skip_pattern = json.load(f)

    print()
    print("SSD skip pattern:")

    print(
        json.dumps(
            skip_pattern,
            indent=2,
        )
    )

    # ========================================================
    # APPLY PATTERN
    # ========================================================

    apply_skip_pattern(
        model,
        skip_pattern,
    )

    # ========================================================
    # WARM-UP
    # ========================================================

    warmup_model(
        model,
        tokenizer,
    )

    # ========================================================
    # DATASET
    # ========================================================

    print()
    print("Loading GSM8K test dataset...")

    dataset = load_dataset(
        "openai/gsm8k",
        "main",
        split="test",
    )

    print(
        f"Full dataset size: "
        f"{len(dataset)}"
    )

    dataset = dataset.select(
        range(
            min(
                args.limit,
                len(dataset),
            )
        )
    )

    print(
        f"Evaluation examples: "
        f"{len(dataset)}"
    )

    # ========================================================
    # RESET LOG
    # ========================================================

    with open(
        args.log,
        "w",
        encoding="utf-8",
    ) as f:

        f.write("")

    # ========================================================
    # STATISTICS
    # ========================================================

    total_examples = len(dataset)

    correct_count = 0

    total_latency = 0.0

    total_generated_tokens = 0

    total_draft_tokens = 0

    total_accepted_tokens = 0

    total_verification_steps = 0

    total_draft_time = 0.0

    total_verify_time = 0.0

    total_prefill_time = 0.0

    total_crop_before_verify = 0.0

    total_crop_after_verify = 0.0

    total_python_overhead = 0.0

    results = []

    # ========================================================
    # EVALUATION
    # ========================================================

    for index, example in enumerate(dataset):

        print()
        print("=" * 90)

        print(
            f"[{index + 1}/{total_examples}]"
        )

        print("=" * 90)

        # ----------------------------------------------------
        # PROMPT
        # ----------------------------------------------------

        prompt = build_gsm8k_prompt(
            example,
            tokenizer,
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        ).to(model.device)

        # ----------------------------------------------------
        # GOLD
        # ----------------------------------------------------

        target = get_gsm8k_target(
            example
        )

        # ----------------------------------------------------
        # SSD
        # ----------------------------------------------------

        (
            generated_text,
            elapsed,
            generated_ids,
            stats,
        ) = ssd_generate(

            model=model,

            tokenizer=tokenizer,

            input_ids=inputs.input_ids,
        )

        generated_tokens = (
            generated_ids.shape[1]
        )

        # ----------------------------------------------------
        # SCORE
        # ----------------------------------------------------

        prediction, is_correct = (
            score_answer(
                generated_text,
                target,
            )
        )

        if is_correct:

            correct_count += 1

        # ----------------------------------------------------
        # ACCUMULATE
        # ----------------------------------------------------

        total_latency += elapsed

        total_generated_tokens += (
            generated_tokens
        )

        total_draft_tokens += (
            stats.draft_tokens_total
        )

        total_accepted_tokens += (
            stats.accepted_tokens_total
        )

        total_verification_steps += (
            stats.verification_steps
        )

        total_draft_time += (
            stats.draft_time_total
        )

        total_verify_time += (
            stats.verify_time_total
        )

        total_prefill_time += (
            stats.prefill_time
        )

        total_crop_before_verify += (
            stats.crop_before_verify_time
        )

        total_crop_after_verify += (
            stats.crop_after_verify_time
        )

        total_python_overhead += (
            stats.python_overhead_time
        )

        running_accuracy = (
            correct_count
            / (index + 1)
        )

        tokens_per_sec = (
            generated_tokens / elapsed
            if elapsed > 0
            else 0.0
        )

        # ====================================================
        # PRINT
        # ====================================================

        print()
        print("RAW SSD OUTPUT:")
        print("-" * 90)

        print(
            generated_text
        )

        print("-" * 90)

        print(
            f"Gold       : {target}"
        )

        print(
            f"Prediction : {prediction} "
            f"{'✓' if is_correct else '✗'}"
        )

        print(
            f"Latency    : {elapsed:.3f}s"
        )

        print(
            f"Tokens     : {generated_tokens}"
        )

        print(
            f"Tok/s      : {tokens_per_sec:.3f}"
        )

        print(
            f"Draft      : "
            f"{stats.draft_tokens_total}"
        )

        print(
            f"Accepted   : "
            f"{stats.accepted_tokens_total}"
        )

        print(
            f"Acceptance : "
            f"{stats.acceptance_rate:.2%}"
        )

        print(
            f"Draft len  : "
            f"{stats.average_draft_length:.3f}"
        )

        print(
            f"Verify     : "
            f"{stats.verification_steps}"
        )

        print(
            f"Gamma      : "
            f"{stats.gamma_start:.4f}"
            f" -> "
            f"{stats.gamma_final:.4f}"
        )

        print(
            f"Accuracy   : "
            f"{running_accuracy:.2%}"
        )

        # ====================================================
        # RECORD
        # ====================================================

        record = {

            "index":
                index,

            "id":
                example.get(
                    "id",
                    str(index),
                ),

            "target":
                target,

            "prediction":
                prediction,

            "correct":
                is_correct,

            "latency_sec":
                elapsed,

            "generated_tokens":
                generated_tokens,

            "tokens_per_sec":
                tokens_per_sec,

            # ------------------------------------------------
            # Exact token sequence.
            # Used later for losslessness comparison.
            # ------------------------------------------------

            "generated_token_ids":
                generated_ids[0].tolist(),

            # ------------------------------------------------
            # SSD metrics
            # ------------------------------------------------

            "draft_tokens":
                stats.draft_tokens_total,

            "accepted_tokens":
                stats.accepted_tokens_total,

            "acceptance_rate":
                stats.acceptance_rate,

            "average_draft_length":
                stats.average_draft_length,

            "accepted_tokens_per_verification":
                stats.accepted_tokens_per_verification,

            "verification_steps":
                stats.verification_steps,

            "gamma_start":
                stats.gamma_start,

            "gamma_final":
                stats.gamma_final,

            # ------------------------------------------------
            # Diagnostic timing
            # ------------------------------------------------

            "prefill_time_sec":
                stats.prefill_time,

            "draft_time_sec":
                stats.draft_time_total,

            "crop_before_verify_time_sec":
                stats.crop_before_verify_time,

            "verify_time_sec":
                stats.verify_time_total,

            "crop_after_verify_time_sec":
                stats.crop_after_verify_time,

            "python_overhead_time_sec":
                stats.python_overhead_time,

            "question":
                example["question"],

            "raw_generation":
                generated_text,
        }

        append_jsonl(
            args.log,
            record,
        )

        results.append(
            record
        )

    # ========================================================
    # FINAL METRICS
    # ========================================================

    accuracy = (
        correct_count
        / total_examples
        if total_examples > 0
        else 0.0
    )

    average_latency = (
        total_latency
        / total_examples
        if total_examples > 0
        else 0.0
    )

    tokens_per_sec = (
        total_generated_tokens
        / total_latency
        if total_latency > 0
        else 0.0
    )

    overall_acceptance = (
        total_accepted_tokens
        / total_draft_tokens
        if total_draft_tokens > 0
        else 0.0
    )

    average_draft_length = (
        total_draft_tokens
        / total_verification_steps
        if total_verification_steps > 0
        else 0.0
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print()
    print("=" * 100)
    print("FINAL GSM8K SSD RESULTS")
    print("=" * 100)

    print()
    print(
        f"{'Metric':<45}"
        f"{'SSD':>25}"
    )

    print("-" * 75)

    print(
        f"{'Examples':<45}"
        f"{total_examples:>25}"
    )

    print(
        f"{'Correct':<45}"
        f"{correct_count:>25}"
    )

    print(
        f"{'Accuracy':<45}"
        f"{accuracy:>24.2%}"
    )

    print(
        f"{'Total latency (sec)':<45}"
        f"{total_latency:>25.3f}"
    )

    print(
        f"{'Average latency/example (sec)':<45}"
        f"{average_latency:>25.3f}"
    )

    print(
        f"{'Generated tokens':<45}"
        f"{total_generated_tokens:>25}"
    )

    print(
        f"{'Tokens/sec':<45}"
        f"{tokens_per_sec:>25.3f}"
    )

    print("-" * 75)

    print(
        f"{'Draft tokens':<45}"
        f"{total_draft_tokens:>25}"
    )

    print(
        f"{'Accepted tokens':<45}"
        f"{total_accepted_tokens:>25}"
    )

    print(
        f"{'Acceptance rate':<45}"
        f"{overall_acceptance:>24.2%}"
    )

    print(
        f"{'Average draft length':<45}"
        f"{average_draft_length:>25.3f}"
    )

    print(
        f"{'Verification steps':<45}"
        f"{total_verification_steps:>25}"
    )

    print("-" * 75)

    print("--- SSD diagnostic timing ---")

    print(
        f"{'Prefill (sec)':<45}"
        f"{total_prefill_time:>25.3f}"
    )

    print(
        f"{'Draft phase (sec)':<45}"
        f"{total_draft_time:>25.3f}"
    )

    print(
        f"{'Crop before verify (sec)':<45}"
        f"{total_crop_before_verify:>25.3f}"
    )

    print(
        f"{'Verify phase (sec)':<45}"
        f"{total_verify_time:>25.3f}"
    )

    print(
        f"{'Crop after verify (sec)':<45}"
        f"{total_crop_after_verify:>25.3f}"
    )

    print(
        f"{'Python/bookkeeping (sec)':<45}"
        f"{total_python_overhead:>25.3f}"
    )

    print("-" * 75)

    print(
        f"Log file    : "
        f"{os.path.abspath(args.log)}"
    )

    print(
        f"Result file : "
        f"{os.path.abspath(args.output)}"
    )

    print("=" * 100)

    # ========================================================
    # SAVE JSON
    # ========================================================

    output = {

        "metadata": {

            "timestamp":
                datetime.now().isoformat(),

            "model":
                MODEL_PATH,

            "dtype":
                str(DTYPE),

            "device":
                DEVICE,

            "torch_threads":
                TORCH_THREADS,

            "torch_interop_threads":
                TORCH_INTEROP_THREADS,

            "dataset":
                "openai/gsm8k",

            "dataset_config":
                "main",

            "split":
                "test",

            "num_fewshot":
                0,

            "max_gen_toks":
                MAX_GEN_TOKS,

            "max_draft_tokens":
                MAX_DRAFT_TOKENS,

            "limit":
                args.limit,

            "skip_pattern_file":
                SKIP_PATTERN_FILE,

            "skip_pattern":
                skip_pattern,

            "gamma":
                GAMMA_KWARGS,
        },

        "summary": {

            "examples":
                total_examples,

            "correct":
                correct_count,

            "accuracy":
                accuracy,

            "total_latency_sec":
                total_latency,

            "average_latency_sec":
                average_latency,

            "generated_tokens":
                total_generated_tokens,

            "tokens_per_sec":
                tokens_per_sec,

            "draft_tokens":
                total_draft_tokens,

            "accepted_tokens":
                total_accepted_tokens,

            "acceptance_rate":
                overall_acceptance,

            "average_draft_length":
                average_draft_length,

            "verification_steps":
                total_verification_steps,

            "prefill_time_sec":
                total_prefill_time,

            "draft_time_sec":
                total_draft_time,

            "crop_before_verify_time_sec":
                total_crop_before_verify,

            "verify_time_sec":
                total_verify_time,

            "crop_after_verify_time_sec":
                total_crop_after_verify,

            "python_overhead_time_sec":
                total_python_overhead,
        },

        "examples":
            results,
    }

    with open(
        args.output,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        "GSM8K SSD "
        "EVALUATION COMPLETE"
    )


if __name__ == "__main__":

    main()

