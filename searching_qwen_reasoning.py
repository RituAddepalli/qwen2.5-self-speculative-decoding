"""
Bayesian optimization search for Qwen2.5-7B-Instruct
Self-Speculative Decoding layer-skip patterns.

Based on the official dilab-zju/self-speculative-decoding
searching.py methodology.

Official search structure:
    1. Bayesian Optimization over one binary decision per
       attention/MLP sublayer.
    2. Convert continuous BO values using:
           value > 0.5 -> skip
           value <= 0.5 -> execute
    3. Apply the candidate skip pattern.
    4. Run complete self-speculative generation.
    5. Measure end-to-end generation time.
    6. Maximize tokens/sec.

Qwen2.5-specific adaptation:
    - Qwen2.5-7B-Instruct
    - 28 transformer layers
    - Layer 0 protected because of our current
      DynamicCache/cache-mask implementation
    - Qwen layer.skip_attn / layer.skip_mlp interface
    - decoding.ssd_generate_tokens()
    - CPU / Intel execution
    - adaptive gamma already implemented in decoding.py
    - persistent trial logging

# IMPORTANT:
 The adaptive-gamma mechanism is an addition to decoding.py
 in this implementation. Gamma adaptation is kept fixed during Bayesian optimization.

 This experiment combines:
   - Qwen2.5-7B-Instruct
  - Self-speculative decoding
   - Layer skipping
  - Adaptive gamma
   - Bayesian optimization for layer-pattern search

 This implementation is an adaptation and is not claimed
 to be a byte-for-byte reproduction of the official repository.

SEARCH SPACE:
    Qwen2.5-7B has 28 transformer layers.

    Layer 0:
        protected

    Layers 1..27:
        27 attention decisions
        27 MLP decisions

    Total:
        54 BO dimensions

OBJECTIVE:
    maximize:

        total generated tokens
        ----------------------
        total generation time

    This is equivalent to minimizing average time per generated
    token for a fixed evaluation workload.

REPO-STYLE SEARCH:
    init_points = 5
    n_iter = 995

    Therefore:
        exactly 1000 BO evaluations.

IMPORTANT:
    JSON files are experiment logs only.
    BayesianOptimization GP state is NOT restored if the program
    is restarted.
"""

import json
import os
import time

import torch
from bayes_opt import BayesianOptimization

from decoding import ssd_generate_tokens


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"

DEVICE = "cpu"
DTYPE = torch.bfloat16

# ------------------------------------------------------------
# Search size
# ------------------------------------------------------------

# Match the repository's search call:
#
#     optimizer.maximize(init_points=0, n_iter=1000)
#
N_INIT_POINTS = 0
N_ITER = 25


RANDOM_STATE = 1

# ------------------------------------------------------------
# Generation configuration
# ------------------------------------------------------------

MAX_NEW_TOKENS = 1028
MAX_DRAFT_TOKENS = 8

# ------------------------------------------------------------
# Layer protection
# ------------------------------------------------------------

NEVER_SKIP_LAYERS = (0,)

# ------------------------------------------------------------
# Output files
# ------------------------------------------------------------

RESULTS_FILE = "bo_search_results_qwen7b_reasoning_alpha0.9.json"
BEST_PATTERN_FILE = "skip_layers_qwen7b_reasoning_alpha0.9.json"


# ============================================================
# ADAPTIVE GAMMA
# ============================================================

"""
These parameters belong to OUR current decoding.py.

They are kept fixed for every BO candidate.

We are NOT asking Bayesian Optimization to optimize gamma.
The only thing BO is searching is the layer-skip pattern.
"""

GAMMA_KWARGS = {
    "initial_gamma": 0.8,
    "alpha": 0.9,
    "beta1": 0.5,
    "beta2": 0.9,
    "epsilon": 0.01,
}


# ============================================================
# DEVELOPMENT PROMPTS
# ============================================================

DEV_PROMPTS_RAW = [

    "A store has 24 apples. It sells 8 apples in the morning and twice as many in the afternoon. How many apples remain?",

    "A train travels 60 kilometers in the first hour and 80 kilometers in the second hour. What is the average speed of the train over these two hours?",

     "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",

    "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",

    "Shiela bought five cell phones for $150 each for a 3-month installment. A 2% interest will be charged for each unit. How much will Shiela pay each month for 3 months?",
]
    # "Give me a short introduction to large language models.",

    # "Explain the difference between supervised and unsupervised learning.",

    # "Explain how a transformer model processes a sequence of tokens.",

    # "What is the difference between a CPU and a GPU for machine learning?",

    # "What are the main causes of climate change?",

    # "Write a short paragraph about the history of the internet.",

    # "Explain why neural networks need activation functions.",

    # "Write a short paragraph explaining how artificial intelligence is used in healthcare.",



SYSTEM_MESSAGE = (
    # "You are Qwen, created by Alibaba Cloud. "
    # "You are a helpful assistant."
   
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. "
    "Solve mathematical problems carefully using step-by-step reasoning. "
    "Verify your calculations before giving the final answer. "
    "Provide a clear final answer after your reasoning."
)



# ============================================================
# SEARCH CLASS
# ============================================================

class QwenLayerSkippingSearch:

    def __init__(
        self,
        model,
        tokenizer,
        dev_prompts,
        max_new_tokens=1028,
        max_draft_tokens=8,
        never_skip_layers=(0,),
        gamma_kwargs=None,
        random_state=1,
        results_file="bo_search_results.json",
        best_pattern_file="skip_layers.json",
    ):

        self.model = model
        self.tokenizer = tokenizer
        self.dev_prompts = dev_prompts

        self.max_new_tokens = max_new_tokens
        self.max_draft_tokens = max_draft_tokens

        self.gamma_kwargs = gamma_kwargs or {}

        self.never_skip = set(never_skip_layers)

        self.random_state = random_state

        self.results_file = results_file
        self.best_pattern_file = best_pattern_file

        self.eval_log = []

        # ----------------------------------------------------
        # QWEN ARCHITECTURE
        # ----------------------------------------------------

        self.num_layers = model.config.num_hidden_layers

        if self.num_layers <= 1:
            raise ValueError(
                f"Invalid number of transformer layers: "
                f"{self.num_layers}"
            )   

        # Layer 0 must remain full for the current implementation.
        if 0 not in self.never_skip:
            raise ValueError(
                "Layer 0 must be protected in this implementation."
            )

        # ----------------------------------------------------
        # SEARCHABLE ATTENTION LAYERS
        # ----------------------------------------------------

        self.attn_dims = [
            i
            for i in range(self.num_layers)
            if i not in self.never_skip
        ]

        # ----------------------------------------------------
        # SEARCHABLE MLP LAYERS
        # ----------------------------------------------------

        self.mlp_dims = [
            i
            for i in range(self.num_layers)
            if i not in self.never_skip
        ]

        # ----------------------------------------------------
        # SAFETY CHECKS
        # ----------------------------------------------------

        expected_searchable_layers = (
                self.num_layers - len(self.never_skip)
        )

        if len(self.attn_dims) != expected_searchable_layers:
            raise RuntimeError(
                f"Expected {expected_searchable_layers} "
                f"searchable attention layers, "
                f"got {len(self.attn_dims)}."
            )

        if len(self.mlp_dims) != expected_searchable_layers:
            raise RuntimeError(
                f"Expected {expected_searchable_layers} "
                f"searchable MLP layers, "
                f"got {len(self.mlp_dims)}."
            )

        # ----------------------------------------------------
        # 54-DIMENSIONAL SEARCH SPACE
        # ----------------------------------------------------

        self.pbounds = {}

        for i in self.attn_dims:
            self.pbounds[f"attn_{i}"] = (0.0, 1.0)

        for i in self.mlp_dims:
            self.pbounds[f"mlp_{i}"] = (0.0, 1.0)

        expected_bo_dimensions = (
            len(self.attn_dims) +
            len(self.mlp_dims)
        )

        if len(self.pbounds) != expected_bo_dimensions:
            raise RuntimeError(
                f"Expected {expected_bo_dimensions} BO dimensions, "
                f"got {len(self.pbounds)}."
            )

        # ----------------------------------------------------
        # BAYESIAN OPTIMIZATION
        # ----------------------------------------------------

        self.optimizer = BayesianOptimization(
            f=self._evaluate,
            pbounds=self.pbounds,
            random_state=self.random_state,
            allow_duplicate_points=True,
        )

        # Same GP noise setting used by the repository.
        self.optimizer.set_gp_params(
            alpha=1e-2
        )

        # ----------------------------------------------------
        # INITIAL RESULT FILE
        # ----------------------------------------------------

        self._save_results()

    # ========================================================
    # PARAMS -> SKIP PATTERN
    # ========================================================

    def _params_to_skip_pattern(self, params):

        attn_skip = sorted(
            i
            for i in self.attn_dims
            if params[f"attn_{i}"] > 0.5
        )

        mlp_skip = sorted(
            i
            for i in self.mlp_dims
            if params[f"mlp_{i}"] > 0.5
        )

        return {
            "attn": attn_skip,
            "mlp": mlp_skip,
        }

    # ========================================================
    # APPLY PATTERN TO QWEN
    # ========================================================

    def _apply_skip_pattern(self, skip_pattern):

        attn_set = set(skip_pattern["attn"])
        mlp_set = set(skip_pattern["mlp"])

        # Store the active pattern in config.
        self.model.config.skip_layer_ids = {
            "attn": sorted(attn_set),
            "mlp": sorted(mlp_set),
        }

        # Apply the actual flags used by modeling_qwen2.py.
        for i, layer in enumerate(self.model.model.layers):

            if i == 0:

                # Layer 0 is ALWAYS full.
                layer.skip_attn = False
                layer.skip_mlp = False

            else:

                layer.skip_attn = i in attn_set
                layer.skip_mlp = i in mlp_set

    # ========================================================
    # SAVE EXPERIMENT RESULTS
    # ========================================================

    def _save_results(self):

        data = {
            "experiment": {
                "model": MODEL_PATH,
                "device": DEVICE,
                "dtype": str(DTYPE),

                "num_layers": self.num_layers,

                "never_skip_layers": sorted(
                    self.never_skip
                ),

                "attention_dimensions": len(
                    self.attn_dims
                ),

                "mlp_dimensions": len(
                    self.mlp_dims
                ),

                "search_dimensions": len(
                    self.pbounds
                ),

                "max_new_tokens": (
                    self.max_new_tokens
                ),

                "max_draft_tokens": (
                    self.max_draft_tokens
                ),

                "init_points": N_INIT_POINTS,

                "n_iter": N_ITER,

                "random_state": (
                    self.random_state
                ),

                "gamma_kwargs": (
                    self.gamma_kwargs
                ),

                "num_dev_prompts": (
                    len(self.dev_prompts)
                ),
            },

            "trials_completed": len(
                self.eval_log
            ),

            "trials": self.eval_log,
        }

        temp_file = self.results_file + ".tmp"

        with open(
            temp_file,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                data,
                f,
                indent=2,
            )

        # Atomic replacement.
        os.replace(
            temp_file,
            self.results_file,
        )

    # ========================================================
    # SAVE BEST PATTERN
    # ========================================================

    def _save_best_pattern(
        self,
        skip_pattern,
        score,
        trial_number,
    ):

        # IMPORTANT:
        #
        # Save the RAW pattern.
        #
        # Do NOT save:
        #
        # {
        #     "skip_layer_ids": {
        #         ...
        #     }
        # }
        #
        # because decoding.py/test code expects:
        #
        # {
        #     "attn": [...],
        #     "mlp": [...]
        # }

        with open(
            self.best_pattern_file,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                skip_pattern,
                f,
                indent=2,
            )

        print()
        print(
            f"Best pattern saved to "
            f"{self.best_pattern_file}"
        )
        print(
            f"Best score: "
            f"{score:.4f} tokens/sec"
        )
        print(
            f"Best trial: "
            f"{trial_number}"
        )

    # ========================================================
    # OBJECTIVE FUNCTION
    # ========================================================

    def _evaluate(self, **params):

        trial_number = len(
            self.eval_log
        ) + 1

        # ----------------------------------------------------
        # Convert BO parameters -> binary pattern
        # ----------------------------------------------------

        skip_pattern = (
            self._params_to_skip_pattern(
                params
            )
        )

        n_attn = len(
            skip_pattern["attn"]
        )

        n_mlp = len(
            skip_pattern["mlp"]
        )

        print()
        print("=" * 70)
        print(
            f"BO TRIAL {trial_number}"
        )
        print("=" * 70)

        print(
            f"Attention skipped "
            f"({n_attn}/{len(self.attn_dims)}): "
            f"{skip_pattern['attn']}"
        )

        print(
            f"MLP skipped "
            f"({n_mlp}/{len(self.mlp_dims)}): "
            f"{skip_pattern['mlp']}"
        )

        print(
            f"Total sublayers skipped: "
            f"{n_attn + n_mlp}"
        )

        # ----------------------------------------------------
        # Apply candidate
        # ----------------------------------------------------

        self._apply_skip_pattern(
            skip_pattern
        )

        total_tokens = 0
        total_time = 0.0

        prompt_results = []

        trial_start = time.perf_counter()

        try:

            # =================================================
            # EVALUATE ALL DEVELOPMENT PROMPTS
            # =================================================

            for prompt_index, prompt in enumerate(
                self.dev_prompts,
                start=1,
            ):

                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                ).to(self.model.device)

                # ------------------------------------------------
                # IMPORTANT REPO-STYLE EVALUATION
                #
                # We time the COMPLETE generation call.
                #
                # Tokenization is outside the timer.
                #
                # This includes:
                #   - initial full prefill
                #   - draft generation
                #   - draft cache rollback
                #   - verification
                #   - acceptance/rejection
                #   - final rollback
                #   - adaptive gamma
                #   - Python SSD overhead
                #
                # EOS is disabled here because the repo's
                # evaluation uses its default early_stop=False.
                #
                # Therefore every candidate is evaluated over
                # the same requested generation length.
                # ------------------------------------------------

                start = time.perf_counter()

                _, stats = ssd_generate_tokens(
                    self.model,
                    inputs.input_ids,
                    max_new_tokens=(
                        self.max_new_tokens
                    ),
                    max_draft_tokens=(
                        self.max_draft_tokens
                    ),

                    # Match repo's default:
                    # early_stop=False
                    #
                    # Our low-level Qwen implementation expresses
                    # this by not supplying an EOS stopping ID.
                    eos_token_id=None,

                    **self.gamma_kwargs,
                )

                elapsed = (
                    time.perf_counter()
                    - start
                )

                generated_tokens = (
                    stats.generated_tokens
                )

                prompt_tokens_per_sec = (
                    generated_tokens / elapsed
                    if elapsed > 0
                    else 0.0
                )

                total_tokens += (
                    generated_tokens
                )

                total_time += elapsed

                prompt_results.append(
                    {
                        "prompt_index":
                            prompt_index,

                        "generated_tokens":
                            generated_tokens,

                        "time_sec":
                            elapsed,

                        "tokens_per_sec":
                            prompt_tokens_per_sec,

                        "draft_tokens":
                            stats.draft_tokens_total,

                        "accepted_tokens":
                            stats.accepted_tokens_total,

                        "acceptance_rate":
                            stats.acceptance_rate,

                        "average_draft_length":
                            stats.average_draft_length,

                        "accepted_per_verification":
                            stats.accepted_tokens_per_verification,

                        "verification_steps":
                            stats.verification_steps,

                        "gamma_start":
                            stats.gamma_start,

                        "gamma_final":
                            stats.gamma_final,

                        # Diagnostic only.
                        "prefill_time":
                            stats.prefill_time,

                        "draft_time":
                            stats.draft_time_total,

                        "verify_time":
                            stats.verify_time_total,

                        "crop_before_verify":
                            stats.crop_before_verify_time,

                        "crop_after_verify":
                            stats.crop_after_verify_time,

                        "python_overhead":
                            stats.python_overhead_time,
                    }
                )

                print(
                    f"Prompt "
                    f"{prompt_index}/"
                    f"{len(self.dev_prompts)}"
                    f" | "
                    f"{generated_tokens} tokens"
                    f" | "
                    f"{elapsed:.3f} sec"
                    f" | "
                    f"{prompt_tokens_per_sec:.3f} tok/s"
                )

            # =================================================
            # AGGREGATE OBJECTIVE
            # =================================================

            if total_time <= 0:

                score = -1.0

            else:

                # EXACT SEARCH OBJECTIVE:
                #
                # total generated tokens
                # ----------------------
                # total generation time
                #
                # BO maximizes this value.
                #
                score = (
                    total_tokens
                    / total_time
                )

            trial_wall_time = (
                time.perf_counter()
                - trial_start
            )

            # =================================================
            # STORE TRIAL
            # =================================================

            trial_record = {
                "trial": trial_number,

                "params": {
                    k: float(v)
                    for k, v in params.items()
                },

                "skip_pattern":
                    skip_pattern,

                "n_attn_skip":
                    n_attn,

                "n_mlp_skip":
                    n_mlp,

                "n_total_skip":
                    n_attn + n_mlp,

                "total_tokens":
                    total_tokens,

                "total_time":
                    total_time,

                "tokens_per_sec":
                    score,

                "score":
                    score,

                "trial_wall_time":
                    trial_wall_time,

                "prompt_results":
                    prompt_results,

                "error":
                    None,
            }

            self.eval_log.append(
                trial_record
            )

            # =================================================
            # CHECK BEST
            # =================================================

            previous_scores = [
                x["score"]
                for x in self.eval_log[:-1]
                if (
                    x.get("error") is None
                    and x.get("score", -1) >= 0
                )
            ]

            previous_best = (
                max(previous_scores)
                if previous_scores
                else float("-inf")
            )

            if score > previous_best:

                self._save_best_pattern(
                    skip_pattern=skip_pattern,
                    score=score,
                    trial_number=trial_number,
                )

                print()
                print(
                    "*** NEW BEST PATTERN ***"
                )

                print(
                    f"Trial: "
                    f"{trial_number}"
                )

                print(
                    f"Tokens/sec: "
                    f"{score:.4f}"
                )

            # =================================================
            # SAVE EVERY TRIAL
            # =================================================

            self._save_results()

            print()
            print(
                f"Trial {trial_number} score: "
                f"{score:.4f} tokens/sec"
            )

            print(
                f"Total search trials recorded: "
                f"{len(self.eval_log)}"
            )

            return score

        except Exception as e:

            error_message = repr(e)

            print()
            print(
                "*** TRIAL FAILED ***"
            )

            print(
                error_message
            )

            trial_record = {
                "trial": trial_number,

                "params": {
                    k: float(v)
                    for k, v in params.items()
                },

                "skip_pattern":
                    skip_pattern,

                "n_attn_skip":
                    n_attn,

                "n_mlp_skip":
                    n_mlp,

                "n_total_skip":
                    n_attn + n_mlp,

                "total_tokens":
                    total_tokens,

                "total_time":
                    total_time,

                "tokens_per_sec":
                    -1.0,

                "score":
                    -1.0,

                "prompt_results":
                    prompt_results,

                "error":
                    error_message,
            }

            self.eval_log.append(
                trial_record
            )

            self._save_results()

            return -1.0

    # ========================================================
    # RUN SEARCH
    # ========================================================

    def run(
        self,
        init_points=0,
        n_iter=1000,
    ):

        # ----------------------------------------------------
        # atleast 1 evaluation should be done
        # ----------------------------------------------------

        if init_points + n_iter <= 0:

            raise ValueError(
                "At least one evaluation is required."
            )
        # if init_points + n_iter <= 0:

        #     raise ValueError(
        #         "At least one evaluation is required."
        #     )

        print()
        print("=" * 70)
        print(
            "STARTING BAYESIAN OPTIMIZATION"
        )
        print("=" * 70)

        print(
            f"Model              : "
            f"{MODEL_PATH}"
        )

        print(
            f"Device             : "
            f"{DEVICE}"
        )

        print(
            f"Transformer layers : "
            f"{self.num_layers}"
        )

        print(
            f"Protected layers   : "
            f"{sorted(self.never_skip)}"
        )

        print(
            f"Attention dims     : "
            f"{len(self.attn_dims)}"
        )

        print(
            f"MLP dims           : "
            f"{len(self.mlp_dims)}"
        )

        print(
            f"Total BO dims      : "
            f"{len(self.pbounds)}"
        )

        print(
            f"Dev prompts        : "
            f"{len(self.dev_prompts)}"
        )

        print(
            f"Max new tokens     : "
            f"{self.max_new_tokens}"
        )

        print(
            f"Max draft tokens   : "
            f"{self.max_draft_tokens}"
        )

        print(
            f"Initial gamma      : "
            f"{self.gamma_kwargs.get('initial_gamma')}"
        )

        print(
            f"Gamma target alpha : "
            f"{self.gamma_kwargs.get('alpha')}"
        )

        print(
            f"Initial points     : "
            f"{init_points}"
        )

        print(
            f"BO iterations      : "
            f"{n_iter}"
        )

        print(
            f"Total evaluations  : "
            f"{init_points + n_iter}"
        )

        print(
            "Objective          : "
            "end-to-end tokens/sec"
        )

        print("=" * 70)
        print()

        # ----------------------------------------------------
        # OFFICIAL REPO-STYLE BO CALL
        # ----------------------------------------------------

        self.optimizer.maximize(
            init_points=init_points,
            n_iter=n_iter,
        )

        # ----------------------------------------------------
        # BEST BO POINT
        # ----------------------------------------------------

        best = self.optimizer.max

        best_params = best["params"]

        best_score = best["target"]

        best_pattern = (
            self._params_to_skip_pattern(
                best_params
            )
        )

        # ----------------------------------------------------
        # SAVE FINAL BEST PATTERN
        # ----------------------------------------------------

        self._save_best_pattern(
            skip_pattern=best_pattern,
            score=best_score,
            trial_number=len(
                self.eval_log
            ),
        )

        self._save_results()

        # ----------------------------------------------------
        # FINAL VALIDATION
        # ----------------------------------------------------

        expected_trials = init_points + n_iter

        if len(self.eval_log) != expected_trials:

            raise RuntimeError(
                f"Expected {expected_trials} recorded "
                f"evaluations, got "
                f"{len(self.eval_log)}."
    )

        # ----------------------------------------------------
        # FINAL REPORT
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print(
            "BAYESIAN OPTIMIZATION COMPLETE"
        )
        print("=" * 70)

        print(
            f"Trials completed : "
            f"{len(self.eval_log)}"
        )

        print(
            f"Best tokens/sec  : "
            f"{best_score:.4f}"
        )

        print()
        print(
            "BEST ATTENTION SKIP LAYERS:"
        )

        print(
            best_pattern["attn"]
        )

        print()
        print(
            "BEST MLP SKIP LAYERS:"
        )

        print(
            best_pattern["mlp"]
        )

        print()
        print(
            f"Attention skipped: "
            f"{len(best_pattern['attn'])}"
        )

        print(
            f"MLP skipped      : "
            f"{len(best_pattern['mlp'])}"
        )

        print(
            f"Total skipped    : "
            f"{len(best_pattern['attn']) + len(best_pattern['mlp'])}"
        )

        print()
        print(
            f"Pattern file     : "
            f"{self.best_pattern_file}"
        )

        print(
            f"Results file     : "
            f"{self.results_file}"
        )

        print("=" * 70)

        return {
            "skip_layer_ids":
                best_pattern,

            "tokens_per_sec":
                best_score,

            "trials":
                len(self.eval_log),
        }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print(
        "QWEN2.5-7B BAYESIAN LAYER-SKIP SEARCH"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # CPU INFORMATION
    # --------------------------------------------------------

    print()
    print(
        f"PyTorch threads       : "
        f"{torch.get_num_threads()}"
    )

    print(
        f"PyTorch interop      : "
        f"{torch.get_num_interop_threads()}"
    )

    print()

    # --------------------------------------------------------
    # LOAD TOKENIZER
    # --------------------------------------------------------

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    print(
        "Loading tokenizer..."
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_PATH
        )
    )

    # --------------------------------------------------------
    # LOAD QWEN
    # --------------------------------------------------------

    print(
        "Loading Qwen2.5-7B-Instruct..."
    )

    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            dtype=DTYPE,
            device_map={"": DEVICE},
        )
    )

    model.eval()

    # --------------------------------------------------------
    # VERIFY MODEL
    # --------------------------------------------------------

    print()

    print(
        f"Model device : "
        f"{model.device}"
    )

    print(
        f"Model dtype  : "
        f"{next(model.parameters()).dtype}"
    )

    print(
        f"Model layers : "
        f"{model.config.num_hidden_layers}"
    )

    print(
        f"Model layers : "
        f"{model.config.num_hidden_layers}"
    )

    # --------------------------------------------------------
    # PREPARE CHAT PROMPTS
    # --------------------------------------------------------

    dev_prompts = []

    for prompt in DEV_PROMPTS_RAW:

        messages = [
            {
                "role": "system",
                "content": SYSTEM_MESSAGE,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        formatted_prompt = (
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        dev_prompts.append(
            formatted_prompt
        )

    if len(dev_prompts) != len(DEV_PROMPTS_RAW):
        
        raise RuntimeError(
            f"Expected {len(DEV_PROMPTS_RAW)} development prompts, "
            f"got {len(dev_prompts)}."
        )

    print()
    print(
        f"Development prompts: "
        f"{len(dev_prompts)}"
    )

    # --------------------------------------------------------
    # CREATE SEARCH OBJECT
    # --------------------------------------------------------

    search = QwenLayerSkippingSearch(
        model=model,
        tokenizer=tokenizer,
        dev_prompts=dev_prompts,
        max_new_tokens=MAX_NEW_TOKENS,
        max_draft_tokens=MAX_DRAFT_TOKENS,
        never_skip_layers=NEVER_SKIP_LAYERS,
        gamma_kwargs=GAMMA_KWARGS,
        random_state=RANDOM_STATE,
        results_file=RESULTS_FILE,
        best_pattern_file=BEST_PATTERN_FILE,
    )

    # --------------------------------------------------------
    # PRINT SEARCH SPACE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "SEARCH SPACE VERIFIED"
    )
    print("=" * 70)

    print(
        f"Attention layers : "
        f"{search.attn_dims}"
    )

    print(
        f"MLP layers       : "
        f"{search.mlp_dims}"
    )

    print(
        f"Attention dims   : "
        f"{len(search.attn_dims)}"
    )

    print(
        f"MLP dims         : "
        f"{len(search.mlp_dims)}"
    )

    print(
        f"Total BO dims    : "
        f"{len(search.pbounds)}"
    )

    print(
        "Conversion       : "
        "> 0.5 = SKIP"
    )

    print(
        "Layer 0          : "
        "PROTECTED"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # FINAL SAFETY CHECKS
    # --------------------------------------------------------

    assert search.num_layers == 28

    assert len(search.attn_dims) == 27

    assert len(search.mlp_dims) == 27

    assert len(search.pbounds) == 54

    assert 0 not in search.attn_dims

    assert 0 not in search.mlp_dims

    assert 0 in search.never_skip

    # --------------------------------------------------------
    # START 1000-TRIAL SEARCH
    # --------------------------------------------------------

    result = search.run(
        init_points=N_INIT_POINTS,
        n_iter=N_ITER,
    )

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "FINAL BEST SKIP PATTERN"
    )
    print("=" * 70)

    print(
        json.dumps(
            result["skip_layer_ids"],
            indent=2,
        )
    )

    print()

    print(
        f"Best tokens/sec: "
        f"{result['tokens_per_sec']:.4f}"
    )

    print(
        f"Total evaluations: "
        f"{result['trials']}"
    )

    print()
    print(
        "Saved files:"
    )

    print(
        f"  {BEST_PATTERN_FILE}"
    )

    print(
        f"  {RESULTS_FILE}"
    )

    print("=" * 70)








# 1.5b model 

# """
# Bayesian optimization search for Qwen2.5-1.5B-Instruct
# Self-Speculative Decoding layer-skip patterns.

# Based on the official dilab-zju/self-speculative-decoding
# searching.py methodology.

# Official search structure:
#     1. Bayesian Optimization over one binary decision per
#        attention/MLP sublayer.
#     2. Convert continuous BO values using:
#            value > 0.5 -> skip
#            value <= 0.5 -> execute
#     3. Apply the candidate skip pattern.
#     4. Run complete self-speculative generation.
#     5. Measure end-to-end generation time.
#     6. Maximize tokens/sec.

# Qwen2.5-specific adaptation:
#     - Qwen2.5-1.5B-Instruct
#     - 28 transformer layers
#     - Layer 0 protected because of our current
#       DynamicCache/cache-mask implementation
#     - Qwen layer.skip_attn / layer.skip_mlp interface
#     - decoding.ssd_generate_tokens()
#     - CPU / Intel execution
#     - adaptive gamma already implemented in decoding.py
#     - persistent trial logging

# IMPORTANT:
#     The adaptive-gamma mechanism is addition to decoding.py .
#     It is kept FIXED during Bayesian optimization.

#     Therefore this experiment is:

#         Qwen2.5
#         + self-speculative decoding
#         + layer skipping
#         + adaptive gamma
#         + Bayesian layer-pattern search

#     It is NOT claimed to be a byte-for-byte reproduction of
#     the official repository.

# SEARCH SPACE:
#     Qwen2.5-1.5B has 28 transformer layers.

#     Layer 0:
#         protected

#     Layers 1..27:
#         27 attention decisions
#         27 MLP decisions

#     Total:
#         54 BO dimensions

# OBJECTIVE:
#     maximize:

#         total generated tokens
#         ----------------------
#         total generation time

#     This is equivalent to minimizing average time per generated
#     token for a fixed evaluation workload.

# REPO-STYLE SEARCH:
#     init_points = 5
#     n_iter = 995

#     Therefore:
#         exactly 1000 BO evaluations.

# IMPORTANT:
#     JSON files are experiment logs only.
#     BayesianOptimization GP state is NOT restored if the program
#     is restarted.
# """

# import json
# import os
# import time

# import torch
# from bayes_opt import BayesianOptimization

# from decoding import ssd_generate_tokens


# # ============================================================
# # CONFIGURATION
# # ============================================================

# MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct"

# DEVICE = "cpu"
# DTYPE = torch.float32

# # ------------------------------------------------------------
# # Search size
# # ------------------------------------------------------------

# # Match the repository's search call:
# #
# #     optimizer.maximize(init_points=0, n_iter=1000)
# #
# N_INIT_POINTS = 5
# N_ITER = 995

# RANDOM_STATE = 1

# # ------------------------------------------------------------
# # Generation configuration
# # ------------------------------------------------------------

# MAX_NEW_TOKENS = 64
# MAX_DRAFT_TOKENS = 8

# # ------------------------------------------------------------
# # Layer protection
# # ------------------------------------------------------------

# NEVER_SKIP_LAYERS = (0,)

# # ------------------------------------------------------------
# # Output files
# # ------------------------------------------------------------

# RESULTS_FILE = "bo_search_results.json"
# BEST_PATTERN_FILE = "skip_layers.json"


# # ============================================================
# # ADAPTIVE GAMMA
# # ============================================================

# """
# These parameters belong to OUR current decoding.py.

# They are kept fixed for every BO candidate.

# We are NOT asking Bayesian Optimization to optimize gamma.
# The only thing BO is searching is the layer-skip pattern.
# """

# GAMMA_KWARGS = {
#     "initial_gamma": 0.8,
#     "alpha": 0.9,
#     "beta1": 0.5,
#     "beta2": 0.9,
#     "epsilon": 0.01,
# }


# # ============================================================
# # DEVELOPMENT PROMPTS
# # ============================================================

# DEV_PROMPTS_RAW = [
#     "Give me a short introduction to large language models.",

#     "Explain the difference between supervised and unsupervised learning.",

#     "Explain how a transformer model processes a sequence of tokens.",

#     "What is the difference between a CPU and a GPU for machine learning?",

#     "What are the main causes of climate change?",

#     "Write a short paragraph about the history of the internet.",

#     "Explain why neural networks need activation functions.",

#     "Write a short paragraph explaining how artificial intelligence is used in healthcare.",
# ]


# SYSTEM_MESSAGE = (
#     "You are Qwen, created by Alibaba Cloud. "
#     "You are a helpful assistant."
# )


# # ============================================================
# # SEARCH CLASS
# # ============================================================

# class QwenLayerSkippingSearch:

#     def __init__(
#         self,
#         model,
#         tokenizer,
#         dev_prompts,
#         max_new_tokens=64,
#         max_draft_tokens=8,
#         never_skip_layers=(0,),
#         gamma_kwargs=None,
#         random_state=1,
#         results_file="bo_search_results.json",
#         best_pattern_file="skip_layers.json",
#     ):

#         self.model = model
#         self.tokenizer = tokenizer
#         self.dev_prompts = dev_prompts

#         self.max_new_tokens = max_new_tokens
#         self.max_draft_tokens = max_draft_tokens

#         self.gamma_kwargs = gamma_kwargs or {}

#         self.never_skip = set(never_skip_layers)

#         self.random_state = random_state

#         self.results_file = results_file
#         self.best_pattern_file = best_pattern_file

#         self.eval_log = []

#         # ----------------------------------------------------
#         # QWEN ARCHITECTURE
#         # ----------------------------------------------------

#         self.num_layers = model.config.num_hidden_layers

#         if self.num_layers != 28:
#             raise ValueError(
#                 "This experiment expects Qwen2.5-1.5B-Instruct "
#                 f"with 28 layers, but model reports "
#                 f"{self.num_layers}."
#             )

#         # Layer 0 must remain full for the current implementation.
#         if 0 not in self.never_skip:
#             raise ValueError(
#                 "Layer 0 must be protected in this implementation."
#             )

#         # ----------------------------------------------------
#         # SEARCHABLE ATTENTION LAYERS
#         # ----------------------------------------------------

#         self.attn_dims = [
#             i
#             for i in range(self.num_layers)
#             if i not in self.never_skip
#         ]

#         # ----------------------------------------------------
#         # SEARCHABLE MLP LAYERS
#         # ----------------------------------------------------

#         self.mlp_dims = [
#             i
#             for i in range(self.num_layers)
#             if i not in self.never_skip
#         ]

#         # ----------------------------------------------------
#         # SAFETY CHECKS
#         # ----------------------------------------------------

#         if len(self.attn_dims) != 27:
#             raise RuntimeError(
#                 f"Expected 27 searchable attention layers, "
#                 f"got {len(self.attn_dims)}."
#             )

#         if len(self.mlp_dims) != 27:
#             raise RuntimeError(
#                 f"Expected 27 searchable MLP layers, "
#                 f"got {len(self.mlp_dims)}."
#             )

#         # ----------------------------------------------------
#         # 54-DIMENSIONAL SEARCH SPACE
#         # ----------------------------------------------------

#         self.pbounds = {}

#         for i in self.attn_dims:
#             self.pbounds[f"attn_{i}"] = (0.0, 1.0)

#         for i in self.mlp_dims:
#             self.pbounds[f"mlp_{i}"] = (0.0, 1.0)

#         if len(self.pbounds) != 54:
#             raise RuntimeError(
#                 f"Expected 54 BO dimensions, "
#                 f"got {len(self.pbounds)}."
#             )

#         # ----------------------------------------------------
#         # BAYESIAN OPTIMIZATION
#         # ----------------------------------------------------

#         self.optimizer = BayesianOptimization(
#             f=self._evaluate,
#             pbounds=self.pbounds,
#             random_state=self.random_state,
#             allow_duplicate_points=True,
#         )

#         # Same GP noise setting used by the repository.
#         self.optimizer.set_gp_params(
#             alpha=1e-2
#         )

#         # ----------------------------------------------------
#         # INITIAL RESULT FILE
#         # ----------------------------------------------------

#         self._save_results()

#     # ========================================================
#     # PARAMS -> SKIP PATTERN
#     # ========================================================

#     def _params_to_skip_pattern(self, params):

#         attn_skip = sorted(
#             i
#             for i in self.attn_dims
#             if params[f"attn_{i}"] > 0.5
#         )

#         mlp_skip = sorted(
#             i
#             for i in self.mlp_dims
#             if params[f"mlp_{i}"] > 0.5
#         )

#         return {
#             "attn": attn_skip,
#             "mlp": mlp_skip,
#         }

#     # ========================================================
#     # APPLY PATTERN TO QWEN
#     # ========================================================

#     def _apply_skip_pattern(self, skip_pattern):

#         attn_set = set(skip_pattern["attn"])
#         mlp_set = set(skip_pattern["mlp"])

#         # Store the active pattern in config.
#         self.model.config.skip_layer_ids = {
#             "attn": sorted(attn_set),
#             "mlp": sorted(mlp_set),
#         }

#         # Apply the actual flags used by modeling_qwen2.py.
#         for i, layer in enumerate(self.model.model.layers):

#             if i == 0:

#                 # Layer 0 is ALWAYS full.
#                 layer.skip_attn = False
#                 layer.skip_mlp = False

#             else:

#                 layer.skip_attn = i in attn_set
#                 layer.skip_mlp = i in mlp_set

#     # ========================================================
#     # SAVE EXPERIMENT RESULTS
#     # ========================================================

#     def _save_results(self):

#         data = {
#             "experiment": {
#                 "model": MODEL_PATH,
#                 "device": DEVICE,
#                 "dtype": str(DTYPE),

#                 "num_layers": self.num_layers,

#                 "never_skip_layers": sorted(
#                     self.never_skip
#                 ),

#                 "attention_dimensions": len(
#                     self.attn_dims
#                 ),

#                 "mlp_dimensions": len(
#                     self.mlp_dims
#                 ),

#                 "search_dimensions": len(
#                     self.pbounds
#                 ),

#                 "max_new_tokens": (
#                     self.max_new_tokens
#                 ),

#                 "max_draft_tokens": (
#                     self.max_draft_tokens
#                 ),

#                 "init_points": N_INIT_POINTS,

#                 "n_iter": N_ITER,

#                 "random_state": (
#                     self.random_state
#                 ),

#                 "gamma_kwargs": (
#                     self.gamma_kwargs
#                 ),

#                 "num_dev_prompts": (
#                     len(self.dev_prompts)
#                 ),
#             },

#             "trials_completed": len(
#                 self.eval_log
#             ),

#             "trials": self.eval_log,
#         }

#         temp_file = self.results_file + ".tmp"

#         with open(
#             temp_file,
#             "w",
#             encoding="utf-8",
#         ) as f:

#             json.dump(
#                 data,
#                 f,
#                 indent=2,
#             )

#         # Atomic replacement.
#         os.replace(
#             temp_file,
#             self.results_file,
#         )

#     # ========================================================
#     # SAVE BEST PATTERN
#     # ========================================================

#     def _save_best_pattern(
#         self,
#         skip_pattern,
#         score,
#         trial_number,
#     ):

#         # IMPORTANT:
#         #
#         # Save the RAW pattern.
#         #
#         # Do NOT save:
#         #
#         # {
#         #     "skip_layer_ids": {
#         #         ...
#         #     }
#         # }
#         #
#         # because decoding.py/test code expects:
#         #
#         # {
#         #     "attn": [...],
#         #     "mlp": [...]
#         # }

#         with open(
#             self.best_pattern_file,
#             "w",
#             encoding="utf-8",
#         ) as f:

#             json.dump(
#                 skip_pattern,
#                 f,
#                 indent=2,
#             )

#         print()
#         print(
#             f"Best pattern saved to "
#             f"{self.best_pattern_file}"
#         )
#         print(
#             f"Best score: "
#             f"{score:.4f} tokens/sec"
#         )
#         print(
#             f"Best trial: "
#             f"{trial_number}"
#         )

#     # ========================================================
#     # OBJECTIVE FUNCTION
#     # ========================================================

#     def _evaluate(self, **params):

#         trial_number = len(
#             self.eval_log
#         ) + 1

#         # ----------------------------------------------------
#         # Convert BO parameters -> binary pattern
#         # ----------------------------------------------------

#         skip_pattern = (
#             self._params_to_skip_pattern(
#                 params
#             )
#         )

#         n_attn = len(
#             skip_pattern["attn"]
#         )

#         n_mlp = len(
#             skip_pattern["mlp"]
#         )

#         print()
#         print("=" * 70)
#         print(
#             f"BO TRIAL {trial_number}"
#         )
#         print("=" * 70)

#         print(
#             f"Attention skipped "
#             f"({n_attn}/27): "
#             f"{skip_pattern['attn']}"
#         )

#         print(
#             f"MLP skipped "
#             f"({n_mlp}/27): "
#             f"{skip_pattern['mlp']}"
#         )

#         print(
#             f"Total sublayers skipped: "
#             f"{n_attn + n_mlp}"
#         )

#         # ----------------------------------------------------
#         # Apply candidate
#         # ----------------------------------------------------

#         self._apply_skip_pattern(
#             skip_pattern
#         )

#         total_tokens = 0
#         total_time = 0.0

#         prompt_results = []

#         trial_start = time.perf_counter()

#         try:

#             # =================================================
#             # EVALUATE ALL DEVELOPMENT PROMPTS
#             # =================================================

#             for prompt_index, prompt in enumerate(
#                 self.dev_prompts,
#                 start=1,
#             ):

#                 inputs = self.tokenizer(
#                     prompt,
#                     return_tensors="pt",
#                 ).to(self.model.device)

#                 # ------------------------------------------------
#                 # IMPORTANT REPO-STYLE EVALUATION
#                 #
#                 # We time the COMPLETE generation call.
#                 #
#                 # Tokenization is outside the timer.
#                 #
#                 # This includes:
#                 #   - initial full prefill
#                 #   - draft generation
#                 #   - draft cache rollback
#                 #   - verification
#                 #   - acceptance/rejection
#                 #   - final rollback
#                 #   - adaptive gamma
#                 #   - Python SSD overhead
#                 #
#                 # EOS is disabled here because the repo's
#                 # evaluation uses its default early_stop=False.
#                 #
#                 # Therefore every candidate is evaluated over
#                 # the same requested generation length.
#                 # ------------------------------------------------

#                 start = time.perf_counter()

#                 _, stats = ssd_generate_tokens(
#                     self.model,
#                     inputs.input_ids,
#                     max_new_tokens=(
#                         self.max_new_tokens
#                     ),
#                     max_draft_tokens=(
#                         self.max_draft_tokens
#                     ),

#                     # Match repo's default:
#                     # early_stop=False
#                     #
#                     # Our low-level Qwen implementation expresses
#                     # this by not supplying an EOS stopping ID.
#                     eos_token_id=None,

#                     **self.gamma_kwargs,
#                 )

#                 elapsed = (
#                     time.perf_counter()
#                     - start
#                 )

#                 generated_tokens = (
#                     stats.generated_tokens
#                 )

#                 prompt_tokens_per_sec = (
#                     generated_tokens / elapsed
#                     if elapsed > 0
#                     else 0.0
#                 )

#                 total_tokens += (
#                     generated_tokens
#                 )

#                 total_time += elapsed

#                 prompt_results.append(
#                     {
#                         "prompt_index":
#                             prompt_index,

#                         "generated_tokens":
#                             generated_tokens,

#                         "time_sec":
#                             elapsed,

#                         "tokens_per_sec":
#                             prompt_tokens_per_sec,

#                         "draft_tokens":
#                             stats.draft_tokens_total,

#                         "accepted_tokens":
#                             stats.accepted_tokens_total,

#                         "acceptance_rate":
#                             stats.acceptance_rate,

#                         "average_draft_length":
#                             stats.average_draft_length,

#                         "accepted_per_verification":
#                             stats.accepted_tokens_per_verification,

#                         "verification_steps":
#                             stats.verification_steps,

#                         "gamma_start":
#                             stats.gamma_start,

#                         "gamma_final":
#                             stats.gamma_final,

#                         # Diagnostic only.
#                         "prefill_time":
#                             stats.prefill_time,

#                         "draft_time":
#                             stats.draft_time_total,

#                         "verify_time":
#                             stats.verify_time_total,

#                         "crop_before_verify":
#                             stats.crop_before_verify_time,

#                         "crop_after_verify":
#                             stats.crop_after_verify_time,

#                         "python_overhead":
#                             stats.python_overhead_time,
#                     }
#                 )

#                 print(
#                     f"Prompt "
#                     f"{prompt_index}/"
#                     f"{len(self.dev_prompts)}"
#                     f" | "
#                     f"{generated_tokens} tokens"
#                     f" | "
#                     f"{elapsed:.3f} sec"
#                     f" | "
#                     f"{prompt_tokens_per_sec:.3f} tok/s"
#                 )

#             # =================================================
#             # AGGREGATE OBJECTIVE
#             # =================================================

#             if total_time <= 0:

#                 score = -1.0

#             else:

#                 # EXACT SEARCH OBJECTIVE:
#                 #
#                 # total generated tokens
#                 # ----------------------
#                 # total generation time
#                 #
#                 # BO maximizes this value.
#                 #
#                 score = (
#                     total_tokens
#                     / total_time
#                 )

#             trial_wall_time = (
#                 time.perf_counter()
#                 - trial_start
#             )

#             # =================================================
#             # STORE TRIAL
#             # =================================================

#             trial_record = {
#                 "trial": trial_number,

#                 "params": {
#                     k: float(v)
#                     for k, v in params.items()
#                 },

#                 "skip_pattern":
#                     skip_pattern,

#                 "n_attn_skip":
#                     n_attn,

#                 "n_mlp_skip":
#                     n_mlp,

#                 "n_total_skip":
#                     n_attn + n_mlp,

#                 "total_tokens":
#                     total_tokens,

#                 "total_time":
#                     total_time,

#                 "tokens_per_sec":
#                     score,

#                 "score":
#                     score,

#                 "trial_wall_time":
#                     trial_wall_time,

#                 "prompt_results":
#                     prompt_results,

#                 "error":
#                     None,
#             }

#             self.eval_log.append(
#                 trial_record
#             )

#             # =================================================
#             # CHECK BEST
#             # =================================================

#             previous_scores = [
#                 x["score"]
#                 for x in self.eval_log[:-1]
#                 if (
#                     x.get("error") is None
#                     and x.get("score", -1) >= 0
#                 )
#             ]

#             previous_best = (
#                 max(previous_scores)
#                 if previous_scores
#                 else float("-inf")
#             )

#             if score > previous_best:

#                 self._save_best_pattern(
#                     skip_pattern=skip_pattern,
#                     score=score,
#                     trial_number=trial_number,
#                 )

#                 print()
#                 print(
#                     "*** NEW BEST PATTERN ***"
#                 )

#                 print(
#                     f"Trial: "
#                     f"{trial_number}"
#                 )

#                 print(
#                     f"Tokens/sec: "
#                     f"{score:.4f}"
#                 )

#             # =================================================
#             # SAVE EVERY TRIAL
#             # =================================================

#             self._save_results()

#             print()
#             print(
#                 f"Trial {trial_number} score: "
#                 f"{score:.4f} tokens/sec"
#             )

#             print(
#                 f"Total search trials recorded: "
#                 f"{len(self.eval_log)}"
#             )

#             return score

#         except Exception as e:

#             error_message = repr(e)

#             print()
#             print(
#                 "*** TRIAL FAILED ***"
#             )

#             print(
#                 error_message
#             )

#             trial_record = {
#                 "trial": trial_number,

#                 "params": {
#                     k: float(v)
#                     for k, v in params.items()
#                 },

#                 "skip_pattern":
#                     skip_pattern,

#                 "n_attn_skip":
#                     n_attn,

#                 "n_mlp_skip":
#                     n_mlp,

#                 "n_total_skip":
#                     n_attn + n_mlp,

#                 "total_tokens":
#                     total_tokens,

#                 "total_time":
#                     total_time,

#                 "tokens_per_sec":
#                     -1.0,

#                 "score":
#                     -1.0,

#                 "prompt_results":
#                     prompt_results,

#                 "error":
#                     error_message,
#             }

#             self.eval_log.append(
#                 trial_record
#             )

#             self._save_results()

#             return -1.0

#     # ========================================================
#     # RUN SEARCH
#     # ========================================================

#     def run(
#         self,
#         init_points=0,
#         n_iter=1000,
#     ):

#         # ----------------------------------------------------
#         # EXACTLY 1000 EVALUATIONS
#         # ----------------------------------------------------

#         if init_points + n_iter != 1000:

#             raise ValueError(
#                 "The final experiment must contain "
#                 "exactly 1000 evaluations."
#             )
#         # if init_points + n_iter <= 0:

#         #     raise ValueError(
#         #         "At least one evaluation is required."
#         #     )

#         print()
#         print("=" * 70)
#         print(
#             "STARTING BAYESIAN OPTIMIZATION"
#         )
#         print("=" * 70)

#         print(
#             f"Model              : "
#             f"{MODEL_PATH}"
#         )

#         print(
#             f"Device             : "
#             f"{DEVICE}"
#         )

#         print(
#             f"Transformer layers : "
#             f"{self.num_layers}"
#         )

#         print(
#             f"Protected layers   : "
#             f"{sorted(self.never_skip)}"
#         )

#         print(
#             f"Attention dims     : "
#             f"{len(self.attn_dims)}"
#         )

#         print(
#             f"MLP dims           : "
#             f"{len(self.mlp_dims)}"
#         )

#         print(
#             f"Total BO dims      : "
#             f"{len(self.pbounds)}"
#         )

#         print(
#             f"Dev prompts        : "
#             f"{len(self.dev_prompts)}"
#         )

#         print(
#             f"Max new tokens     : "
#             f"{self.max_new_tokens}"
#         )

#         print(
#             f"Max draft tokens   : "
#             f"{self.max_draft_tokens}"
#         )

#         print(
#             f"Initial gamma      : "
#             f"{self.gamma_kwargs.get('initial_gamma')}"
#         )

#         print(
#             f"Gamma target alpha : "
#             f"{self.gamma_kwargs.get('alpha')}"
#         )

#         print(
#             f"Initial points     : "
#             f"{init_points}"
#         )

#         print(
#             f"BO iterations      : "
#             f"{n_iter}"
#         )

#         print(
#             f"Total evaluations  : "
#             f"{init_points + n_iter}"
#         )

#         print(
#             "Objective          : "
#             "end-to-end tokens/sec"
#         )

#         print("=" * 70)
#         print()

#         # ----------------------------------------------------
#         # OFFICIAL REPO-STYLE BO CALL
#         # ----------------------------------------------------

#         self.optimizer.maximize(
#             init_points=init_points,
#             n_iter=n_iter,
#         )

#         # ----------------------------------------------------
#         # BEST BO POINT
#         # ----------------------------------------------------

#         best = self.optimizer.max

#         best_params = best["params"]

#         best_score = best["target"]

#         best_pattern = (
#             self._params_to_skip_pattern(
#                 best_params
#             )
#         )

#         # ----------------------------------------------------
#         # SAVE FINAL BEST PATTERN
#         # ----------------------------------------------------

#         self._save_best_pattern(
#             skip_pattern=best_pattern,
#             score=best_score,
#             trial_number=len(
#                 self.eval_log
#             ),
#         )

#         self._save_results()

#         # ----------------------------------------------------
#         # FINAL VALIDATION
#         # ----------------------------------------------------

#         if len(self.eval_log) != 1000:

#             raise RuntimeError(
#                 f"Expected exactly 1000 recorded "
#                 f"evaluations, got "
#                 f"{len(self.eval_log)}."
#             )

#         # ----------------------------------------------------
#         # FINAL REPORT
#         # ----------------------------------------------------

#         print()
#         print("=" * 70)
#         print(
#             "BAYESIAN OPTIMIZATION COMPLETE"
#         )
#         print("=" * 70)

#         print(
#             f"Trials completed : "
#             f"{len(self.eval_log)}"
#         )

#         print(
#             f"Best tokens/sec  : "
#             f"{best_score:.4f}"
#         )

#         print()
#         print(
#             "BEST ATTENTION SKIP LAYERS:"
#         )

#         print(
#             best_pattern["attn"]
#         )

#         print()
#         print(
#             "BEST MLP SKIP LAYERS:"
#         )

#         print(
#             best_pattern["mlp"]
#         )

#         print()
#         print(
#             f"Attention skipped: "
#             f"{len(best_pattern['attn'])}"
#         )

#         print(
#             f"MLP skipped      : "
#             f"{len(best_pattern['mlp'])}"
#         )

#         print(
#             f"Total skipped    : "
#             f"{len(best_pattern['attn']) + len(best_pattern['mlp'])}"
#         )

#         print()
#         print(
#             f"Pattern file     : "
#             f"{self.best_pattern_file}"
#         )

#         print(
#             f"Results file     : "
#             f"{self.results_file}"
#         )

#         print("=" * 70)

#         return {
#             "skip_layer_ids":
#                 best_pattern,

#             "tokens_per_sec":
#                 best_score,

#             "trials":
#                 len(self.eval_log),
#         }


# # ============================================================
# # MAIN
# # ============================================================

# if __name__ == "__main__":

#     print()
#     print("=" * 70)
#     print(
#         "QWEN2.5-1.5B INTEL CPU "
#         "BAYESIAN LAYER-SKIP SEARCH"
#     )
#     print("=" * 70)

#     # --------------------------------------------------------
#     # CPU INFORMATION
#     # --------------------------------------------------------

#     print()
#     print(
#         f"PyTorch threads       : "
#         f"{torch.get_num_threads()}"
#     )

#     print(
#         f"PyTorch interop      : "
#         f"{torch.get_num_interop_threads()}"
#     )

#     print()

#     # --------------------------------------------------------
#     # LOAD TOKENIZER
#     # --------------------------------------------------------

#     from transformers import (
#         AutoModelForCausalLM,
#         AutoTokenizer,
#     )

#     print(
#         "Loading tokenizer..."
#     )

#     tokenizer = (
#         AutoTokenizer.from_pretrained(
#             MODEL_PATH
#         )
#     )

#     # --------------------------------------------------------
#     # LOAD QWEN
#     # --------------------------------------------------------

#     print(
#         "Loading Qwen2.5-1.5B-Instruct..."
#     )

#     model = (
#         AutoModelForCausalLM.from_pretrained(
#             MODEL_PATH,
#             dtype=DTYPE,
#             device_map={"": DEVICE},
#         )
#     )

#     model.eval()

#     # --------------------------------------------------------
#     # VERIFY MODEL
#     # --------------------------------------------------------

#     print()

#     print(
#         f"Model device : "
#         f"{model.device}"
#     )

#     print(
#         f"Model dtype  : "
#         f"{next(model.parameters()).dtype}"
#     )

#     print(
#         f"Model layers : "
#         f"{model.config.num_hidden_layers}"
#     )

#     if model.config.num_hidden_layers != 28:

#         raise RuntimeError(
#             "Expected Qwen2.5-1.5B-Instruct "
#             "to have 28 transformer layers."
#         )

#     # --------------------------------------------------------
#     # PREPARE CHAT PROMPTS
#     # --------------------------------------------------------

#     dev_prompts = []

#     for prompt in DEV_PROMPTS_RAW:

#         messages = [
#             {
#                 "role": "system",
#                 "content": SYSTEM_MESSAGE,
#             },
#             {
#                 "role": "user",
#                 "content": prompt,
#             },
#         ]

#         formatted_prompt = (
#             tokenizer.apply_chat_template(
#                 messages,
#                 tokenize=False,
#                 add_generation_prompt=True,
#             )
#         )

#         dev_prompts.append(
#             formatted_prompt
#         )

#     if len(dev_prompts) != 8:

#         raise RuntimeError(
#             f"Expected 8 development prompts, "
#             f"got {len(dev_prompts)}."
#         )

#     print()
#     print(
#         f"Development prompts: "
#         f"{len(dev_prompts)}"
#     )

#     # --------------------------------------------------------
#     # CREATE SEARCH OBJECT
#     # --------------------------------------------------------

#     search = QwenLayerSkippingSearch(
#         model=model,
#         tokenizer=tokenizer,
#         dev_prompts=dev_prompts,
#         max_new_tokens=MAX_NEW_TOKENS,
#         max_draft_tokens=MAX_DRAFT_TOKENS,
#         never_skip_layers=NEVER_SKIP_LAYERS,
#         gamma_kwargs=GAMMA_KWARGS,
#         random_state=RANDOM_STATE,
#         results_file=RESULTS_FILE,
#         best_pattern_file=BEST_PATTERN_FILE,
#     )

#     # --------------------------------------------------------
#     # PRINT SEARCH SPACE
#     # --------------------------------------------------------

#     print()
#     print("=" * 70)
#     print(
#         "SEARCH SPACE VERIFIED"
#     )
#     print("=" * 70)

#     print(
#         f"Attention layers : "
#         f"{search.attn_dims}"
#     )

#     print(
#         f"MLP layers       : "
#         f"{search.mlp_dims}"
#     )

#     print(
#         f"Attention dims   : "
#         f"{len(search.attn_dims)}"
#     )

#     print(
#         f"MLP dims         : "
#         f"{len(search.mlp_dims)}"
#     )

#     print(
#         f"Total BO dims    : "
#         f"{len(search.pbounds)}"
#     )

#     print(
#         "Conversion       : "
#         "> 0.5 = SKIP"
#     )

#     print(
#         "Layer 0          : "
#         "PROTECTED"
#     )

#     print("=" * 70)

#     # --------------------------------------------------------
#     # FINAL SAFETY CHECKS
#     # --------------------------------------------------------

#     assert search.num_layers == 28

#     assert len(search.attn_dims) == 27

#     assert len(search.mlp_dims) == 27

#     assert len(search.pbounds) == 54

#     assert 0 not in search.attn_dims

#     assert 0 not in search.mlp_dims

#     assert 0 in search.never_skip

#     # --------------------------------------------------------
#     # START 1000-TRIAL SEARCH
#     # --------------------------------------------------------

#     result = search.run(
#         init_points=N_INIT_POINTS,
#         n_iter=N_ITER,
#     )

#     # --------------------------------------------------------
#     # FINAL OUTPUT
#     # --------------------------------------------------------

#     print()
#     print("=" * 70)
#     print(
#         "FINAL BEST SKIP PATTERN"
#     )
#     print("=" * 70)

#     print(
#         json.dumps(
#             result["skip_layer_ids"],
#             indent=2,
#         )
#     )

#     print()

#     print(
#         f"Best tokens/sec: "
#         f"{result['tokens_per_sec']:.4f}"
#     )

#     print(
#         f"Total evaluations: "
#         f"{result['trials']}"
#     )

#     print()
#     print(
#         "Saved files:"
#     )

#     print(
#         f"  {BEST_PATTERN_FILE}"
#     )

#     print(
#         f"  {RESULTS_FILE}"
#     )

#     print("=" * 70)

