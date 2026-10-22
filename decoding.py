"""
Qwen2.5 implementation of Draft & Verify
Self-Speculative Decoding.

Based on:

    Zhang et al.
    "Draft & Verify: Lossless Large Language Model
     Acceleration via Self-Speculative Decoding"

Official repository:
    dilab-zju/self-speculative-decoding

Qwen-specific adaptations:
    - Qwen2.5 architecture
    - Qwen-compatible DynamicCache
    - Qwen attention/MLP layer skipping
    - Layer 0 protected as cache/mask anchor

Adaptive draft-exit:
    - follows the paper's adaptive gamma equations
    - uses the official repository's practical defaults
    - initial gamma = 0.8
    - alpha = 0.9
    - beta1 = 0.5
    - beta2 = 0.9
    - epsilon = 0.01
    - max draft tokens = 8

Diagnostic timing fields are additional instrumentation only.
They do not affect the decoding decisions.
"""

import time
from dataclasses import dataclass, field

import torch
from transformers.cache_utils import DynamicCache


# ============================================================
# SSD STATISTICS
# ============================================================

@dataclass
class SSDStats:

    prompt_tokens: int = 0
    generated_tokens: int = 0

    draft_tokens_total: int = 0
    accepted_tokens_total: int = 0

    verification_steps: int = 0

    total_time: float = 0.0

    gamma_start: float = 0.0
    gamma_final: float = 0.0

    per_round: list = field(default_factory=list)

    # --------------------------------------------------------
    # Diagnostic timing.
    #
    # These are NOT part of the paper's metrics.
    # They are only used to understand Intel CPU behavior.
    # --------------------------------------------------------

    prefill_time: float = 0.0

    draft_time_total: float = 0.0

    crop_before_verify_time: float = 0.0

    verify_time_total: float = 0.0

    crop_after_verify_time: float = 0.0

    python_overhead_time: float = 0.0

    # ========================================================
    # METRICS
    # ========================================================

    @property
    def acceptance_rate(self) -> float:

        if self.draft_tokens_total == 0:
            return 0.0

        return (
            self.accepted_tokens_total
            / self.draft_tokens_total
        )

    @property
    def average_draft_length(self) -> float:

        if self.verification_steps == 0:
            return 0.0

        return (
            self.draft_tokens_total
            / self.verification_steps
        )

    @property
    def accepted_tokens_per_verification(self) -> float:

        if self.verification_steps == 0:
            return 0.0

        return (
            self.accepted_tokens_total
            / self.verification_steps
        )

    @property
    def tokens_per_sec(self) -> float:

        if self.total_time <= 0:
            return 0.0

        return (
            self.generated_tokens
            / self.total_time
        )

    # ========================================================
    # REPORT
    # ========================================================

    def report(self) -> str:

        lines = [

            "SSD generation stats",
            "---------------------",

            f"Prompt tokens:        "
            f"{self.prompt_tokens}",

            f"Generated tokens:     "
            f"{self.generated_tokens}",

            f"Draft tokens:         "
            f"{self.draft_tokens_total}",

            f"Accepted tokens:      "
            f"{self.accepted_tokens_total}",

            f"Acceptance rate:      "
            f"{self.acceptance_rate:.1%}",

            f"Average draft length: "
            f"{self.average_draft_length:.2f}",

            f"Accepted/verification:"
            f"{self.accepted_tokens_per_verification:.2f}",

            f"Verification steps:   "
            f"{self.verification_steps}",

            f"Gamma (start -> end): "
            f"{self.gamma_start:.3f} -> "
            f"{self.gamma_final:.3f}",

            f"Latency:              "
            f"{self.total_time:.3f} sec",

            f"Tokens/sec:           "
            f"{self.tokens_per_sec:.2f}",

            "",
            "--- Phase breakdown "
            "(diagnostic only) ---",

            f"Prefill:              "
            f"{self.prefill_time:.3f} sec",

            f"Draft phase total:    "
            f"{self.draft_time_total:.3f} sec"
            + (
                f"  ({1000 * self.draft_time_total / self.draft_tokens_total:.1f} "
                f"ms/draft token)"
                if self.draft_tokens_total
                else ""
            ),

            f"Crop before verify:   "
            f"{self.crop_before_verify_time:.3f} sec",

            f"Verify phase total:   "
            f"{self.verify_time_total:.3f} sec"
            + (
                f"  ({1000 * self.verify_time_total / self.verification_steps:.1f} "
                f"ms/verify pass)"
                if self.verification_steps
                else ""
            ),

            f"Crop after verify:    "
            f"{self.crop_after_verify_time:.3f} sec",

            f"Python/bookkeeping:   "
            f"{self.python_overhead_time:.3f} sec",
        ]

        return "\n".join(lines)


# ============================================================
# ADAPTIVE GAMMA
# ============================================================

def _update_gamma(
    gamma: float,
    matchness: float,
    accepted: int,
    drafted: int,
    alpha: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    max_draft_tokens: int,
    gamma_min: float,
    gamma_max: float,
):
    """
    Adaptive draft-exit threshold.

    Paper logic:

        AR_e = accepted / drafted

        AR =
            beta1 * AR
            + (1 - beta1) * AR_e

        if AR <= alpha:
            gamma_target = gamma + epsilon
        else:
            gamma_target = gamma - epsilon

        gamma =
            beta2 * gamma
            + (1 - beta2) * gamma_target

    Repository practical rule:

        If the draft already reached max_draft_tokens,
        gamma is not decreased because doing so cannot
        make the current draft longer.

    IMPORTANT:

        matchness is initialized to 0.0.

    This matches the repository's state initialization
    rather than bootstrapping the first observation directly.
    """

    if drafted == 0:

        return gamma, matchness

    # --------------------------------------------------------
    # Current-round acceptance/matchness.
    # --------------------------------------------------------

    current_matchness = (
        accepted / drafted
    )

    # --------------------------------------------------------
    # Paper Eq. (2):
    #
    # AR = beta1 * AR
    #      + (1-beta1) * AR_e
    #
    # matchness starts at 0.0.
    # --------------------------------------------------------

    matchness = (
        beta1 * matchness
        + (1.0 - beta1) * current_matchness
    )

    # --------------------------------------------------------
    # Paper Eq. (3)
    #
    # <= alpha -> increase gamma
    # >  alpha -> decrease gamma
    #
    # The official repository additionally prevents gamma
    # from decreasing when maximum draft length was already
    # reached.
    # --------------------------------------------------------

    # if matchness <= alpha:
    if matchness <= alpha:
        gamma_target = (
            gamma + epsilon
        )

    else:

        if drafted == max_draft_tokens:

            gamma_target = gamma

        else:

            gamma_target = (
                gamma - epsilon
            )

    # --------------------------------------------------------
    # Paper Eq. (4)
    # --------------------------------------------------------

    gamma = (
        beta2 * gamma
        + (1.0 - beta2) * gamma_target
    )

    # --------------------------------------------------------
    # Safety bounds.
    # --------------------------------------------------------

    gamma = min(
        max(gamma, gamma_min),
        gamma_max,
    )

    return gamma, matchness


# ============================================================
# SELF-SPECULATIVE DECODING
# ============================================================

@torch.no_grad()
def ssd_generate_tokens(
    model,
    input_ids: torch.LongTensor,
    max_new_tokens: int = 200,
    max_draft_tokens: int = 8,
    eos_token_id: int | None = None,

    # --------------------------------------------------------
    # Adaptive gamma parameters.
    #
    # These correspond to the repository configuration:
    #
    # th_stop_draft = 0.8
    # max_step_draft = 8
    # auto_parameters =
    #     [1, 0.5, 0.9, 1e-2, 0.9]
    #
    # We use:
    #
    # gamma0 = 0.8
    # beta1  = 0.5
    # alpha  = 0.9
    # epsilon = 0.01
    # beta2  = 0.9
    # --------------------------------------------------------

    initial_gamma: float = 0.8,

    alpha: float = 0.9,

    beta1: float = 0.5,

    beta2: float = 0.9,

    epsilon: float = 0.01,

    gamma_min: float = 0.0,

    gamma_max: float = 1.0,
):
    """
    Qwen2.5 self-speculative decoding.

    Pipeline:

        Full-model prefill
              ↓
        anchor token
              ↓
        draft with selected layers skipped
              ↓
        confidence-based draft exit
              ↓
        rollback draft cache
              ↓
        full-model verification
              ↓
        accept matching prefix
              ↓
        verifier bonus token
              ↓
        rollback to confirmed prefix
              ↓
        repeat
    """

    # ========================================================
    # INITIAL STATE
    # ========================================================

    stats = SSDStats(
        prompt_tokens=input_ids.shape[1],
        gamma_start=initial_gamma,
    )

    # Qwen-compatible cache.
    cache = DynamicCache(
        config=model.config
    )

    start = time.perf_counter()

    # --------------------------------------------------------
    # Adaptive gamma.
    # --------------------------------------------------------

    gamma = initial_gamma

    # IMPORTANT:
    #
    # Repository starts tmp_matchness at ZERO.
    #
    # We intentionally do NOT use:
    #
    #     matchness = None
    #
    # or:
    #
    #     matchness = first_round_acceptance
    #
    # --------------------------------------------------------

    matchness = 0.0

    # ========================================================
    # FULL MODEL PREFILL
    # ========================================================

    _t0 = time.perf_counter()

    out = model(
        input_ids=input_ids,
        use_cache=True,
        past_key_values=cache,
        draft_mode=False,
    )

    stats.prefill_time += (
        time.perf_counter() - _t0
    )

    # First token from FULL model.
    anchor = (
        out.logits[:, -1:, :]
        .argmax(-1)
    )

    generated = [
        anchor
    ]

    num_generated = 1

    finished = (
        eos_token_id is not None
        and anchor.item() == eos_token_id
    )

    # ========================================================
    # SSD LOOP
    # ========================================================

    while (
        not finished
        and num_generated < max_new_tokens
    ):

        # ----------------------------------------------------
        # Confirmed prefix length.
        #
        # Layer 0 is always executed and therefore provides
        # the cache sequence-length anchor.
        # ----------------------------------------------------

        P = cache.get_seq_length(0)

        # ----------------------------------------------------
        # Maximum number of draft tokens in this round.
        # ----------------------------------------------------

        n_draft = min(
            max_draft_tokens,
            max_new_tokens - num_generated,
        )

        if n_draft <= 0:
            break

        # ====================================================
        # DRAFT PHASE
        # ====================================================

        draft_tokens = []

        cur = anchor

        # Qwen model switches its layer behavior into
        # self-draft mode.
        with model.self_draft():

            for _ in range(n_draft):

                # ------------------------------------------------
                # Partially executed Qwen forward pass.
                # ------------------------------------------------

                _t0 = time.perf_counter()

                draft_out = model(
                    input_ids=cur,
                    use_cache=True,
                    past_key_values=cache,
                )

                stats.draft_time_total += (
                    time.perf_counter() - _t0
                )

                # ------------------------------------------------
                # Probability of the selected draft token.
                # ------------------------------------------------

                probs = torch.softmax(
                    draft_out.logits[:, -1, :],
                    dim=-1,
                )

                confidence, next_token = (
                    probs.max(dim=-1)
                )

                next_token = (
                    next_token.unsqueeze(-1)
                )

                # ------------------------------------------------
                # IMPORTANT:
                #
                # The current token is included FIRST.
                #
                # Gamma decides whether to produce ANOTHER
                # draft token.
                # ------------------------------------------------

                draft_tokens.append(
                    next_token
                )

                cur = next_token

                # ------------------------------------------------
                # Draft exit conditions.
                # ------------------------------------------------

                stop_low_confidence = (
                    confidence.item() < gamma
                )

                stop_eos = (
                    eos_token_id is not None
                    and next_token.item()
                    == eos_token_id
                )

                if (
                    stop_low_confidence
                    or stop_eos
                ):
                    break

        # ====================================================
        # DRAFT COUNT
        # ====================================================

        N = len(draft_tokens)

        if N == 0:
            break

        stats.draft_tokens_total += N

        # ====================================================
        # ROLLBACK DRAFT CACHE
        # ====================================================

        _t0 = time.perf_counter()

        cache.crop(P)

        stats.crop_before_verify_time += (
            time.perf_counter() - _t0
        )

        # ====================================================
        # FULL VERIFICATION
        # ====================================================

        draft_cat = torch.cat(
            draft_tokens,
            dim=1,
        )

        # ----------------------------------------------------
        # The verifier receives:
        #
        #       anchor + all draft tokens
        #
        # and executes ALL layers.
        # ----------------------------------------------------

        verify_input = torch.cat(
            [
                anchor,
                draft_cat,
            ],
            dim=1,
        )

        _t0 = time.perf_counter()

        verify_out = model(
            input_ids=verify_input,
            use_cache=True,
            past_key_values=cache,
            draft_mode=False,
        )

        stats.verify_time_total += (
            time.perf_counter() - _t0
        )

        stats.verification_steps += 1

        verify_preds = (
            verify_out.logits
            .argmax(-1)
        )

        # ====================================================
        # ACCEPT MATCHING PREFIX
        # ====================================================

        accepted = 0

        for i in range(N):

            verifier_token = (
                verify_preds[0, i].item()
            )

            draft_token = (
                draft_cat[0, i].item()
            )

            if verifier_token == draft_token:

                accepted += 1

            else:

                break

        stats.accepted_tokens_total += (
            accepted
        )

        # ====================================================
        # ROUND STATISTICS
        # ====================================================

        current_matchness = (
            accepted / N
        )

        stats.per_round.append(
            {
                "drafted": N,

                "accepted": accepted,

                "acceptance_rate":
                    current_matchness,

                "gamma_before":
                    gamma,
            }
        )

        # ====================================================
        # ADAPTIVE GAMMA
        # ====================================================

        gamma, matchness = _update_gamma(
            gamma=gamma,

            matchness=matchness,

            accepted=accepted,

            drafted=N,

            alpha=alpha,

            beta1=beta1,

            beta2=beta2,

            epsilon=epsilon,

            max_draft_tokens=max_draft_tokens,

            gamma_min=gamma_min,

            gamma_max=gamma_max,
        )

        stats.per_round[-1][
            "matchness_ema"
        ] = matchness

        stats.per_round[-1][
            "gamma_after"
        ] = gamma

        # ====================================================
        # BONUS TOKEN
        # ====================================================

        # If accepted = 0:
        #
        #     verify_preds[:, 0:1]
        #
        # If accepted = 3:
        #
        #     verify_preds[:, 3:4]
        #
        # This is the verifier's prediction at the first
        # mismatch position.
        # ====================================================

        bonus_token = verify_preds[
            :,
            accepted:accepted + 1,
        ]

        # ====================================================
        # FINAL CACHE ROLLBACK
        # ====================================================

        # Confirmed cache length:
        #
        # P
        # + accepted draft tokens
        # + bonus token
        #
        confirmed_length = (
            P + accepted + 1
        )

        _t0 = time.perf_counter()

        cache.crop(
            confirmed_length
        )

        stats.crop_after_verify_time += (
            time.perf_counter() - _t0
        )

        # ====================================================
        # APPEND ACCEPTED TOKENS + BONUS
        # ====================================================

        new_tokens = (
            draft_tokens[:accepted]
            + [bonus_token]
        )

        for token in new_tokens:

            generated.append(token)

            num_generated += 1

            # ------------------------------------------------
            # EOS
            # ------------------------------------------------

            if (
                eos_token_id is not None
                and token.item() == eos_token_id
            ):

                finished = True

                break

            # ------------------------------------------------
            # Generation limit
            # ------------------------------------------------

            if num_generated >= max_new_tokens:

                finished = True

                break

        # ----------------------------------------------------
        # Bonus token becomes next round's anchor.
        # ----------------------------------------------------

        anchor = bonus_token

    # ========================================================
    # FINAL STATISTICS
    # ========================================================

    stats.total_time = (
        time.perf_counter() - start
    )

    stats.generated_tokens = (
        num_generated
    )

    stats.gamma_final = gamma

    # --------------------------------------------------------
    # Remaining time = Python/bookkeeping overhead.
    # --------------------------------------------------------

    stats.python_overhead_time = (
        stats.total_time
        - (
            stats.prefill_time
            + stats.draft_time_total
            + stats.crop_before_verify_time
            + stats.verify_time_total
            + stats.crop_after_verify_time
        )
    )

    # ========================================================
    # BUILD OUTPUT
    # ========================================================

    all_generated = torch.cat(
        generated,
        dim=1,
    )[:, :max_new_tokens]

    full_sequence = torch.cat(
        [
            input_ids,
            all_generated,
        ],
        dim=1,
    )

    return full_sequence, stats


# ============================================================
# HIGH-LEVEL GENERATION WRAPPER
# ============================================================

def ssd_generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    max_draft_tokens: int = 8,
    **gamma_kwargs,
):
    """
    High-level text generation wrapper.
    """

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    eos_token_id = (
        tokenizer.eos_token_id
    )

    full_ids, stats = (
        ssd_generate_tokens(
            model=model,

            input_ids=inputs.input_ids,

            max_new_tokens=max_new_tokens,

            max_draft_tokens=max_draft_tokens,

            eos_token_id=eos_token_id,

            **gamma_kwargs,
        )
    )

    # Only decode newly generated tokens.
    new_ids = full_ids[
        :,
        inputs.input_ids.shape[1]:,
    ]

    text = tokenizer.decode(
        new_ids[0],
        skip_special_tokens=True,
    )

    return text, stats


# ============================================================
# DIRECT SANITY TEST
# ============================================================

if __name__ == "__main__":

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    MODEL_PATH = (
        "Qwen/Qwen2.5-7B-Instruct"
    )

    print()
    print("=" * 70)
    print("QWEN2.5-7B SSD SANITY TEST")
    print("=" * 70)

    # ========================================================
    # TOKENIZER
    # ========================================================

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_PATH
        )
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,

            dtype=torch.bfloat16,

            device_map={
                "": "cpu"
            },
        )
    )

    model.eval()

    print(
        "Model:",
        MODEL_PATH
    )

    print(
        "Device:",
        model.device
    )

    print(
        "Dtype:",
        next(model.parameters()).dtype
    )

    print(
        "Layers:",
        model.config.num_hidden_layers
    )

    # ========================================================
    # VERIFY MODEL
    # ========================================================

    print(
        "Layers:",
        model.config.num_hidden_layers
    )

    # ========================================================
    # TEMPORARY SKIP PATTERN
    #
    # This is ONLY a sanity-test pattern.
    #
    # It is NOT the final BO pattern.
    # ========================================================

    attn_skip = [
        4,
        8,
        12,
        16,
        20,
        24,
    ]

    mlp_skip = [
        6,
        10,
        14,
        18,
        22,
    ]

    model.config.skip_layer_ids = {
        "attn": attn_skip,
        "mlp": mlp_skip,
    }

    for i, layer in enumerate(
        model.model.layers
    ):

        # Layer 0 MUST remain full.
        if i == 0:

            layer.skip_attn = False
            layer.skip_mlp = False

        else:

            layer.skip_attn = (
                i in attn_skip
            )

            layer.skip_mlp = (
                i in mlp_skip
            )

    # ========================================================
    # PROMPT
    # ========================================================

    messages = [

        {
            "role": "system",
            "content":
                "You are Qwen, created by Alibaba Cloud. "
                "You are a helpful assistant.",
        },

        {
            "role": "user",
            "content":
                "Give me a short introduction to "
                "large language models.",
        },

    ]

    text = (
        tokenizer.apply_chat_template(
            messages,

            tokenize=False,

            add_generation_prompt=True,
        )
    )

    # ========================================================
    # RUN
    # ========================================================

    generated_text, stats = ssd_generate(

        model=model,

        tokenizer=tokenizer,

        prompt=text,

        max_new_tokens=64,

        max_draft_tokens=8,

        # Repository configuration.
        initial_gamma=0.8,

        alpha=0.9,

        beta1=0.5,

        beta2=0.9,

        epsilon=0.01,
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    print()
    print(
        "--- Generated text ---"
    )

    print(
        generated_text
    )

    print()
    print(
        "--- SSD statistics ---"
    )

    print(
        stats.report()
    )

    print()
    print(
        "--- Adaptive gamma ---"
    )

    print(
        f"Initial gamma : "
        f"{stats.gamma_start:.4f}"
    )

    print(
        f"Final gamma   : "
        f"{stats.gamma_final:.4f}"
    )

    print()
    print(
        "=" * 70
    )
    print(
        "SANITY TEST COMPLETE"
    )
    print(
        "=" * 70
    )



# qwe-1.5

# """
# Qwen2.5 implementation of Draft & Verify
# Self-Speculative Decoding.

# Based on:

#     Zhang et al.
#     "Draft & Verify: Lossless Large Language Model
#      Acceleration via Self-Speculative Decoding"

# Official repository:
#     dilab-zju/self-speculative-decoding

# Qwen-specific adaptations:
#     - Qwen2.5 architecture
#     - Qwen-compatible DynamicCache
#     - Qwen attention/MLP layer skipping
#     - Layer 0 protected as cache/mask anchor

# Adaptive draft-exit:
#     - follows the paper's adaptive gamma equations
#     - uses the official repository's practical defaults
#     - initial gamma = 0.8
#     - alpha = 0.9
#     - beta1 = 0.5
#     - beta2 = 0.9
#     - epsilon = 0.01
#     - max draft tokens = 8

# Diagnostic timing fields are additional instrumentation only.
# They do not affect the decoding decisions.
# """

# import time
# from dataclasses import dataclass, field

# import torch
# from transformers.cache_utils import DynamicCache


# # ============================================================
# # SSD STATISTICS
# # ============================================================

# @dataclass
# class SSDStats:

#     prompt_tokens: int = 0
#     generated_tokens: int = 0

#     draft_tokens_total: int = 0
#     accepted_tokens_total: int = 0

#     verification_steps: int = 0

#     total_time: float = 0.0

#     gamma_start: float = 0.0
#     gamma_final: float = 0.0

#     per_round: list = field(default_factory=list)

#     # --------------------------------------------------------
#     # Diagnostic timing.
#     #
#     # These are NOT part of the paper's metrics.
#     # They are only used to understand Intel CPU behavior.
#     # --------------------------------------------------------

#     prefill_time: float = 0.0

#     draft_time_total: float = 0.0

#     crop_before_verify_time: float = 0.0

#     verify_time_total: float = 0.0

#     crop_after_verify_time: float = 0.0

#     python_overhead_time: float = 0.0

#     # ========================================================
#     # METRICS
#     # ========================================================

#     @property
#     def acceptance_rate(self) -> float:

#         if self.draft_tokens_total == 0:
#             return 0.0

#         return (
#             self.accepted_tokens_total
#             / self.draft_tokens_total
#         )

#     @property
#     def average_draft_length(self) -> float:

#         if self.verification_steps == 0:
#             return 0.0

#         return (
#             self.draft_tokens_total
#             / self.verification_steps
#         )

#     @property
#     def accepted_tokens_per_verification(self) -> float:

#         if self.verification_steps == 0:
#             return 0.0

#         return (
#             self.accepted_tokens_total
#             / self.verification_steps
#         )

#     @property
#     def tokens_per_sec(self) -> float:

#         if self.total_time <= 0:
#             return 0.0

#         return (
#             self.generated_tokens
#             / self.total_time
#         )

#     # ========================================================
#     # REPORT
#     # ========================================================

#     def report(self) -> str:

#         lines = [

#             "SSD generation stats",
#             "---------------------",

#             f"Prompt tokens:        "
#             f"{self.prompt_tokens}",

#             f"Generated tokens:     "
#             f"{self.generated_tokens}",

#             f"Draft tokens:         "
#             f"{self.draft_tokens_total}",

#             f"Accepted tokens:      "
#             f"{self.accepted_tokens_total}",

#             f"Acceptance rate:      "
#             f"{self.acceptance_rate:.1%}",

#             f"Average draft length: "
#             f"{self.average_draft_length:.2f}",

#             f"Accepted/verification:"
#             f"{self.accepted_tokens_per_verification:.2f}",

#             f"Verification steps:   "
#             f"{self.verification_steps}",

#             f"Gamma (start -> end): "
#             f"{self.gamma_start:.3f} -> "
#             f"{self.gamma_final:.3f}",

#             f"Latency:              "
#             f"{self.total_time:.3f} sec",

#             f"Tokens/sec:           "
#             f"{self.tokens_per_sec:.2f}",

#             "",
#             "--- Phase breakdown "
#             "(diagnostic only) ---",

#             f"Prefill:              "
#             f"{self.prefill_time:.3f} sec",

#             f"Draft phase total:    "
#             f"{self.draft_time_total:.3f} sec"
#             + (
#                 f"  ({1000 * self.draft_time_total / self.draft_tokens_total:.1f} "
#                 f"ms/draft token)"
#                 if self.draft_tokens_total
#                 else ""
#             ),

#             f"Crop before verify:   "
#             f"{self.crop_before_verify_time:.3f} sec",

#             f"Verify phase total:   "
#             f"{self.verify_time_total:.3f} sec"
#             + (
#                 f"  ({1000 * self.verify_time_total / self.verification_steps:.1f} "
#                 f"ms/verify pass)"
#                 if self.verification_steps
#                 else ""
#             ),

#             f"Crop after verify:    "
#             f"{self.crop_after_verify_time:.3f} sec",

#             f"Python/bookkeeping:   "
#             f"{self.python_overhead_time:.3f} sec",
#         ]

#         return "\n".join(lines)


# # ============================================================
# # ADAPTIVE GAMMA
# # ============================================================

# def _update_gamma(
#     gamma: float,
#     matchness: float,
#     accepted: int,
#     drafted: int,
#     alpha: float,
#     beta1: float,
#     beta2: float,
#     epsilon: float,
#     max_draft_tokens: int,
#     gamma_min: float,
#     gamma_max: float,
# ):
#     """
#     Adaptive draft-exit threshold.

#     Paper logic:

#         AR_e = accepted / drafted

#         AR =
#             beta1 * AR
#             + (1 - beta1) * AR_e

#         if AR <= alpha:
#             gamma_target = gamma + epsilon
#         else:
#             gamma_target = gamma - epsilon

#         gamma =
#             beta2 * gamma
#             + (1 - beta2) * gamma_target

#     Repository practical rule:

#         If the draft already reached max_draft_tokens,
#         gamma is not decreased because doing so cannot
#         make the current draft longer.

#     IMPORTANT:

#         matchness is initialized to 0.0.

#     This matches the repository's state initialization
#     rather than bootstrapping the first observation directly.
#     """

#     if drafted == 0:

#         return gamma, matchness

#     # --------------------------------------------------------
#     # Current-round acceptance/matchness.
#     # --------------------------------------------------------

#     current_matchness = (
#         accepted / drafted
#     )

#     # --------------------------------------------------------
#     # Paper Eq. (2):
#     #
#     # AR = beta1 * AR
#     #      + (1-beta1) * AR_e
#     #
#     # matchness starts at 0.0.
#     # --------------------------------------------------------

#     matchness = (
#         beta1 * matchness
#         + (1.0 - beta1) * current_matchness
#     )

#     # --------------------------------------------------------
#     # Paper Eq. (3)
#     #
#     # <= alpha -> increase gamma
#     # >  alpha -> decrease gamma
#     #
#     # The official repository additionally prevents gamma
#     # from decreasing when maximum draft length was already
#     # reached.
#     # --------------------------------------------------------

#     if matchness <= alpha:

#         gamma_target = (
#             gamma + epsilon
#         )

#     else:

#         if drafted == max_draft_tokens:

#             gamma_target = gamma

#         else:

#             gamma_target = (
#                 gamma - epsilon
#             )

#     # --------------------------------------------------------
#     # Paper Eq. (4)
#     # --------------------------------------------------------

#     gamma = (
#         beta2 * gamma
#         + (1.0 - beta2) * gamma_target
#     )

#     # --------------------------------------------------------
#     # Safety bounds.
#     # --------------------------------------------------------

#     gamma = min(
#         max(gamma, gamma_min),
#         gamma_max,
#     )

#     return gamma, matchness


# # ============================================================
# # SELF-SPECULATIVE DECODING
# # ============================================================

# @torch.no_grad()
# def ssd_generate_tokens(
#     model,
#     input_ids: torch.LongTensor,
#     max_new_tokens: int = 200,
#     max_draft_tokens: int = 8,
#     eos_token_id: int | None = None,

#     # --------------------------------------------------------
#     # Adaptive gamma parameters.
#     #
#     # These correspond to the repository configuration:
#     #
#     # th_stop_draft = 0.8
#     # max_step_draft = 8
#     # auto_parameters =
#     #     [1, 0.5, 0.9, 1e-2, 0.9]
#     #
#     # We use:
#     #
#     # gamma0 = 0.8
#     # beta1  = 0.5
#     # alpha  = 0.9
#     # epsilon = 0.01
#     # beta2  = 0.9
#     # --------------------------------------------------------

#     initial_gamma: float = 0.8,

#     alpha: float = 0.9,

#     beta1: float = 0.5,

#     beta2: float = 0.9,

#     epsilon: float = 0.01,

#     gamma_min: float = 0.0,

#     gamma_max: float = 1.0,
# ):
#     """
#     Qwen2.5 self-speculative decoding.

#     Pipeline:

#         Full-model prefill
#               ↓
#         anchor token
#               ↓
#         draft with selected layers skipped
#               ↓
#         confidence-based draft exit
#               ↓
#         rollback draft cache
#               ↓
#         full-model verification
#               ↓
#         accept matching prefix
#               ↓
#         verifier bonus token
#               ↓
#         rollback to confirmed prefix
#               ↓
#         repeat
#     """

#     # ========================================================
#     # INITIAL STATE
#     # ========================================================

#     stats = SSDStats(
#         prompt_tokens=input_ids.shape[1],
#         gamma_start=initial_gamma,
#     )

#     # Qwen-compatible cache.
#     cache = DynamicCache(
#         config=model.config
#     )

#     start = time.perf_counter()

#     # --------------------------------------------------------
#     # Adaptive gamma.
#     # --------------------------------------------------------

#     gamma = initial_gamma

#     # IMPORTANT:
#     #
#     # Repository starts tmp_matchness at ZERO.
#     #
#     # We intentionally do NOT use:
#     #
#     #     matchness = None
#     #
#     # or:
#     #
#     #     matchness = first_round_acceptance
#     #
#     # --------------------------------------------------------

#     matchness = 0.0

#     # ========================================================
#     # FULL MODEL PREFILL
#     # ========================================================

#     _t0 = time.perf_counter()

#     out = model(
#         input_ids=input_ids,
#         use_cache=True,
#         past_key_values=cache,
#         draft_mode=False,
#     )

#     stats.prefill_time += (
#         time.perf_counter() - _t0
#     )

#     # First token from FULL model.
#     anchor = (
#         out.logits[:, -1:, :]
#         .argmax(-1)
#     )

#     generated = [
#         anchor
#     ]

#     num_generated = 1

#     finished = (
#         eos_token_id is not None
#         and anchor.item() == eos_token_id
#     )

#     # ========================================================
#     # SSD LOOP
#     # ========================================================

#     while (
#         not finished
#         and num_generated < max_new_tokens
#     ):

#         # ----------------------------------------------------
#         # Confirmed prefix length.
#         #
#         # Layer 0 is always executed and therefore provides
#         # the cache sequence-length anchor.
#         # ----------------------------------------------------

#         P = cache.get_seq_length(0)

#         # ----------------------------------------------------
#         # Maximum number of draft tokens in this round.
#         # ----------------------------------------------------

#         n_draft = min(
#             max_draft_tokens,
#             max_new_tokens - num_generated,
#         )

#         if n_draft <= 0:
#             break

#         # ====================================================
#         # DRAFT PHASE
#         # ====================================================

#         draft_tokens = []

#         cur = anchor

#         # Qwen model switches its layer behavior into
#         # self-draft mode.
#         with model.self_draft():

#             for _ in range(n_draft):

#                 # ------------------------------------------------
#                 # Partially executed Qwen forward pass.
#                 # ------------------------------------------------

#                 _t0 = time.perf_counter()

#                 draft_out = model(
#                     input_ids=cur,
#                     use_cache=True,
#                     past_key_values=cache,
#                 )

#                 stats.draft_time_total += (
#                     time.perf_counter() - _t0
#                 )

#                 # ------------------------------------------------
#                 # Probability of the selected draft token.
#                 # ------------------------------------------------

#                 probs = torch.softmax(
#                     draft_out.logits[:, -1, :],
#                     dim=-1,
#                 )

#                 confidence, next_token = (
#                     probs.max(dim=-1)
#                 )

#                 next_token = (
#                     next_token.unsqueeze(-1)
#                 )

#                 # ------------------------------------------------
#                 # IMPORTANT:
#                 #
#                 # The current token is included FIRST.
#                 #
#                 # Gamma decides whether to produce ANOTHER
#                 # draft token.
#                 # ------------------------------------------------

#                 draft_tokens.append(
#                     next_token
#                 )

#                 cur = next_token

#                 # ------------------------------------------------
#                 # Draft exit conditions.
#                 # ------------------------------------------------

#                 stop_low_confidence = (
#                     confidence.item() < gamma
#                 )

#                 stop_eos = (
#                     eos_token_id is not None
#                     and next_token.item()
#                     == eos_token_id
#                 )

#                 if (
#                     stop_low_confidence
#                     or stop_eos
#                 ):
#                     break

#         # ====================================================
#         # DRAFT COUNT
#         # ====================================================

#         N = len(draft_tokens)

#         if N == 0:
#             break

#         stats.draft_tokens_total += N

#         # ====================================================
#         # ROLLBACK DRAFT CACHE
#         # ====================================================

#         _t0 = time.perf_counter()

#         cache.crop(P)

#         stats.crop_before_verify_time += (
#             time.perf_counter() - _t0
#         )

#         # ====================================================
#         # FULL VERIFICATION
#         # ====================================================

#         draft_cat = torch.cat(
#             draft_tokens,
#             dim=1,
#         )

#         # ----------------------------------------------------
#         # The verifier receives:
#         #
#         #       anchor + all draft tokens
#         #
#         # and executes ALL layers.
#         # ----------------------------------------------------

#         verify_input = torch.cat(
#             [
#                 anchor,
#                 draft_cat,
#             ],
#             dim=1,
#         )

#         _t0 = time.perf_counter()

#         verify_out = model(
#             input_ids=verify_input,
#             use_cache=True,
#             past_key_values=cache,
#             draft_mode=False,
#         )

#         stats.verify_time_total += (
#             time.perf_counter() - _t0
#         )

#         stats.verification_steps += 1

#         verify_preds = (
#             verify_out.logits
#             .argmax(-1)
#         )

#         # ====================================================
#         # ACCEPT MATCHING PREFIX
#         # ====================================================

#         accepted = 0

#         for i in range(N):

#             verifier_token = (
#                 verify_preds[0, i].item()
#             )

#             draft_token = (
#                 draft_cat[0, i].item()
#             )

#             if verifier_token == draft_token:

#                 accepted += 1

#             else:

#                 break

#         stats.accepted_tokens_total += (
#             accepted
#         )

#         # ====================================================
#         # ROUND STATISTICS
#         # ====================================================

#         current_matchness = (
#             accepted / N
#         )

#         stats.per_round.append(
#             {
#                 "drafted": N,

#                 "accepted": accepted,

#                 "acceptance_rate":
#                     current_matchness,

#                 "gamma_before":
#                     gamma,
#             }
#         )

#         # ====================================================
#         # ADAPTIVE GAMMA
#         # ====================================================

#         gamma, matchness = _update_gamma(
#             gamma=gamma,

#             matchness=matchness,

#             accepted=accepted,

#             drafted=N,

#             alpha=alpha,

#             beta1=beta1,

#             beta2=beta2,

#             epsilon=epsilon,

#             max_draft_tokens=max_draft_tokens,

#             gamma_min=gamma_min,

#             gamma_max=gamma_max,
#         )

#         stats.per_round[-1][
#             "matchness_ema"
#         ] = matchness

#         stats.per_round[-1][
#             "gamma_after"
#         ] = gamma

#         # ====================================================
#         # BONUS TOKEN
#         # ====================================================

#         # If accepted = 0:
#         #
#         #     verify_preds[:, 0:1]
#         #
#         # If accepted = 3:
#         #
#         #     verify_preds[:, 3:4]
#         #
#         # This is the verifier's prediction at the first
#         # mismatch position.
#         # ====================================================

#         bonus_token = verify_preds[
#             :,
#             accepted:accepted + 1,
#         ]

#         # ====================================================
#         # FINAL CACHE ROLLBACK
#         # ====================================================

#         # Confirmed cache length:
#         #
#         # P
#         # + accepted draft tokens
#         # + bonus token
#         #
#         confirmed_length = (
#             P + accepted + 1
#         )

#         _t0 = time.perf_counter()

#         cache.crop(
#             confirmed_length
#         )

#         stats.crop_after_verify_time += (
#             time.perf_counter() - _t0
#         )

#         # ====================================================
#         # APPEND ACCEPTED TOKENS + BONUS
#         # ====================================================

#         new_tokens = (
#             draft_tokens[:accepted]
#             + [bonus_token]
#         )

#         for token in new_tokens:

#             generated.append(token)

#             num_generated += 1

#             # ------------------------------------------------
#             # EOS
#             # ------------------------------------------------

#             if (
#                 eos_token_id is not None
#                 and token.item() == eos_token_id
#             ):

#                 finished = True

#                 break

#             # ------------------------------------------------
#             # Generation limit
#             # ------------------------------------------------

#             if num_generated >= max_new_tokens:

#                 finished = True

#                 break

#         # ----------------------------------------------------
#         # Bonus token becomes next round's anchor.
#         # ----------------------------------------------------

#         anchor = bonus_token

#     # ========================================================
#     # FINAL STATISTICS
#     # ========================================================

#     stats.total_time = (
#         time.perf_counter() - start
#     )

#     stats.generated_tokens = (
#         num_generated
#     )

#     stats.gamma_final = gamma

#     # --------------------------------------------------------
#     # Remaining time = Python/bookkeeping overhead.
#     # --------------------------------------------------------

#     stats.python_overhead_time = (
#         stats.total_time
#         - (
#             stats.prefill_time
#             + stats.draft_time_total
#             + stats.crop_before_verify_time
#             + stats.verify_time_total
#             + stats.crop_after_verify_time
#         )
#     )

#     # ========================================================
#     # BUILD OUTPUT
#     # ========================================================

#     all_generated = torch.cat(
#         generated,
#         dim=1,
#     )[:, :max_new_tokens]

#     full_sequence = torch.cat(
#         [
#             input_ids,
#             all_generated,
#         ],
#         dim=1,
#     )

#     return full_sequence, stats


# # ============================================================
# # HIGH-LEVEL GENERATION WRAPPER
# # ============================================================

# def ssd_generate(
#     model,
#     tokenizer,
#     prompt: str,
#     max_new_tokens: int = 200,
#     max_draft_tokens: int = 8,
#     **gamma_kwargs,
# ):
#     """
#     High-level text generation wrapper.
#     """

#     inputs = tokenizer(
#         prompt,
#         return_tensors="pt",
#     ).to(model.device)

#     eos_token_id = (
#         tokenizer.eos_token_id
#     )

#     full_ids, stats = (
#         ssd_generate_tokens(
#             model=model,

#             input_ids=inputs.input_ids,

#             max_new_tokens=max_new_tokens,

#             max_draft_tokens=max_draft_tokens,

#             eos_token_id=eos_token_id,

#             **gamma_kwargs,
#         )
#     )

#     # Only decode newly generated tokens.
#     new_ids = full_ids[
#         :,
#         inputs.input_ids.shape[1]:,
#     ]

#     text = tokenizer.decode(
#         new_ids[0],
#         skip_special_tokens=True,
#     )

#     return text, stats


# # ============================================================
# # DIRECT SANITY TEST
# # ============================================================

# if __name__ == "__main__":

#     from transformers import (
#         AutoModelForCausalLM,
#         AutoTokenizer,
#     )

#     MODEL_PATH = (
#         "Qwen/Qwen2.5-1.5B-Instruct"
#     )

#     print()
#     print("=" * 70)
#     print("QWEN2.5-1.5B SSD SANITY TEST")
#     print("=" * 70)

#     # ========================================================
#     # TOKENIZER
#     # ========================================================

#     tokenizer = (
#         AutoTokenizer.from_pretrained(
#             MODEL_PATH
#         )
#     )

#     # ========================================================
#     # MODEL
#     # ========================================================

#     model = (
#         AutoModelForCausalLM.from_pretrained(
#             MODEL_PATH,

#             dtype=torch.float32,

#             device_map={
#                 "": "cpu"
#             },
#         )
#     )

#     model.eval()

#     print(
#         "Model:",
#         MODEL_PATH
#     )

#     print(
#         "Device:",
#         model.device
#     )

#     print(
#         "Dtype:",
#         next(model.parameters()).dtype
#     )

#     print(
#         "Layers:",
#         model.config.num_hidden_layers
#     )

#     # ========================================================
#     # VERIFY MODEL
#     # ========================================================

#     if model.config.num_hidden_layers != 28:

#         raise RuntimeError(
#             "Expected Qwen2.5-1.5B-Instruct "
#             "to have 28 transformer layers."
#         )

#     # ========================================================
#     # TEMPORARY SKIP PATTERN
#     #
#     # This is ONLY a sanity-test pattern.
#     #
#     # It is NOT the final BO pattern.
#     # ========================================================

#     attn_skip = [
#         4,
#         8,
#         12,
#         16,
#         20,
#         24,
#     ]

#     mlp_skip = [
#         6,
#         10,
#         14,
#         18,
#         22,
#     ]

#     model.config.skip_layer_ids = {
#         "attn": attn_skip,
#         "mlp": mlp_skip,
#     }

#     for i, layer in enumerate(
#         model.model.layers
#     ):

#         # Layer 0 MUST remain full.
#         if i == 0:

#             layer.skip_attn = False
#             layer.skip_mlp = False

#         else:

#             layer.skip_attn = (
#                 i in attn_skip
#             )

#             layer.skip_mlp = (
#                 i in mlp_skip
#             )

#     # ========================================================
#     # PROMPT
#     # ========================================================

#     messages = [

#         {
#             "role": "system",
#             "content":
#                 "You are Qwen, created by Alibaba Cloud. "
#                 "You are a helpful assistant.",
#         },

#         {
#             "role": "user",
#             "content":
#                 "Give me a short introduction to "
#                 "large language models.",
#         },

#     ]

#     text = (
#         tokenizer.apply_chat_template(
#             messages,

#             tokenize=False,

#             add_generation_prompt=True,
#         )
#     )

#     # ========================================================
#     # RUN
#     # ========================================================

#     generated_text, stats = ssd_generate(

#         model=model,

#         tokenizer=tokenizer,

#         prompt=text,

#         max_new_tokens=64,

#         max_draft_tokens=8,

#         # Repository configuration.
#         initial_gamma=0.8,

#         alpha=0.9,

#         beta1=0.5,

#         beta2=0.9,

#         epsilon=0.01,
#     )

#     # ========================================================
#     # OUTPUT
#     # ========================================================

#     print()
#     print(
#         "--- Generated text ---"
#     )

#     print(
#         generated_text
#     )

#     print()
#     print(
#         "--- SSD statistics ---"
#     )

#     print(
#         stats.report()
#     )

#     print()
#     print(
#         "--- Adaptive gamma ---"
#     )

#     print(
#         f"Initial gamma : "
#         f"{stats.gamma_start:.4f}"
#     )

#     print(
#         f"Final gamma   : "
#         f"{stats.gamma_final:.4f}"
#     )

#     print()
#     print(
#         "=" * 70
#     )
#     print(
#         "SANITY TEST COMPLETE"
#     )
#     print(
#         "=" * 70
#     )



# /home/ziroh/ritu/ritu-ssd/decoding_draftlendebug.py



"""
Qwen2.5 implementation of Draft & Verify
Self-Speculative Decoding.

Based on:

    Zhang et al.
    "Draft & Verify: Lossless Large Language Model
     Acceleration via Self-Speculative Decoding"

Official repository:
    dilab-zju/self-speculative-decoding

Qwen-specific adaptations:
    - Qwen2.5 architecture
    - Qwen-compatible DynamicCache
    - Qwen attention/MLP layer skipping
    - Layer 0 protected as cache/mask anchor

Adaptive draft-exit:
    - initial gamma = 0.8
    - alpha = 0.9
    - beta1 = 0.5
    - beta2 = 0.9
    - epsilon = 0.01
    - max draft tokens = 8

IMPORTANT:
    This version adds diagnostic instrumentation to determine
    WHY draft rounds terminate.

The diagnostics do NOT alter decoding decisions.
"""


import time

from dataclasses import dataclass, field

import torch
from transformers.cache_utils import DynamicCache


# ============================================================
# SSD STATISTICS
# ============================================================

@dataclass
class SSDStats:

    # --------------------------------------------------------
    # Basic generation statistics
    # --------------------------------------------------------

    prompt_tokens: int = 0

    generated_tokens: int = 0

    draft_tokens_total: int = 0

    accepted_tokens_total: int = 0

    verification_steps: int = 0

    total_time: float = 0.0

    # --------------------------------------------------------
    # Adaptive gamma
    # --------------------------------------------------------

    gamma_start: float = 0.0

    gamma_final: float = 0.0

    # --------------------------------------------------------
    # Per-round statistics
    # --------------------------------------------------------

    per_round: list = field(
        default_factory=list
    )

    # ========================================================
    # DRAFT STOP DIAGNOSTICS
    # ========================================================

    # Number of rounds that ended because the confidence
    # of the latest draft token was below gamma.
    low_confidence_stops: int = 0

    # Number of rounds that reached max_draft_tokens.
    max_draft_stops: int = 0

    # Number of rounds that stopped because EOS was generated.
    eos_stops: int = 0

    # Number of rounds that reached neither condition.
    other_stops: int = 0

    # --------------------------------------------------------
    # Confidence statistics
    # --------------------------------------------------------

    confidence_sum: float = 0.0

    confidence_count: int = 0

    stop_confidence_sum: float = 0.0

    stop_confidence_count: int = 0

    # --------------------------------------------------------
    # Draft-length histogram.
    #
    # Example:
    #
    # {
    #     "1": 100,
    #     "2": 500,
    #     "3": 1000,
    #     ...
    # }
    # --------------------------------------------------------

    draft_length_histogram: dict = field(
        default_factory=dict
    )

    # ========================================================
    # Diagnostic timing
    # ========================================================

    prefill_time: float = 0.0

    draft_time_total: float = 0.0

    crop_before_verify_time: float = 0.0

    verify_time_total: float = 0.0

    crop_after_verify_time: float = 0.0

    python_overhead_time: float = 0.0

    # ========================================================
    # METRICS
    # ========================================================

    @property
    def acceptance_rate(self) -> float:

        if self.draft_tokens_total == 0:
            return 0.0

        return (
            self.accepted_tokens_total
            / self.draft_tokens_total
        )

    @property
    def average_draft_length(self) -> float:

        if self.verification_steps == 0:
            return 0.0

        return (
            self.draft_tokens_total
            / self.verification_steps
        )

    @property
    def accepted_tokens_per_verification(self) -> float:

        if self.verification_steps == 0:
            return 0.0

        return (
            self.accepted_tokens_total
            / self.verification_steps
        )

    @property
    def tokens_per_sec(self) -> float:

        if self.total_time <= 0:
            return 0.0

        return (
            self.generated_tokens
            / self.total_time
        )

    @property
    def average_confidence(self) -> float:

        if self.confidence_count == 0:
            return 0.0

        return (
            self.confidence_sum
            / self.confidence_count
        )

    @property
    def average_stop_confidence(self) -> float:

        if self.stop_confidence_count == 0:
            return 0.0

        return (
            self.stop_confidence_sum
            / self.stop_confidence_count
        )

    # ========================================================
    # REPORT
    # ========================================================

    def report(self) -> str:

        lines = [

            "SSD generation stats",
            "---------------------",

            f"Prompt tokens:        "
            f"{self.prompt_tokens}",

            f"Generated tokens:     "
            f"{self.generated_tokens}",

            f"Draft tokens:         "
            f"{self.draft_tokens_total}",

            f"Accepted tokens:      "
            f"{self.accepted_tokens_total}",

            f"Acceptance rate:      "
            f"{self.acceptance_rate:.1%}",

            f"Average draft length: "
            f"{self.average_draft_length:.2f}",

            f"Accepted/verification:"
            f"{self.accepted_tokens_per_verification:.2f}",

            f"Verification steps:   "
            f"{self.verification_steps}",

            f"Gamma (start -> end): "
            f"{self.gamma_start:.3f} -> "
            f"{self.gamma_final:.3f}",

            f"Latency:              "
            f"{self.total_time:.3f} sec",

            f"Tokens/sec:           "
            f"{self.tokens_per_sec:.2f}",

            "",
            "--- Draft stopping diagnostics ---",

            f"Low-confidence stops: "
            f"{self.low_confidence_stops}",

            f"Max-draft stops:       "
            f"{self.max_draft_stops}",

            f"EOS stops:             "
            f"{self.eos_stops}",

            f"Other stops:           "
            f"{self.other_stops}",

            f"Average confidence:    "
            f"{self.average_confidence:.4f}",

            f"Average stop confidence:"
            f" {self.average_stop_confidence:.4f}",

            "",
            "Draft-length histogram:",
            str(
                dict(
                    sorted(
                        self.draft_length_histogram.items(),
                        key=lambda x: int(x[0])
                    )
                )
            ),

            "",
            "--- Phase breakdown "
            "(diagnostic only) ---",

            f"Prefill:              "
            f"{self.prefill_time:.3f} sec",

            f"Draft phase total:    "
            f"{self.draft_time_total:.3f} sec"
            + (
                f"  ({1000 * self.draft_time_total / self.draft_tokens_total:.1f} "
                f"ms/draft token)"
                if self.draft_tokens_total
                else ""
            ),

            f"Crop before verify:   "
            f"{self.crop_before_verify_time:.3f} sec",

            f"Verify phase total:   "
            f"{self.verify_time_total:.3f} sec"
            + (
                f"  ({1000 * self.verify_time_total / self.verification_steps:.1f} "
                f"ms/verify pass)"
                if self.verification_steps
                else ""
            ),

            f"Crop after verify:    "
            f"{self.crop_after_verify_time:.3f} sec",

            f"Python/bookkeeping:   "
            f"{self.python_overhead_time:.3f} sec",
        ]

        return "\n".join(lines)


# ============================================================
# ADAPTIVE GAMMA
# ============================================================

def _update_gamma(
    gamma: float,
    matchness: float,
    accepted: int,
    drafted: int,
    alpha: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    max_draft_tokens: int,
    gamma_min: float,
    gamma_max: float,
):
    """
    Adaptive draft-exit threshold.

    Paper logic:

        AR_e = accepted / drafted

        AR =
            beta1 * AR
            + (1 - beta1) * AR_e

        if AR <= alpha:
            gamma_target = gamma + epsilon
        else:
            gamma_target = gamma - epsilon

        gamma =
            beta2 * gamma
            + (1 - beta2) * gamma_target

    Repository practical rule:

        If the draft already reached max_draft_tokens,
        gamma is not decreased.
    """

    if drafted == 0:

        return gamma, matchness

    current_matchness = (
        accepted / drafted
    )

    matchness = (
        beta1 * matchness
        + (1.0 - beta1) * current_matchness
    )

    if matchness <= alpha:

        gamma_target = (
            gamma + epsilon
        )

    else:

        if drafted == max_draft_tokens:

            gamma_target = gamma

        else:

            gamma_target = (
                gamma - epsilon
            )

    gamma = (
        beta2 * gamma
        + (1.0 - beta2) * gamma_target
    )

    gamma = min(
        max(gamma, gamma_min),
        gamma_max,
    )

    return gamma, matchness


# ============================================================
# SELF-SPECULATIVE DECODING
# ============================================================

@torch.no_grad()
def ssd_generate_tokens(
    model,
    input_ids: torch.LongTensor,
    max_new_tokens: int = 200,
    max_draft_tokens: int = 8,
    eos_token_id: int | None = None,

    initial_gamma: float = 0.8,

    alpha: float = 0.9,

    beta1: float = 0.5,

    beta2: float = 0.9,

    epsilon: float = 0.01,

    gamma_min: float = 0.0,

    gamma_max: float = 1.0,
):
    """
    Qwen2.5 self-speculative decoding.

    Diagnostic additions:

        - low-confidence stop count
        - max-draft stop count
        - EOS stop count
        - confidence of every draft token
        - confidence of stopping token
        - draft-length histogram
        - per-round stop reason

    These diagnostics do NOT modify the decoding algorithm.
    """

    # ========================================================
    # INITIAL STATE
    # ========================================================

    stats = SSDStats(
        prompt_tokens=input_ids.shape[1],
        gamma_start=initial_gamma,
    )

    cache = DynamicCache(
        config=model.config
    )

    start = time.perf_counter()

    gamma = initial_gamma

    # Repository starts matchness at zero.
    matchness = 0.0

    # ========================================================
    # FULL MODEL PREFILL
    # ========================================================

    _t0 = time.perf_counter()

    out = model(
        input_ids=input_ids,
        use_cache=True,
        past_key_values=cache,
        draft_mode=False,
    )

    stats.prefill_time += (
        time.perf_counter() - _t0
    )

    # First token comes from the FULL model.
    anchor = (
        out.logits[:, -1:, :]
        .argmax(-1)
    )

    generated = [
        anchor
    ]

    num_generated = 1

    finished = (
        eos_token_id is not None
        and anchor.item() == eos_token_id
    )

    # ========================================================
    # SSD LOOP
    # ========================================================

    while (
        not finished
        and num_generated < max_new_tokens
    ):

        # ----------------------------------------------------
        # Confirmed prefix length.
        # ----------------------------------------------------

        P = cache.get_seq_length(0)

        # ----------------------------------------------------
        # Number of draft tokens allowed this round.
        # ----------------------------------------------------

        n_draft = min(
            max_draft_tokens,
            max_new_tokens - num_generated,
        )

        if n_draft <= 0:
            break

        # ====================================================
        # DRAFT PHASE
        # ====================================================

        draft_tokens = []

        cur = anchor

        # Diagnostics for this round.
        round_confidences = []

        stop_reason = None

        stop_confidence = None

        # ----------------------------------------------------
        # Partial-model draft.
        # ----------------------------------------------------

        with model.self_draft():

            for draft_step in range(
                n_draft
            ):

                _t0 = time.perf_counter()

                draft_out = model(
                    input_ids=cur,
                    use_cache=True,
                    past_key_values=cache,
                )

                stats.draft_time_total += (
                    time.perf_counter() - _t0
                )

                # ------------------------------------------------
                # Greedy draft token.
                # ------------------------------------------------

                probs = torch.softmax(
                    draft_out.logits[:, -1, :],
                    dim=-1,
                )

                confidence, next_token = (
                    probs.max(dim=-1)
                )

                confidence_value = (
                    confidence.item()
                )

                next_token = (
                    next_token.unsqueeze(-1)
                )

                # ------------------------------------------------
                # Record confidence.
                # ------------------------------------------------

                round_confidences.append(
                    confidence_value
                )

                stats.confidence_sum += (
                    confidence_value
                )

                stats.confidence_count += 1

                # ------------------------------------------------
                # Add token FIRST.
                # ------------------------------------------------

                draft_tokens.append(
                    next_token
                )

                cur = next_token

                # ------------------------------------------------
                # Determine why this round stops.
                # ------------------------------------------------

                stop_low_confidence = (
                    confidence_value < gamma
                )

                stop_eos = (
                    eos_token_id is not None
                    and next_token.item()
                    == eos_token_id
                )

                reached_max_draft = (
                    draft_step + 1
                    >= n_draft
                )

                # ------------------------------------------------
                # Priority:
                #
                # EOS
                #   -> eos
                #
                # confidence < gamma
                #   -> low_confidence
                #
                # max draft
                #   -> max_draft
                # ------------------------------------------------

                if stop_eos:

                    stop_reason = "eos"

                    stop_confidence = (
                        confidence_value
                    )

                    break

                if stop_low_confidence:

                    stop_reason = (
                        "low_confidence"
                    )

                    stop_confidence = (
                        confidence_value
                    )

                    break

                if reached_max_draft:

                    stop_reason = (
                        "max_draft"
                    )

                    stop_confidence = (
                        confidence_value
                    )

                    break

        # ====================================================
        # DRAFT COUNT
        # ====================================================

        N = len(draft_tokens)

        if N == 0:

            stop_reason = "empty"

            stats.other_stops += 1

            break

        stats.draft_tokens_total += N

        # ====================================================
        # UPDATE STOP DIAGNOSTICS
        # ====================================================

        if stop_reason == "low_confidence":

            stats.low_confidence_stops += 1

        elif stop_reason == "max_draft":

            stats.max_draft_stops += 1

        elif stop_reason == "eos":

            stats.eos_stops += 1

        else:

            stats.other_stops += 1

        # ----------------------------------------------------
        # Stop confidence.
        # ----------------------------------------------------

        if stop_confidence is not None:

            stats.stop_confidence_sum += (
                stop_confidence
            )

            stats.stop_confidence_count += 1

        # ----------------------------------------------------
        # Draft length histogram.
        # ----------------------------------------------------

        length_key = str(N)

        stats.draft_length_histogram[
            length_key
        ] = (
            stats.draft_length_histogram.get(
                length_key,
                0,
            )
            + 1
        )

        # ====================================================
        # ROLLBACK DRAFT CACHE
        # ====================================================

        _t0 = time.perf_counter()

        cache.crop(P)

        stats.crop_before_verify_time += (
            time.perf_counter() - _t0
        )

        # ====================================================
        # FULL VERIFICATION
        # ====================================================

        draft_cat = torch.cat(
            draft_tokens,
            dim=1,
        )

        # ----------------------------------------------------
        # Verifier sees:
        #
        # anchor + draft tokens
        # ----------------------------------------------------

        verify_input = torch.cat(
            [
                anchor,
                draft_cat,
            ],
            dim=1,
        )

        _t0 = time.perf_counter()

        verify_out = model(
            input_ids=verify_input,
            use_cache=True,
            past_key_values=cache,
            draft_mode=False,
        )

        stats.verify_time_total += (
            time.perf_counter() - _t0
        )

        stats.verification_steps += 1

        verify_preds = (
            verify_out.logits
            .argmax(-1)
        )

        # ====================================================
        # ACCEPT MATCHING PREFIX
        # ====================================================

        accepted = 0

        for i in range(N):

            verifier_token = (
                verify_preds[
                    0,
                    i
                ].item()
            )

            draft_token = (
                draft_cat[
                    0,
                    i
                ].item()
            )

            if (
                verifier_token
                == draft_token
            ):

                accepted += 1

            else:

                break

        stats.accepted_tokens_total += (
            accepted
        )

        # ====================================================
        # ROUND STATISTICS
        # ====================================================

        current_matchness = (
            accepted / N
        )

        round_record = {

            "drafted": N,

            "accepted": accepted,

            "rejected": N - accepted,

            "acceptance_rate":
                current_matchness,

            "gamma_before":
                gamma,

            "stop_reason":
                stop_reason,

            "stop_confidence":
                stop_confidence,

            "draft_confidences":
                round_confidences,
        }

        stats.per_round.append(
            round_record
        )

        # ====================================================
        # ADAPTIVE GAMMA
        # ====================================================

        gamma, matchness = _update_gamma(

            gamma=gamma,

            matchness=matchness,

            accepted=accepted,

            drafted=N,

            alpha=alpha,

            beta1=beta1,

            beta2=beta2,

            epsilon=epsilon,

            max_draft_tokens=max_draft_tokens,

            gamma_min=gamma_min,

            gamma_max=gamma_max,
        )

        stats.per_round[-1][
            "matchness_ema"
        ] = matchness

        stats.per_round[-1][
            "gamma_after"
        ] = gamma

        # ====================================================
        # BONUS TOKEN
        # ====================================================

        bonus_token = verify_preds[
            :,
            accepted:accepted + 1,
        ]

        # ====================================================
        # FINAL CACHE ROLLBACK
        # ====================================================

        confirmed_length = (
            P
            + accepted
            + 1
        )

        _t0 = time.perf_counter()

        cache.crop(
            confirmed_length
        )

        stats.crop_after_verify_time += (
            time.perf_counter() - _t0
        )

        # ====================================================
        # APPEND ACCEPTED TOKENS + BONUS
        # ====================================================

        new_tokens = (
            draft_tokens[:accepted]
            + [bonus_token]
        )

        for token in new_tokens:

            generated.append(
                token
            )

            num_generated += 1

            # ------------------------------------------------
            # EOS
            # ------------------------------------------------

            if (
                eos_token_id is not None
                and token.item()
                == eos_token_id
            ):

                finished = True

                break

            # ------------------------------------------------
            # Generation limit
            # ------------------------------------------------

            if (
                num_generated
                >= max_new_tokens
            ):

                finished = True

                break

        # ----------------------------------------------------
        # Bonus becomes next anchor.
        # ----------------------------------------------------

        anchor = bonus_token

    # ========================================================
    # FINAL STATISTICS
    # ========================================================

    stats.total_time = (
        time.perf_counter()
        - start
    )

    stats.generated_tokens = (
        num_generated
    )

    stats.gamma_final = gamma

    # --------------------------------------------------------
    # Python/bookkeeping time.
    # --------------------------------------------------------

    stats.python_overhead_time = (
        stats.total_time
        - (
            stats.prefill_time
            + stats.draft_time_total
            + stats.crop_before_verify_time
            + stats.verify_time_total
            + stats.crop_after_verify_time
        )
    )

    # Prevent tiny negative values from timer noise.
    stats.python_overhead_time = max(
        0.0,
        stats.python_overhead_time,
    )

    # ========================================================
    # BUILD OUTPUT
    # ========================================================

    all_generated = torch.cat(
        generated,
        dim=1,
    )[:, :max_new_tokens]

    full_sequence = torch.cat(
        [
            input_ids,
            all_generated,
        ],
        dim=1,
    )

    return (
        full_sequence,
        stats,
    )


# ============================================================
# HIGH-LEVEL GENERATION WRAPPER
# ============================================================

def ssd_generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    max_draft_tokens: int = 8,
    **gamma_kwargs,
):
    """
    High-level text generation wrapper.
    """

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    eos_token_id = (
        tokenizer.eos_token_id
    )

    full_ids, stats = (
        ssd_generate_tokens(

            model=model,

            input_ids=inputs.input_ids,

            max_new_tokens=max_new_tokens,

            max_draft_tokens=max_draft_tokens,

            eos_token_id=eos_token_id,

            **gamma_kwargs,
        )
    )

    # --------------------------------------------------------
    # Decode ONLY newly generated tokens.
    # --------------------------------------------------------

    new_ids = full_ids[
        :,
        inputs.input_ids.shape[1]:,
    ]

    text = tokenizer.decode(
        new_ids[0],
        skip_special_tokens=True,
    )

    return (
        text,
        stats,
    )


# ============================================================
# DIRECT SANITY TEST
# ============================================================

if __name__ == "__main__":

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    MODEL_PATH = (
        "Qwen/Qwen2.5-7B-Instruct"
    )

    print()
    print("=" * 70)
    print("QWEN2.5-7B SSD SANITY TEST")
    print("=" * 70)

    # ========================================================
    # TOKENIZER
    # ========================================================

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_PATH
        )
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = (
        AutoModelForCausalLM.from_pretrained(

            MODEL_PATH,

            dtype=torch.bfloat16,

            device_map={
                "": "cpu"
            },
        )
    )

    model.eval()

    print(
        "Model:",
        MODEL_PATH,
    )

    print(
        "Device:",
        model.device,
    )

    print(
        "Dtype:",
        next(model.parameters()).dtype,
    )

    print(
        "Layers:",
        model.config.num_hidden_layers,
    )

    # ========================================================
    # TEMPORARY SANITY-TEST SKIP PATTERN
    # ========================================================

    attn_skip = [
        4,
        8,
        12,
        16,
        20,
        24,
    ]

    mlp_skip = [
        6,
        10,
        14,
        18,
        22,
    ]

    model.config.skip_layer_ids = {

        "attn":
            attn_skip,

        "mlp":
            mlp_skip,
    }

    for i, layer in enumerate(
        model.model.layers
    ):

        # Layer 0 protected.

        if i == 0:

            layer.skip_attn = False

            layer.skip_mlp = False

        else:

            layer.skip_attn = (
                i in attn_skip
            )

            layer.skip_mlp = (
                i in mlp_skip
            )

    # ========================================================
    # PROMPT
    # ========================================================

    messages = [

        {
            "role": "system",

            "content":
                "You are Qwen, created by Alibaba Cloud. "
                "You are a helpful assistant.",
        },

        {
            "role": "user",

            "content":
                "Give me a short introduction to "
                "large language models.",
        },

    ]

    text = (
        tokenizer.apply_chat_template(

            messages,

            tokenize=False,

            add_generation_prompt=True,
        )
    )

    # ========================================================
    # RUN
    # ========================================================

    generated_text, stats = ssd_generate(

        model=model,

        tokenizer=tokenizer,

        prompt=text,

        max_new_tokens=64,

        max_draft_tokens=8,

        initial_gamma=0.8,

        alpha=0.9,

        beta1=0.5,

        beta2=0.9,

        epsilon=0.01,
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    print()
    print(
        "--- Generated text ---"
    )

    print(
        generated_text
    )

    print()
    print(
        "--- SSD statistics ---"
    )

    print(
        stats.report()
    )

    print()
    print(
        "--- Adaptive gamma ---"
    )

    print(
        f"Initial gamma : "
        f"{stats.gamma_start:.4f}"
    )

    print(
        f"Final gamma   : "
        f"{stats.gamma_final:.4f}"
    )

    print()
    print(
        "--- Draft stopping ---"
    )

    print(
        f"Low-confidence stops : "
        f"{stats.low_confidence_stops}"
    )

    print(
        f"Max-draft stops      : "
        f"{stats.max_draft_stops}"
    )

    print(
        f"EOS stops            : "
        f"{stats.eos_stops}"
    )

    print(
        f"Other stops          : "
        f"{stats.other_stops}"
    )

    print()
    print(
        "Draft-length histogram:"
    )

    print(
        dict(
            sorted(
                stats.draft_length_histogram.items(),
                key=lambda x: int(x[0])
            )
        )
    )

    print()
    print(
        "=" * 70
    )

    print(
        "SANITY TEST COMPLETE"
    )

    print(
        "=" * 70
    )