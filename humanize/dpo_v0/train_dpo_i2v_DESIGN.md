# `train_dpo_i2v.py` Design (handoff to morning luke1)

This document is the design for the v0 i2v-original-init DPO trainer that
landed under `videodpoWan:humanize/dpo_v0/train_dpo_i2v.py` after Round 2.
The design is committed *before* the trainer code so that morning luke1
can validate the architecture, fill in any gaps in the high-noise model
forward path, and run M2 + M3 under supervision.

The plumbing pieces it composes are all already committed and tested:

| Building block | Path | Tested in Round | Status |
|---|---|---|---|
| recipe_id (`6bef6e104cdd3442`) + canonical YAML | `humanize/dpo_v0/recipes/{wan22_i2v_a14b__round2_v0.yaml, recipe_id, canonical_serializer.py}` | 1 | rl2 PASS msg `69bea853` |
| canonical sharded loader + per-key sidecar JSONL | `humanize/dpo_v0/loader/canonical_loader.py` | 1 | rl2 PASS msg `9f56abb1`, 21 tests |
| real-shard manifests under `<UPSTREAM>/{high,low}_noise_model/` | `humanize/dpo_v0/loader/out/20260428T040305Z/` | 2 sub-step 2a | rl2 PASS msg `8e40d34c` |
| video latent encoder + heldout-leak guard | `humanize/dpo_v0/encode_videos.py` | 2 sub-step 2b | running, tier_a 32 latents |
| flow-matching DPO loss + 7 unit tests | `humanize/dpo_v0/{dpo_loss.py, test_dpo_loss.py}` | 2 sub-step 2b | 7/7 PASS |

The trainer is the missing layer on top.

## Compute envelope (rl2 + luke1 directives)

| Path | GPU | Wall budget | Status |
|---|---|---|---|
| M2 identity-gate smoke | 1×A100 80GB or 1×A6000 48GB | < 5 min | pending — first morning sub-step |
| M3 tier_a tiny-overfit (16 pairs, ≤ 200 steps, beta=0.1, lr=1e-5) | 1× A100/A6000 | ~30 min | pending |
| M4 tier_b short DPO (200-pair scope-cut for proof-of-pipeline) | 4× A100 80GB DS Zero-2 on `juyi-finetune` | ~2 h | pending — luke1 push-button after M3 PASS |

`compute_envelope ∈ {single_gpu, dpo_multi_gpu_zero2}` is stamped into
every run manifest per the AC-6 contract.

## Module layout (`train_dpo_i2v.py`)

```
load_canonical_state_to_model(manifest_path, model)
    Reads the per-key sidecar JSONL line by line, loads the matching
    safetensors shard for each key on demand, populates the model state.
    Asserts merged_state_sha256 of the loaded weights matches the
    manifest before any forward.

build_policy_and_reference(high_noise_manifest, low_noise_manifest, dtype, device, ref_offload)
    Returns (policy, reference, low_noise_frozen).
    - policy: high-noise expert weights + LoRA adapter (rank=16, alpha=16,
      target_modules from the parent T3_design — typically attn1.{q,k,v,o},
      attn2.{q,k,v,o}, ffn.{0,2}). LoRA initialized to zero so policy
      starts byte-identical to reference at step 0.
    - reference: high-noise expert weights, no_grad, no LoRA. If
      ref_offload=True, the reference state lives on CPU pinned memory
      and is moved to GPU only inside the forward; on return the GPU
      copy is freed. Tracked by a `RefOffloadHandle` context manager.
    - low_noise_frozen: low-noise expert weights, no_grad, no LoRA.
      Used during inference (M5/M6) only; not loaded for tier_a training
      because tier_a stays inside the high-noise band [0, 0.358].

PerPairSampler(pair_ids, seed_namespace='dpo-tier_a')
    Deterministic per-pair (t, eps) sampler: for pair_id p, returns
    (t_p, eps_p) where eps_p has the same shape as the latent and t_p
    is sampled inside the high-noise training band (raw timestep > 900,
    boundary fractions [0, 0.358] over 1000-step scheduler). The sampler
    is independent across pairs but identical between winner/loser AND
    between policy/reference within one pair.

RoutingCounter
    Per-forward, increments {high_noise, low_noise} based on
    raw_timestep > switch_DiT_boundary * 1000 (= 900). Logs
    (sampled_timestep_id, raw_timestep, detected_expert) per forward.
    Asserts 100% high-noise on tier_a (HARD, AC-5.U3); any low-noise
    increment raises immediately.

DPOTrainer
    main training loop:
    - per step: sample pair, load (encoded_winner_latent, encoded_loser_latent),
      sample (t, eps) shared via PerPairSampler, noise both latents to
      z_w_t / z_l_t, forward through policy + reference (4 forwards
      total), compute v_target_winner = z_winner - eps (flow matching
      target), call flow_matching_dpo_loss, backward only on policy
      LoRA params, optimizer step.
    - every K steps: snapshot LoRA state to safetensors at
      humanize/dpo_v0/ckpts/<UTC>/lora_high_noise.safetensors.
    - logs: loss, components (per-sample MSEs, advantages, logit, prob),
      routing counter, peak GPU memory.
    - run manifest at humanize/dpo_v0/ckpts/<UTC>/run_manifest.json with
      compute_envelope, GPU count, --ref_offload status, recipe_id pin,
      policy_base_merged_sha256, reference_merged_sha256, lora config,
      optimizer state path, code_commit_id, machine_internal_ip_tail,
      wall start/end timestamps.

identity_gate_smoke(policy, reference, sampler, dtype, atol, rtol)
    AC-2.2 + AC-2.3 single-rank smoke. Disables LoRA on policy (or sets
    it to no-op via zero adapter), forwards the same batch through
    policy + reference, asserts allclose(rtol=atol=1e-3) on noise pred,
    samples one short generation, computes per-frame L1 / SSIM / PSNR
    against the corresponding raw-original-init no-DPO baseline. The
    gate halts the run before any optimizer step is taken if any axis
    fails.
```

## Open questions for morning luke1 review

1. **`WanModel` LoRA injection point**: peek at
   `videodpoWan/wan/modules/model.py` to confirm the LoRA adapter
   positions in attn / FFN layers. Round 1 task2 did NOT touch model
   internals; this is the first round that does.
2. **DiffSynth `train.py` reuse vs custom loop**: the trainer can either
   subclass `DiffusionTrainingModule` (reusing DiffSynth's gradient
   checkpointing + accelerate integration) OR run a custom loop. Custom
   loop is simpler for the M3 single-GPU run; DiffSynth subclass is
   needed for M4 4-GPU DS Zero-2. Recommend: write the M3 single-GPU
   loop first (custom), promote to DiffSynth integration for M4.
3. **Reference offload semantics for the smoke test**: M2 identity gate
   runs both models in GPU memory simultaneously; ref_offload is OFF
   for the smoke (3-min run, 80 GB GPU memory) and ON for the actual
   training (`--ref_offload`).
4. **Per-pair (t, eps) determinism across re-runs**: the sampler uses
   sha256(seed_namespace + pair_id) → deterministic (t, eps). This is
   the same byte-spectrum-lock pattern as the recipe_id and the
   canonical hash — caller can re-run a step and reproduce.
5. **Tier_a tiny-overfit halting condition**: per AC-5 + DEC-CAL,
   loss ≤ 0.30 in ≤ 200 steps is the preliminary target. rl2's `797e84b6`
   pre-approves plateau in [0.30, 0.45] as proceed; > 0.50 or diverge
   triggers halt + log RootCause for morning luke1.

## Round 2 closure (if M2/M3 not run tonight)

If the trainer code lands but does not run M2/M3 tonight (because of
time / supervision risk), the round-2-summary documents:

- Round 0 + Round 1 carryforward (recipe pin, canonical loader, real-shard
  manifests, encoder with heldout guard, DPO loss with sign-checked
  tests).
- Round 2 sub-step 2a (real-shard manifests committed with merged_state_sha256
  values).
- Round 2 sub-step 2b (encoder + dpo_loss + 7 tests passing, tier_a 32
  latents written to disk under `humanize/dpo_v0/latents/<UTC>/tier_a/`,
  manifest.jsonl recording per-pair encode provenance).
- Round 2 sub-step 2c (this design + the `train_dpo_i2v.py` skeleton).
- Status: M2 identity gate + M3 tier_a tiny-overfit defer to morning
  luke1; rl2 has agreed to the M4/M5/M6/M8 defer (msg `fd55f12c` accepts
  scaffold-only for the trainer-execution-half of Round 2).
- Suggested morning sequence: review trainer skeleton → fill any
  open-question gaps → run M2 (5 min) → run M3 (30 min) → if both pass,
  ship to `juyi-finetune` and run M4 (~2 h) → M5 (10 min) → M6 8-prompt
  partial heldout (30 min) → M8 PhyJudge (30 min). Total ~3.5 h
  supervised wall.

## Hard invariants (carried forward)

- recipe_id `6bef6e104cdd3442` (frozen, asserted at trainer init)
- heldout 42 prompts / 245 groups / 579 pairs NEVER pre-encoded (encoder
  enforces; trainer asserts before any forward via `T3_subset.json:heldout_excluded`)
- routing counter 100% high-noise on tier_a (hard fail on any low-noise hit)
- low-noise expert byte-equality post-M4 (asserted via canonical_loader
  re-hash at end of training)
- generation_config byte-identical between baseline + trained ckpt for M5/M6
- PhyJudge probe-time field-name mapping or `judge-axis-missing` halt
