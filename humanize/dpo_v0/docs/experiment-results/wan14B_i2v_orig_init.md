# Wan2.2-I2V-A14B Direct I2V DPO from Original Base — Terminal Results (v0)

> **Status**: SCAFFOLD. All numeric fields below are placeholders awaiting the
> M4 trained checkpoint, the M5 eval-harness smoke gate, the M6 42-prompt
> heldout regen, and the M8 PhyJudge-9B paired delta. Filled values must
> originate from the run manifest produced by `manifest_writer.py`; see
> "Provenance fingerprint" section below for the manifest fields each
> table reads from.
>
> Plan reference: `videodpo:humanize/i2v.md` AC-7 (heldout regen +
> generation_config), AC-8 (terminal report), DEC-7 (composite primary
> scalar), AC-5.U3/U4 (routing counter).

## Comparator pair

| Slot | Identity | Source path |
|------|----------|-------------|
| **Baseline** | original-init no-DPO baseline | `<UPSTREAM>/Wan2.2-I2V-A14B/high_noise_model` (loaded under the canonical sharded loader, `merged_state_sha256` stamped in manifest) + frozen low-noise expert |
| **Trained** | original-init DPO ckpt | same baseline weights + LoRA adapter from M4 short DPO run on tier_b (200-pair, single beta) + frozen low-noise expert |

The two are byte-identical except for the LoRA adapter; both go through the
same generation_config and the same canonical T2 conditioning-image map.

## Generation config (byte-identical across baseline + trained)

```json
{
  "seed": null,
  "sampler": null,
  "inference_steps": 50,
  "guidance_scale": null,
  "negative_prompt": null,
  "resolution": null,
  "num_frames": 81,
  "dtype": null,
  "judge_preprocessing": null
}
```

Per the contract, every field above must match byte-for-byte between the
baseline and trained runs. The full dict is stamped under
`manifest["generation_config"]` for both runs.

## Run provenance (per ckpt)

| Field | Baseline run | Trained run |
|-------|--------------|-------------|
| `commit_id` | TBD | TBD |
| `machine_internal_ip_tail` | TBD | TBD |
| `compute_envelope` | `multi_gpu_inference_seed_parallel` (4 ranks over 42 prompts) | same |
| `recipe_id` | `6bef6e104cdd3442` | `6bef6e104cdd3442` |
| `vae_sha256` | TBD | TBD |
| `t5_sha256` | TBD | TBD |
| `tokenizer_tree_sha256` | TBD | TBD |
| `merged_state_sha256` (high-noise base) | TBD | TBD (must equal baseline's value byte-for-byte) |
| `merged_state_sha256` (low-noise frozen, post-run) | TBD | TBD (must equal pre-run baseline's value byte-for-byte) |
| `lora_adapter_sha256` | (none) | TBD |
| `routing_counter`: high_noise / low_noise / total_forwards | TBD / 0 / TBD | TBD / 0 / TBD |

### Compute-envelope rollup (per phase)

| Phase | Envelope | Box |
|-------|----------|-----|
| M2 identity gate | `single_gpu` | dev box A6000 |
| M3 tier_a tiny-overfit | `dpo_multi_gpu_zero2` (4×A100) | `juyi-finetune` |
| M4 tier_b short DPO | `dpo_multi_gpu_zero2` (8×A100) | `juyi-videorl` |
| M5 eval-harness smoke (dual sample) | `single_gpu` | TBD |
| M6 42-prompt heldout regen | `multi_gpu_inference_seed_parallel` (4 ranks over seeds) | TBD |
| M8 PhyJudge-9B paired delta | (CPU bootstrap) | TBD |

The `compute_envelope` field per run is taken from the run manifest; if a
phase ran on a different envelope than the table above, the manifest is the
source of truth and the table here is amended in the same commit.

## Primary terminal scalar (composite `SA + PTV + persistence`)

Per DEC-7, the primary scalar is the sum of three PhyJudge-9B axes after the
trainer probes `serveandeval_9b.sh` for the explicit field-name mapping. If
any axis is missing, the trainer halts with `judge-axis-missing` rather than
substituting (audit at `<run_dir>/judge_axis_missing.json`).

| Quantity | Mean | Std | n_pairs | 95 % bootstrap CI on paired Δ (trained − baseline) |
|----------|------|-----|---------|-----|
| Composite (SA + PTV + persistence) | TBD | TBD | TBD (= 42, all heldout prompts paired) | TBD |

CI is per AC-8.4: bootstrap with `B = 10000` resamples over the 42 paired
deltas, percentile method.

## Secondary axes (every other field returned by the judge probe)

Each row is filled from `manifest["judge_field_probe"]["raw_probe_payload"]`
unioned over both runs. Columns mirror the primary scalar; rows are emitted
in alphabetical-by-axis-name order so reviewers can diff two reports
line-for-line.

| Axis | Probed field name | Trained mean ± std | Baseline mean ± std | Paired Δ mean ± std | 95 % CI on Δ |
|------|--------------------|---------------------|----------------------|----------------------|------|
| TBD | TBD | TBD | TBD | TBD | TBD |

## Routing-counter sanity (must be 100 % high-noise)

Per AC-5.U3 the runtime routing counter is the sole hard assert; tier_a
forwards must increment the high-noise counter only, and any low-noise
increment is a hard fail. The same contract applies to tier_b under AC-6:
post-run `merged_state_sha256` of the low-noise expert must equal the
pre-run value byte-for-byte. Both fields land in the manifest.

| Phase | high_noise count | low_noise count (must be 0) | post-run low-noise byte-equality |
|-------|------------------|------------------------------|-----------------------------------|
| M3 tier_a | TBD | 0 | n/a (M3 is high-noise tiny-overfit only) |
| M4 tier_b | TBD | 0 | TBD (PASS / FAIL) |

## Reproduction commands (filled in once M4 lands)

```bash
# Baseline regeneration (frozen low-noise + frozen original-init high-noise)
python -m humanize.dpo_v0.heldout_regen \
  --baseline-ckpt <UPSTREAM>/Wan2.2-I2V-A14B \
  --heldout <T0_T3_ROOT>/splits/heldout.json \
  --t2-image-manifest <T0_T3_ROOT>/t2/image_manifest.json \
  --generation-config <run_dir>/generation_config.json \
  --out <baseline_run_dir>

# Trained regeneration (frozen low-noise + LoRA-adapted high-noise)
python -m humanize.dpo_v0.heldout_regen \
  --baseline-ckpt <UPSTREAM>/Wan2.2-I2V-A14B \
  --lora-adapter <m4_run_dir>/lora_final.safetensors \
  --heldout <T0_T3_ROOT>/splits/heldout.json \
  --t2-image-manifest <T0_T3_ROOT>/t2/image_manifest.json \
  --generation-config <run_dir>/generation_config.json \
  --out <trained_run_dir>

# PhyJudge paired delta + 95 % CI
python -m humanize.dpo_v0.judge_paired_delta \
  --baseline-run <baseline_run_dir> \
  --trained-run <trained_run_dir> \
  --judge-script /shared/user60/worldmodel/wmbench/evals/script/serveandeval_9b.sh \
  --bootstrap-resamples 10000 \
  --out <report_dir>
```

## Watch items (carried forward into M4 / M6 / M8)

- M4 activation budget on `juyi-videorl` (8×A100) under DDP + `--ref-on-cpu`
  (round-3 trainer uses plain DDP; round-4+ aspirations to real DS Zero-2).
- cu128 vs cu130 cross-box drift between `juyi-finetune` and `juyi-videorl`
  (cu128 torch on juyi-videorl, cu130 torch on juyi-finetune; bf16 forward
  numerics could differ at the rtol = atol = 1e-3 identity-gate threshold —
  flagged for an M2-on-each-box re-run if cross-box ckpt comparison ever
  becomes load-bearing).
- 200-pair tier_b is proof-of-pipeline scope; primary-scalar paired Δ may
  be statistically indistinguishable from zero with n = 42 even if the
  trained ckpt is genuinely better. Direction-of-effect is the read-out;
  significance comes only with a longer tier_b run.
- VAE / T5 / tokenizer-tree SHA pins are documented in i2v.md AC-3.2 / AC-3.3
  but not enforced at trainer startup as of round-3 (`videodpoWan@3a78128`);
  see `pin_verification_report.md` D1 / D2 / D3. M0 audit's pre-train re-hash
  on each box is the current mitigation.

## Provenance fingerprint (append-only)

Every value filled into the tables above is sourced from one of these
`manifest_writer.RunManifest` fields. Reviewers diffing a filled-in copy
of this doc against the next run should be able to walk:

1. `commit_id` ↔ git commit at run time
2. `machine_internal_ip_tail` ↔ box that produced the artifacts
3. `compute_envelope` ↔ DS-Zero-2 vs DDP vs single-GPU
4. `recipe_pins.recipe_id` ↔ canonical YAML pin (`6bef6e104cdd3442`)
5. `merged_state_sha256` + `per_key_sidecar_path` + `per_key_sidecar_sha256`
   ↔ canonical hash of the policy / reference base
6. `routing_counter_log` ↔ AC-5.U4 per-forward log
7. `judge_field_probe.axis_to_field` ↔ DEC-7 mapping
8. `generation_config` ↔ AC-7.2 byte-identical config

A row that cannot be traced back to one of these fields is a documentation
bug; halt the report rather than ship.
