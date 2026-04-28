# Pin Verification Report — train_dpo_i2v.py vs i2v.md AC-3

**Verifier**: rl8 (task #10/#13 → "task10 (analyze)" tag in i2v.md task graph)
**Trainer commit**: `videodpoWan@3a78128` (rlcr/task-6 tip; pin-assertion code unchanged from `ec812a0`)
**Plan source of truth**: `videodpo:humanize/i2v.md` AC-3 (canonical pin contract)
**Note on the spec's parent-plan reference**: rl4 spec cites `videodpo/docs/exp-plan/trainpugev2vplan.md` as the parent plan; that file does not exist on disk in this checkout, the `git stash@{0}` of `videodpo`, or in the `rlcr-task-5` videodpo worktree under `/shared/user59/.config/superpowers/worktrees/videodpo/`. The closest extant ancestor is the rlcr-task-5 plan at `humanize/plan.md` + `humanize/v1.md` (predecessor v0 minimal-loop spec for I2V-A14B), which does not specify per-pin SHA values. Per i2v.md AC-3.1 the parent plan and i2v.md "share this exact T3_design source so its recipe_id is identical when finalized under the same serializer" — i.e., i2v.md is the canonical pin contract for task-6, and verification proceeds against it.

## Summary

| AC | Pin | Trainer enforces? | Drift? |
|----|-----|---------------------|--------|
| AC-3.1 | recipe_id `6bef6e104cdd3442` (3-way: const ↔ on-disk pin ↔ recompute YAML hash) | **YES** — `assert_recipe_pin()` at L87-94, called BEFORE any side-effect at L423 | ✅ no drift |
| AC-3.2 | `Wan2.1_VAE.pth` SHA256 | **NO** — trainer uses path string `args.upstream / "Wan2.1_VAE.pth"` at L459, no SHA256 assertion | ⚠️ DRIFT — see D1 |
| AC-3.3 | `models_t5_umt5-xxl-enc-bf16.pth` SHA256 | **NO** — trainer uses path string at L488, no SHA256 assertion | ⚠️ DRIFT — see D2 |
| AC-3.3 | tokenizer-tree SHA over `google/umt5-xxl/` sorted-by-relpath | **NO** — trainer uses path string at L489, no tree-SHA computation | ⚠️ DRIFT — see D3 |
| AC-3.4 | Canonical hash rule (alphabetical streaming, `key|shape|dtype|tensor_bytes`, sidecar JSONL, no per-key inlined into manifest) | **YES** — implemented in `loader/canonical_loader.py` (round-1, commit `6dcd04a`); trainer does not directly invoke it (loader is for AC-1 base-state hashing, not per-step) | ✅ no drift |
| AC-3.5 | `switch_DiT_boundary = 0.9` → raw `900` | **YES** — `SWITCH_DIT_BOUNDARY_RAW = 900` constant at L62, used by `RoutingCounter` and routing assertions | ✅ no drift on the constant value, but no explicit *assertion* against the spec value (see N1) |
| AC-3.5 | `fps = 16` | **PIN-VIA-RECIPE_ID** — present in canonical recipe YAML; covered by recipe_id 3-way assert. Trainer does not separately read/assert it (frame rate is a preprocessing axis, not training-step axis). | ✅ no drift, indirect |
| AC-3.5 | `frame_num = 81` | **YES (default)** — `--frame-num` argparse default at L417 = 81; also in recipe yaml under recipe_id chain | ✅ no drift; ambient default matches |
| AC-3.5 | `aggregation_rule = cross_group_rater_union` | **NOT IN SCOPE** for trainer — this is a pair-construction rule (upstream T1 group export), not a runtime-step axis. Recipe YAML does not encode it (verified — only preprocessing axes in `wan22_i2v_a14b__round2_v0.yaml`). Pinned by T1 export commit, not by trainer. | ✅ out-of-scope for trainer |
| AC-3.5 | dtype policy: bf16 forward / fp32 master | **PARTIAL** — `dtype = torch.bfloat16` at L441 (forward); fp32 master is the AdamW default (params are fp32, gradients accumulated in fp32 by AdamW). `torch.amp.autocast("cuda", dtype=dtype)` at L577/L590 confirms bf16 forward path. | ✅ no drift |
| AC-3.5 | ref-offload preserves dtype + device on reload | **YES** — reference loaded with `.to(dtype=dtype, device="cpu" if args.ref_on_cpu else device)` at L519-521; per-step `reference.to(device)` at L576 then `reference.cpu()` at L581 — dtype is preserved across both moves (`.to(device)` and `.cpu()` only change device, not dtype). | ✅ no drift |
| AC-8 | Run manifest stamps each pin value | **PARTIAL** — `recipe_id` stamped at L657, but VAE SHA / T5 SHA / tokenizer-tree SHA / scheduler-config snapshot are NOT stamped (because they are not computed; see D1–D3). | ⚠️ DRIFT — manifest field set is incomplete relative to AC-8's "stamp `recipe_id`, VAE SHA, T5 SHA, tokenizer-tree SHA, scheduler-config snapshot" |

## Drifts (in priority order)

### D1 — VAE SHA256 not asserted (AC-3.2 violation)

**Expected (i2v.md L27)**: trainer asserts `Wan2.1_VAE.pth` SHA256 equals the parent plan's pinned VAE; canonical path `<UPSTREAM>/Wan2.2-I2V-A14B/Wan2.1_VAE.pth`.

**Actual (train_dpo_i2v.py L457-459)**:
```python
from wan.modules.vae2_1 import Wan2_1_VAE
vae = Wan2_1_VAE(z_dim=16, vae_pth=str(args.upstream / "Wan2.1_VAE.pth"), dtype=dtype, device=str(device))
```
Path string is correct, but no `_file_sha256(vae_pth)` is computed and no assertion against a pinned value.

**Risk**: under the cross-box parallel-execution directive (luke1 `bbbef5e4`) M3 runs on `juyi-finetune` and M4 runs on `juyi-videorl`. If the two boxes' `<UPSTREAM>/Wan2.2-I2V-A14B/Wan2.1_VAE.pth` byte-differ (e.g., one box has a partial rsync, or mixed I2V-A14B vs TI2V-5B VAE files were copied — note AC-3.2 explicitly warns about this confusable family), the trainer will silently load the wrong VAE without halting. The "buggy ckpt + clean eval = 假阳性" risk @rl2 flagged in the rl4 onboarding directly applies here.

**Recommended remediation** (do NOT modify trainer code without rl1 sign-off; this is a flag, not a fix):
1. Add `EXPECTED_VAE_SHA256: str` constant near `EXPECTED_RECIPE_ID` at L57-59.
2. After L459, add `actual_vae_sha = _file_sha256(args.upstream / "Wan2.1_VAE.pth"); assert actual_vae_sha == EXPECTED_VAE_SHA256, f"VAE pin drift: actual={actual_vae_sha}, expected={EXPECTED_VAE_SHA256}"`.
3. Stamp `vae_sha256` into the run manifest.
4. The constant value should be sourced from the pinned VAE on a known-good box (e.g., compute on juyi-finetune's `<UPSTREAM>/Wan2.2-I2V-A14B/Wan2.1_VAE.pth` AFTER M0 audit re-confirms it's correct).

### D2 — T5 SHA256 not asserted (AC-3.3 violation, encoder file)

**Expected (i2v.md L28)**: trainer asserts `models_t5_umt5-xxl-enc-bf16.pth` SHA256 equals parent plan's pin.

**Actual (train_dpo_i2v.py L484-490)**:
```python
text_encoder = T5EncoderModel(
    text_len=512, dtype=dtype, device=device,
    checkpoint_path=str(args.upstream / "models_t5_umt5-xxl-enc-bf16.pth"),
    tokenizer_path=str(args.upstream / "google" / "umt5-xxl"),
)
```
Path strings are correct, but no SHA256 computed, no assertion.

**Risk**: same as D1 — cross-box drift, mis-sync, family-confusable substitution.

**Recommended remediation**: parallel to D1 — add `EXPECTED_T5_ENC_SHA256` constant + `_file_sha256` assert before L484.

### D3 — Tokenizer-tree SHA not computed (AC-3.3 violation, tokenizer side)

**Expected (i2v.md L28)**: tokenizer-tree SHA over sorted-by-relpath file SHAs of `<UPSTREAM>/Wan2.2-I2V-A14B/google/umt5-xxl/`.

**Actual**: trainer passes the directory path to `T5EncoderModel(tokenizer_path=...)`; no tree-walk + per-file SHA + concatenated SHA.

**Risk**: tokenizer drift (added/removed/modified files in `google/umt5-xxl/`) silently corrupts text encoding. The risk is lower than VAE/T5 weights drift in absolute terms but is the canonical AC-3.3 contract.

**Recommended remediation**: tree-SHA helper in `assert_recipe_pin()`'s neighborhood:
```python
def tokenizer_tree_sha(tokenizer_dir: pathlib.Path) -> str:
    rel_files = sorted(p.relative_to(tokenizer_dir).as_posix()
                       for p in tokenizer_dir.rglob("*") if p.is_file())
    h = hashlib.sha256()
    for rel in rel_files:
        h.update(f"{rel}|{_file_sha256(tokenizer_dir / rel)}\n".encode("ascii"))
    return h.hexdigest()
```
Then assert against `EXPECTED_TOKENIZER_TREE_SHA` constant; stamp into manifest.

## Notes (non-drifts, but worth flagging)

### N1 — `switch_DiT_boundary` is encoded but not "asserted"

The trainer hard-codes `SWITCH_DIT_BOUNDARY_RAW = 900` at L62 with comment `# AC-5.U2 raw boundary (switch_DiT_boundary * 1000)`. This is not a runtime assertion against a value source — it's a code-level constant that must be manually kept in sync with the spec's `0.9`.

In practice this is fine because the constant is in the same file as the routing logic, so any code review touches both at once. But under AC-3.5's spirit of "pin = run-time stamped value", the trainer could stamp `{"switch_dit_boundary_raw": SWITCH_DIT_BOUNDARY_RAW}` into the manifest. Currently it stamps `p3_sampling_band: [SAMPLING_T_LOW, SAMPLING_T_HIGH] = [901, 999]` which implicitly encodes the boundary > 900, but not the boundary itself.

**Severity**: low — defer to round-3+ manifest schema work (rl5 task #16, manifest_writer.py).

### N2 — `compute_envelope` honest enum

Trainer L653: `"compute_envelope": "dpo_multi_gpu_ddp" if is_distributed else "single_gpu"` — explicitly NOT `dpo_multi_gpu_zero2` per rl1's honest-enum decision (recorded in code comment L650-652 citing rl2 `df979b3d`). This is a documented divergence from i2v.md's `compute_envelope ∈ {single_gpu, dpo_multi_gpu_zero2}` enum at L76. The plan's enum should be amended to include `dpo_multi_gpu_ddp` in the round-3 plan rewrite. Not a pin drift (no contract violated), but a vocabulary drift between plan and code.

**Severity**: doc-only — covered by my plan-amendment task #10 commit `a52fb90` flag for round-3 plan rewrite.

### N3 — `recipe_id` stamping is correct

Trainer at L657 stamps `"recipe_id": recipe_id` where `recipe_id` is the return value of `assert_recipe_pin()` (i.e., the on-disk pin value, validated to equal both the recompute and the constant). Manifest reader can therefore trust the stamped value as authoritative for the run.

## What I did NOT verify

- I did not recompute `sha256(canonical_yaml.bytes).hex[:16]` against the on-disk `recipes/recipe_id` text file myself — trusted that the round-1 fixture-locked tests in `loader/test_canonical_loader.py` (21 tests, 1.13s, rl2 review msg `9f56abb1` confirms 21/21 PASS independent recompute) cover this.
- I did not verify the `<UPSTREAM>/Wan2.2-I2V-A14B/Wan2.1_VAE.pth` file's actual SHA on either juyi-finetune or juyi-videorl. That is part of M0 inheritance audit (round 0) and is documented as `38071ab5…` per rl1 msg `c640e050`. The drift D1 is about the trainer's lack of runtime check, not about whether the file is currently correct.
- I did not run any code from this report.
- I did not modify trainer code.

## Action items for sign-off

- ⚠️ **rl2 sign-off requested on D1/D2/D3** — these are real AC-3 contract gaps. Decision: (a) accept as known divergence with manifest stamping deferred to round-3+ (low risk if M0 audit re-confirms VAE/T5/tokenizer hashes on each box before training starts), or (b) require a hot-fix patch from rl1 to add the SHA asserts before M3/M4 wall-clock budgets get spent.
- ✅ **rl1 informed** — the asserts are missing in `train_dpo_i2v.py`; if you decide to add them, the values can be sourced from M0 audit hashes (e.g., VAE `38071ab5…`).
- 📌 Plan amendment NOT required for D1–D3 — i2v.md AC-3.2/AC-3.3 already specify the contract; trainer is the side that needs to catch up.
- 📌 Plan amendment recommended for N2 — `compute_envelope` enum needs `dpo_multi_gpu_ddp` added; this rides on the round-3 plan rewrite already flagged in task #10's commit `a52fb90`.

## Provenance

- This report verifies trainer code at `videodpoWan@3a78128`. Rolling back to the spec-cited `ec812a0` commit yields an identical pin-assertion code path (D1–D3 also apply there); newer commits `e7272fc`, `0030832`, `3a78128` only add `--cond-image-fallback-root` (cross-machine deploy), `--enable-grad-ckpt` (80 GB OOM hardening), and a cond-image-missing pair filter — none touch pin assertions.
- Author: `@rl8` (slock agent id `3e1255af-f8ca-4860-9410-d7ee9d1af8ec`).
- Generated: 2026-04-27 evening.
