# Canonical Loader — Run Manifest Schema

`schema_version: 1` (declared explicitly in every emitted manifest).

The canonical sharded loader emits two artifacts per expert directory:

* a single **run manifest** JSON file (this schema)
* a sibling **per-key sidecar JSONL** (one ``{"key", "shape", "dtype", "sha256"}`` line per tensor, alphabetical key order, ``sha256`` per AC-3.4)

M2 / M4 / trainer code reads the manifest fields documented here without parsing source.

## Manifest fields

| field | type | required | example | notes |
|---|---|---|---|---|
| `schema_version` | int | yes | `1` | bump on incompatible changes |
| `expert_tag` | string enum | yes | `"high_noise"` | one of `"high_noise"` / `"low_noise"` |
| `expert_dir` | string (absolute path) | yes | `"/shared/user63/workspace/data/Wan/Wan2.2-I2V-A14B/high_noise_model"` | directory containing the 6 shards + `safetensors_index.json` |
| `shard_index_filename` | string | yes | `"diffusion_pytorch_model.safetensors.index.json"` | filename relative to `expert_dir` |
| `shards` | list[ShardEntry] | yes | see below | sorted by filename |
| `shard_count` | int | yes | `6` | length of `shards` |
| `tensor_count` | int | yes | `1095` | number of keys streamed |
| `total_parameter_count` | int | yes | `14288901184` | sum of `numel()` across all tensors |
| `shard_dtype_observed` | list[string] | yes | `["torch.float32"]` | sorted list of distinct on-disk dtypes |
| `intended_runtime_dtype` | string | yes | `"bf16"` | trainer-side dtype per AC-3.5 (loader does not convert) |
| `merged_state_sha256` | string (64 hex) | yes | see AC-3.4 | streaming canonical hash over alphabetically walked state |
| `sidecar_jsonl_path` | string (absolute path) | yes | `"…/sidecar.jsonl"` | path of the per-key sidecar JSONL |
| `sidecar_jsonl_sha256` | string (64 hex) | yes | sha256 of sidecar bytes | reproducible across emits |
| `recipe_id` | string (16 hex) | yes | `"6bef6e104cdd3442"` | reread from `humanize/dpo_v0/recipes/recipe_id` and re-hashed against the recipe YAML before any shard I/O |
| `recipes_dir` | string (absolute path) | yes | `"…/humanize/dpo_v0/recipes"` | directory containing `wan22_i2v_a14b__round2_v0.yaml` and `recipe_id` |
| `code_commit_id` | string \| null | no | `"70588f9…"` | git rev-parse HEAD of the trainer repo if available |
| `machine_internal_ip_tail` | string \| null | no | `"2"` | last octet of the machine's primary internal IP if resolvable |
| `loader_module` | string | yes | `"canonical_loader"` | for provenance grep |

### `ShardEntry` (one per shard file under `expert_dir`)

| field | type | required | example |
|---|---|---|---|
| `filename` | string | yes | `"diffusion_pytorch_model-00001-of-00006.safetensors"` |
| `sha256` | string (64 hex) | yes | sha256 of the shard file bytes |
| `size_bytes` | int | yes | shard file size |

### Sidecar JSONL line schema

| field | type | required | example |
|---|---|---|---|
| `key` | string | yes | `"transformer.blocks.0.attn1.q.weight"` |
| `shape` | list[int] | yes | `[5120, 5120]` |
| `dtype` | string | yes | `"torch.float32"` |
| `sha256` | string (64 hex) | yes | per-key SHA per AC-3.4 |

JSON serialization rules: `sort_keys=True`, `ensure_ascii=True`, `separators=(",", ":")`, single trailing LF per line. The file's SHA256 is `sha256(bytes)` of all lines concatenated.

## AC-3.4 streaming canonical hash — locked byte spectrum

For each tensor the hasher (whether per-key or merged) is updated with the following fields, in this exact order, separated by a single `b"|"` byte:

```
key.encode("utf-8")
b"|"
repr(tuple(tensor.shape)).encode("utf-8")
b"|"
str(tensor.dtype).encode("utf-8")
b"|"
tensor.detach().cpu().contiguous().numpy().tobytes()
```

Locked PyTorch repr forms (do not silently drift if PyTorch changes):

| value | exact bytes |
|---|---|
| `repr(tuple([2]))` | `b"(2,)"` (note trailing comma) |
| `repr(tuple([]))` | `b"()"` (scalar) |
| `repr(tuple([1, 3]))` | `b"(1, 3)"` (comma + space) |
| `repr(tuple([5120, 5120]))` | `b"(5120, 5120)"` |
| `str(torch.float32)` | `b"torch.float32"` (with `"torch."` prefix) |
| `str(torch.int64)` | `b"torch.int64"` |
| `str(torch.bfloat16)` | `b"torch.bfloat16"` |
| `str(torch.float16)` | `b"torch.float16"` |

If PyTorch ever changes any of these forms, the existing test fixture fails on its hardcoded expected SHA and the loader requires an explicit schema_version bump.

## Hardcoded test fixtures (regression guard)

The smoke test `test_canonical_loader.py` verifies these expected SHAs against the streaming code:

| key | shape | dtype | values | per-key sha256 (first 16 hex) |
|---|---|---|---|---|
| `alpha` | `(2,)` | `torch.float32` | `[1.0, 2.0]` | `6f96dbdbbcae404d` |
| `beta` | `(1, 3)` | `torch.float32` | `[[3.0, 4.0, 5.0]]` | `13c478754549f532` |
| `gamma` | `()` | `torch.int64` | `42` | `0330d2082b1a99ec` |

Merged-state SHA over the alphabetical walk of `{alpha, beta, gamma}` (first 16 hex): `14c5c6a235105f3f`.

Full hashes are in the test source.

## Real-shard run requirement (Round 2 / task3 prerequisite)

Round 1 does NOT run the loader against the full `<UPSTREAM>` 75 GB to avoid redundant reads (M0 audit already verified the per-shard SHAs and the index integrity). However, Round 2 / task3 **must** execute exactly one real-shard run per expert before trainer init, landing manifest + sidecar at:

```
videodpoWan/humanize/dpo_v0/loader/out/<UTC_timestamp>/{high_noise,low_noise}/{manifest.json, sidecar.jsonl}
```

Trainer init reads the manifest from this path. The manifest's `merged_state_sha256` becomes the AC-2.1 provenance reference. Re-running the loader on the same `<UPSTREAM>` must reproduce the same manifest + sidecar bytes (idempotency).
