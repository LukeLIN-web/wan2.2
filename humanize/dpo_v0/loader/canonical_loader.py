"""Canonical sharded loader for the original Wan2.2-I2V-A14B base.

Reads a single expert directory (``high_noise_model`` or ``low_noise_model``)
under the upstream root, walks every tensor in alphabetical key order, and
emits two artifacts under the streaming canonical hash rule:

* a run-manifest dict (also serialized to a manifest JSON) recording the
  ordered shard list, per-shard SHA256, total parameter count, on-disk
  dtype, expert tag, ``merged_state_sha256``, sidecar JSONL path, sidecar's
  own SHA256, and tensor count
* a per-key sidecar JSONL: one ``{"key", "shape", "dtype", "sha256"}``
  object per line in the same alphabetical order

The streaming canonical hash is computed by maintaining a single
``hashlib.sha256()`` and updating it tensor-by-tensor with these fields,
in this exact order, separated by single ``b"|"`` bytes:

    key.encode("utf-8")
    repr(tuple(tensor.shape)).encode("utf-8")
    str(tensor.dtype).encode("utf-8")
    tensor.detach().cpu().contiguous().numpy().tobytes()

No intermediate buffer is ever constructed; tensors are released after
their bytes are fed to the hasher. The per-key SHA uses the same field
order in a fresh hasher per tensor.

The recipe_id pin from the v0 video_preprocessing recipe is read at
startup, re-hashed against the canonical YAML bytes, and asserted equal
to the frozen value (currently 6bef6e104cdd3442) before any shard I/O.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import pathlib
import socket
import subprocess
import sys
from typing import Iterable

import safetensors
import safetensors.torch
import torch

FIELD_SEPARATOR = b"|"
SHARD_INDEX_FILENAME = "diffusion_pytorch_model.safetensors.index.json"
RECIPE_ID_FILENAME = "recipe_id"
RECIPE_YAML_FILENAME = "wan22_i2v_a14b__round2_v0.yaml"
INTENDED_RUNTIME_DTYPE = "bf16"
SCHEMA_VERSION = 1
KNOWN_GOOD_RECIPE_ID = "6bef6e104cdd3442"


def canonical_field_bytes(key: str, shape: tuple[int, ...], dtype: str) -> tuple[bytes, bytes, bytes]:
    """Return the three header byte-strings that prefix a tensor's bytes.

    The canonical update order is ``key | shape | dtype | tensor_bytes``,
    each field separated by a single ``b"|"`` byte. Returns the three
    header fields without the trailing tensor bytes so callers can feed
    each piece to the hasher (or to a JSONL writer) explicitly.
    """
    return (
        key.encode("utf-8"),
        repr(tuple(shape)).encode("utf-8"),
        dtype.encode("utf-8"),
    )


class StreamingHasher:
    """Builds the AC-3.4 merged-state SHA tensor by tensor.

    The hasher accepts tensors in caller-supplied order. The caller is
    responsible for walking keys alphabetically; the hasher does not
    sort. This decoupling lets the loader write the sidecar JSONL,
    update the merged hasher, and free the tensor in one pass.
    """

    def __init__(self) -> None:
        self._h = hashlib.sha256()
        self._count = 0

    def update(self, key: str, tensor: torch.Tensor) -> str:
        contiguous = tensor.detach().cpu().contiguous()
        shape = tuple(contiguous.shape)
        dtype = str(contiguous.dtype)
        body = contiguous.numpy().tobytes()
        key_b, shape_b, dtype_b = canonical_field_bytes(key, shape, dtype)
        per_key = hashlib.sha256()
        for piece in (key_b, FIELD_SEPARATOR, shape_b, FIELD_SEPARATOR, dtype_b, FIELD_SEPARATOR, body):
            per_key.update(piece)
            self._h.update(piece)
        self._count += 1
        return per_key.hexdigest()

    def hexdigest(self) -> str:
        return self._h.hexdigest()

    @property
    def tensor_count(self) -> int:
        return self._count


def streaming_merged_sha256(state: dict[str, torch.Tensor]) -> tuple[str, int]:
    """Compute the merged-state SHA over an in-memory state dict.

    Convenience wrapper for tests; production code uses ``load_expert``
    which streams from disk and never holds the full state.
    """
    hasher = StreamingHasher()
    for key in sorted(state):
        hasher.update(key, state[key])
    return hasher.hexdigest(), hasher.tensor_count


def per_key_sidecar(state: dict[str, torch.Tensor], out_path: pathlib.Path) -> tuple[str, int]:
    """Write the per-key sidecar JSONL and return ``(sidecar_sha256, count)``.

    Walks ``state`` in alphabetical key order. Each line is a JSON
    object with keys ``"key"``, ``"shape"``, ``"dtype"``, ``"sha256"``;
    the file is written with sorted JSON keys + LF line breaks so its
    bytes are reproducible. Returns the SHA256 of the resulting file.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    file_hasher = hashlib.sha256()
    count = 0
    with out_path.open("wb") as f:
        for key in sorted(state):
            tensor = state[key]
            contiguous = tensor.detach().cpu().contiguous()
            shape = tuple(contiguous.shape)
            dtype = str(contiguous.dtype)
            body = contiguous.numpy().tobytes()
            key_b, shape_b, dtype_b = canonical_field_bytes(key, shape, dtype)
            per_key = hashlib.sha256()
            for piece in (key_b, FIELD_SEPARATOR, shape_b, FIELD_SEPARATOR, dtype_b, FIELD_SEPARATOR, body):
                per_key.update(piece)
            line = json.dumps(
                {"key": key, "shape": list(shape), "dtype": dtype, "sha256": per_key.hexdigest()},
                sort_keys=True,
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("ascii") + b"\n"
            f.write(line)
            file_hasher.update(line)
            count += 1
    return file_hasher.hexdigest(), count


@dataclasses.dataclass
class ShardEntry:
    filename: str
    sha256: str
    size_bytes: int


def _file_sha256(path: pathlib.Path, buf_size: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(buf_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_recipe_pin(recipes_dir: pathlib.Path, expected: str | None = None) -> str:
    """Re-read the recipe id from disk and verify it matches the recipe YAML.

    The optional ``expected`` argument enables a defense-in-depth check
    against a known-good constant (``KNOWN_GOOD_RECIPE_ID``); the
    canonical authority is still the on-disk pair.
    """
    yaml_path = recipes_dir / RECIPE_YAML_FILENAME
    id_path = recipes_dir / RECIPE_ID_FILENAME
    if not yaml_path.exists():
        raise FileNotFoundError(f"recipe YAML missing: {yaml_path}")
    if not id_path.exists():
        raise FileNotFoundError(f"recipe id missing: {id_path}")
    fresh = hashlib.sha256(yaml_path.read_bytes()).hexdigest()[:16]
    on_disk = id_path.read_text(encoding="ascii").strip()
    if len(on_disk) != 16 or any(c not in "0123456789abcdef" for c in on_disk):
        raise ValueError(f"recipe id is not 16 hex chars: {on_disk!r}")
    if fresh != on_disk:
        raise ValueError(
            f"recipe id mismatch: fresh sha256(yaml)[:16]={fresh} on_disk={on_disk}"
        )
    if expected is not None and on_disk != expected:
        raise ValueError(
            f"recipe id drift: on_disk={on_disk} expected (known-good)={expected}"
        )
    return on_disk


def _enumerate_shards(expert_dir: pathlib.Path) -> tuple[list[ShardEntry], dict[str, str]]:
    index_path = expert_dir / SHARD_INDEX_FILENAME
    if not index_path.exists():
        raise FileNotFoundError(f"safetensors index missing: {index_path}")
    index_bytes = index_path.read_bytes()
    index_data = json.loads(index_bytes)
    weight_map: dict[str, str] = index_data["weight_map"]
    shard_filenames = sorted(set(weight_map.values()))
    shards: list[ShardEntry] = []
    for filename in shard_filenames:
        path = expert_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"shard missing: {path}")
        shards.append(
            ShardEntry(filename=filename, sha256=_file_sha256(path), size_bytes=path.stat().st_size)
        )
    return shards, weight_map


def _resolve_expert_tag(expert_dir: pathlib.Path) -> str:
    name = expert_dir.name
    if name == "high_noise_model":
        return "high_noise"
    if name == "low_noise_model":
        return "low_noise"
    raise ValueError(f"unrecognized expert directory name: {name!r}")


def _git_commit_id(repo_root: pathlib.Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _internal_ip_tail() -> str | None:
    try:
        host = socket.gethostbyname(socket.gethostname())
        return host.rsplit(".", 1)[-1]
    except OSError:
        return None


def load_expert(
    expert_dir: pathlib.Path,
    manifest_out: pathlib.Path,
    sidecar_out: pathlib.Path,
    recipes_dir: pathlib.Path,
    repo_root: pathlib.Path | None = None,
    expected_recipe_id: str | None = KNOWN_GOOD_RECIPE_ID,
) -> dict:
    """Stream a single expert into the canonical manifest + sidecar pair.

    ``expert_dir`` must point at a directory containing the 6 ordered
    shards plus a ``diffusion_pytorch_model.safetensors.index.json``.
    ``recipes_dir`` is the directory holding the recipe YAML + id (the
    pin asserts before any shard I/O begins).

    Returns the manifest dict and writes both artifacts. The merged-state
    SHA is computed in alphabetical key order across all shards under
    AC-3.4. Tensors are loaded one at a time via ``safe_open`` and freed
    after their bytes are fed to the hasher; peak memory is bounded by
    the largest single tensor.
    """
    expert_dir = expert_dir.resolve()
    manifest_out = pathlib.Path(manifest_out).resolve()
    sidecar_out = pathlib.Path(sidecar_out).resolve()
    recipes_dir = pathlib.Path(recipes_dir).resolve()

    recipe_id = _read_recipe_pin(recipes_dir, expected=expected_recipe_id)
    expert_tag = _resolve_expert_tag(expert_dir)
    shards, weight_map = _enumerate_shards(expert_dir)
    sorted_keys = sorted(weight_map)

    sidecar_out.parent.mkdir(parents=True, exist_ok=True)
    merged = StreamingHasher()
    sidecar_file_hasher = hashlib.sha256()
    total_params = 0
    seen_dtypes: set[str] = set()
    open_handles: dict[str, safetensors.safe_open] = {}

    def _open(filename: str) -> safetensors.safe_open:
        if filename not in open_handles:
            open_handles[filename] = safetensors.safe_open(
                str(expert_dir / filename), framework="pt", device="cpu"
            )
        return open_handles[filename]

    try:
        with sidecar_out.open("wb") as sidecar_fh:
            for key in sorted_keys:
                shard_filename = weight_map[key]
                handle = _open(shard_filename)
                tensor = handle.get_tensor(key)
                shape = tuple(tensor.shape)
                dtype = str(tensor.dtype)
                seen_dtypes.add(dtype)
                total_params += int(tensor.numel())
                per_key_sha = merged.update(key, tensor)
                line = json.dumps(
                    {"key": key, "shape": list(shape), "dtype": dtype, "sha256": per_key_sha},
                    sort_keys=True,
                    ensure_ascii=True,
                    separators=(",", ":"),
                ).encode("ascii") + b"\n"
                sidecar_fh.write(line)
                sidecar_file_hasher.update(line)
                del tensor
    finally:
        for handle in open_handles.values():
            handle.__exit__(None, None, None) if hasattr(handle, "__exit__") else None

    sidecar_sha = sidecar_file_hasher.hexdigest()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "expert_tag": expert_tag,
        "expert_dir": str(expert_dir),
        "shard_index_filename": SHARD_INDEX_FILENAME,
        "shards": [dataclasses.asdict(s) for s in shards],
        "shard_count": len(shards),
        "tensor_count": merged.tensor_count,
        "total_parameter_count": total_params,
        "shard_dtype_observed": sorted(seen_dtypes),
        "intended_runtime_dtype": INTENDED_RUNTIME_DTYPE,
        "merged_state_sha256": merged.hexdigest(),
        "sidecar_jsonl_path": str(sidecar_out),
        "sidecar_jsonl_sha256": sidecar_sha,
        "recipe_id": recipe_id,
        "recipes_dir": str(recipes_dir),
        "code_commit_id": _git_commit_id(repo_root) if repo_root else None,
        "machine_internal_ip_tail": _internal_ip_tail(),
        "loader_module": __name__,
    }
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def cross_expert_keys_diff(expert_dir_a: pathlib.Path, expert_dir_b: pathlib.Path) -> dict:
    """Compare key sets and shard-index SHAs between two experts.

    No tensor bytes are loaded; only the two ``safetensors_index.json``
    files are read. Returns a dict with both key sets, the symmetric
    difference, the intersection size, and the index-file SHAs.

    A clean pair (e.g. high_noise_model vs low_noise_model under the same
    upstream) should report ``{"keys_only_in_a": [], "keys_only_in_b": []}``
    and identical ``index_sha256`` values, evidence that the experts
    share architecture and any divergence in the merged-state SHA is due
    to tensor bytes alone.
    """
    expert_dir_a = pathlib.Path(expert_dir_a).resolve()
    expert_dir_b = pathlib.Path(expert_dir_b).resolve()
    index_a = expert_dir_a / SHARD_INDEX_FILENAME
    index_b = expert_dir_b / SHARD_INDEX_FILENAME
    weight_map_a = json.loads(index_a.read_bytes())["weight_map"]
    weight_map_b = json.loads(index_b.read_bytes())["weight_map"]
    keys_a = set(weight_map_a)
    keys_b = set(weight_map_b)
    return {
        "expert_a": str(expert_dir_a),
        "expert_b": str(expert_dir_b),
        "key_count_a": len(keys_a),
        "key_count_b": len(keys_b),
        "keys_only_in_a": sorted(keys_a - keys_b),
        "keys_only_in_b": sorted(keys_b - keys_a),
        "intersection_size": len(keys_a & keys_b),
        "index_sha256_a": _file_sha256(index_a),
        "index_sha256_b": _file_sha256(index_b),
        "indexes_identical": _file_sha256(index_a) == _file_sha256(index_b),
    }


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[1] == "diff":
        if len(argv) != 4:
            print("usage: canonical_loader.py diff <expert_dir_a> <expert_dir_b>", file=sys.stderr)
            return 2
        report = cross_expert_keys_diff(pathlib.Path(argv[2]), pathlib.Path(argv[3]))
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if (not report["keys_only_in_a"] and not report["keys_only_in_b"]) else 1
    if len(argv) != 5:
        print(
            "usage: canonical_loader.py <expert_dir> <manifest_out> <sidecar_out> <recipes_dir>",
            file=sys.stderr,
        )
        print(
            "       canonical_loader.py diff <expert_dir_a> <expert_dir_b>",
            file=sys.stderr,
        )
        return 2
    expert_dir = pathlib.Path(argv[1])
    manifest_out = pathlib.Path(argv[2])
    sidecar_out = pathlib.Path(argv[3])
    recipes_dir = pathlib.Path(argv[4])
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    manifest = load_expert(expert_dir, manifest_out, sidecar_out, recipes_dir, repo_root=repo_root)
    print(json.dumps({"merged_state_sha256": manifest["merged_state_sha256"], "tensor_count": manifest["tensor_count"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
