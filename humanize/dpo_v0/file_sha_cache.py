"""Sidecar cache for file / shard sha256 used by all inference paths.

Cold sha256 over Wan2.2 high+low_noise shards is ~28 GB of disk IO and
adds ~6-10 min to every inference startup. The aggregated digest is
stamped into AC-7.3 manifests as `high_noise_base_sha256` /
`low_noise_frozen_sha256`, so callers cannot just skip it.

This module exposes:

* ``cached_file_sha256(path)`` -- drop-in for an uncached
  ``file_sha256``. Writes a sidecar JSON next to the file's directory
  keying by basename + (size, mtime_ns) -> sha256. On cache hit the
  per-file sha is reused; on miss it is recomputed and the cache is
  rewritten atomically.
* ``cached_sharded_ckpt_sha(shards)`` -- drop-in for the canonical
  ``sharded_ckpt_sha``. Same byte-stream aggregation
  (``<basename>|<file_sha>\\n`` per shard, alphabetical), so the
  resulting digest is identical to a cold recompute.

Cache invariant: (size, mtime_ns) match means the bytes are unchanged.
The OS reports nanosecond mtime on Linux, which is finer than any
realistic write-then-rename window for safetensors deploys; on shared
filesystems with coarser mtime resolution this remains correct because
we still recompute on every (size, mtime) miss. Safetensors / weight
files are write-once-then-frozen on every deploy we run, so cache
churn after the first warm-up is zero.

The eval / inference manifests are byte-equivalent to a cold-recompute
run, so existing AC-7.3 / paired-delta provenance is preserved. The
cache is purely a startup-time optimization shared by inference
entrypoints (rl5's external `inference_smoke.py`).
"""

from __future__ import annotations

import hashlib
import json
import pathlib

# Sidecar JSON sits next to the directory holding the hashed files.
# inference shards: <upstream_root>/{high,low}_noise_model/*.safetensors
#   -> cache at <upstream_root>/file_sha.cache.json
# T5 / VAE singletons: <upstream_root>/<file>
#   -> cache at <upstream_root>/file_sha.cache.json (same JSON, separate section)
CACHE_BASENAME = "file_sha.cache.json"

# 4 MiB read buffer; cold-miss timing matches the canonical file_sha256
# implementations used by callers (manifest_writer, encode_videos, etc.).
_BUF = 4 * 1024 * 1024


def _file_sha256_uncached(path: pathlib.Path, buf: int = _BUF) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_path_for(file_path: pathlib.Path) -> pathlib.Path:
    """Sidecar JSON lives one level above the file's parent dir.

    For shard files at ``<root>/<expert>/<shard>``, this returns
    ``<root>/file_sha.cache.json`` so high_noise + low_noise share one
    JSON. For loose files at ``<root>/<file>`` it returns
    ``<root>/file_sha.cache.json`` as well.
    """
    parent = file_path.parent
    if parent.name in {"high_noise_model", "low_noise_model"}:
        return parent.parent / CACHE_BASENAME
    return parent / CACHE_BASENAME


def _load(path: pathlib.Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"[file-sha-cache] unreadable {path}: {e}; ignoring", flush=True)
        return {}


def _save(path: pathlib.Path, data: dict) -> None:
    try:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
        tmp.replace(path)
    except OSError as e:
        print(f"[file-sha-cache] cannot persist {path}: {e}", flush=True)


def _section_key(file_path: pathlib.Path) -> str:
    return str(file_path.parent.resolve())


def cached_file_sha256(path: pathlib.Path) -> str:
    """Return sha256(path) using the sidecar cache when (size, mtime_ns) match."""
    cache_file = _cache_path_for(path)
    cache = _load(cache_file)
    section_key = _section_key(path)
    section = cache.get(section_key, {})

    st = path.stat()
    size_b, mtime_ns = st.st_size, st.st_mtime_ns
    rec = section.get(path.name)
    if (
        rec is not None
        and rec.get("size") == size_b
        and rec.get("mtime_ns") == mtime_ns
        and isinstance(rec.get("sha256"), str)
        and len(rec["sha256"]) == 64
    ):
        return rec["sha256"]

    sha = _file_sha256_uncached(path)
    section[path.name] = {"size": size_b, "mtime_ns": mtime_ns, "sha256": sha}
    cache[section_key] = section
    _save(cache_file, cache)
    return sha


def cached_sharded_ckpt_sha(shards: list[pathlib.Path]) -> str:
    """Aggregate sha matching the canonical sharded_ckpt_sha byte stream.

    Walks shards in alphabetical filename order and folds
    ``<basename>|<per_file_sha>\\n`` into a single sha256. Per-file sha
    is read from the sidecar cache; misses recompute and update the
    cache atomically. The aggregation byte stream (and therefore the
    resulting digest) is identical to a cold recompute, so AC-7.3
    manifest stamps remain byte-stable.
    """
    if not shards:
        return hashlib.sha256().hexdigest()

    sorted_shards = sorted(shards, key=lambda p: p.name)
    cache_file = _cache_path_for(sorted_shards[0])
    cache = _load(cache_file)
    section_key = _section_key(sorted_shards[0])
    section = cache.get(section_key, {})

    h = hashlib.sha256()
    cache_changed = False
    hits = misses = 0
    for s in sorted_shards:
        st = s.stat()
        size_b, mtime_ns = st.st_size, st.st_mtime_ns
        rec = section.get(s.name)
        if (
            rec is not None
            and rec.get("size") == size_b
            and rec.get("mtime_ns") == mtime_ns
            and isinstance(rec.get("sha256"), str)
            and len(rec["sha256"]) == 64
        ):
            sha = rec["sha256"]
            hits += 1
        else:
            sha = _file_sha256_uncached(s)
            section[s.name] = {"size": size_b, "mtime_ns": mtime_ns, "sha256": sha}
            cache_changed = True
            misses += 1
        h.update(s.name.encode("utf-8"))
        h.update(b"|")
        h.update(sha.encode("ascii"))
        h.update(b"\n")

    if cache_changed:
        cache[section_key] = section
        _save(cache_file, cache)

    digest = h.hexdigest()
    print(
        f"[file-sha-cache] {sorted_shards[0].parent.name}: "
        f"hits={hits} misses={misses} -> {digest[:12]}...",
        flush=True,
    )
    return digest


__all__ = [
    "CACHE_BASENAME",
    "cached_file_sha256",
    "cached_sharded_ckpt_sha",
]
