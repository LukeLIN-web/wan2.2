"""Canonical serializer for the v0 video_preprocessing recipe.

The serializer emits a byte-for-byte stable YAML representation of the
preprocessing recipe so that ``recipe_id = sha256(canonical_yaml.bytes)[:16]``
is reproducible across machines, Python versions, and PyYAML patch revisions
(within a major-version pin).

Canonicality rules (rl2 acceptance, msg ``22e2a02c``):

* sorted top-level + nested keys (``sort_keys=True``)
* block style only, no flow-style (``default_flow_style=False``)
* ASCII output, no unicode escapes (``allow_unicode=False``)
* unbounded line width, never wrap (``width=2**31 - 1``)
* two-space indent (``indent=2``)
* unix line endings (``line_break='\n'``)
* PyYAML major version pinned at 6.x
* trailing newline preserved as-is from ``yaml.safe_dump``

Any change to any of these knobs changes the bytes and therefore the id.

Boolean values are emitted as ``true`` / ``false`` (PyYAML 6.x default).
Strings starting with a digit but containing hyphens or dots are emitted
unquoted by PyYAML 6.x; this is checked at serialization time and the
output bytes are hashed verbatim.

Usage::

    python canonical_serializer.py emit       # writes recipe.yaml + recipe_id
    python canonical_serializer.py verify     # checks the on-disk recipe_id
                                              # matches a fresh hash
"""

from __future__ import annotations

import hashlib
import pathlib
import sys

import yaml

YAML_MAJOR_REQUIRED = 6
RECIPE_FILENAME = "wan22_i2v_a14b__round2_v0.yaml"
RECIPE_ID_FILENAME = "recipe_id"
HERE = pathlib.Path(__file__).resolve().parent

RECIPE = {
    "codec_normalization": "yuv420p_to_rgb_bt709_full",
    "color_space": "bt709-tv-range",
    "decoder": {
        "ffmpeg": "4.4.2-0ubuntu0.22.04.1",
        "imageio": "2.37.3",
        "imageio_ffmpeg": "0.6.0",
    },
    "first_frame_is_conditioning": True,
    "fps": 16,
    "frame_count_policy": "first_n",
    "frame_num": 81,
    "frame_stride": 1,
    "pad_color": [0, 0, 0],
    "resize_mode": "letterbox_pad",
    "target_resolution": {
        "aspect_ratio_router": {
            "landscape": "832x480",
            "portrait": "480x832",
        },
    },
}

REQUIRED_AXES = (
    "codec_normalization",
    "color_space",
    "decoder",
    "first_frame_is_conditioning",
    "fps",
    "frame_count_policy",
    "frame_num",
    "frame_stride",
    "pad_color",
    "resize_mode",
    "target_resolution",
)


def _check_yaml_version() -> None:
    major = int(yaml.__version__.split(".")[0])
    if major != YAML_MAJOR_REQUIRED:
        raise RuntimeError(
            f"PyYAML major version mismatch: required {YAML_MAJOR_REQUIRED}, got {yaml.__version__}"
        )


def canonical_bytes(recipe: dict) -> bytes:
    """Serialize ``recipe`` to canonical YAML bytes."""
    _check_yaml_version()
    missing = [axis for axis in REQUIRED_AXES if axis not in recipe]
    if missing:
        raise ValueError(f"recipe is missing required axes: {missing}")
    text = yaml.safe_dump(
        recipe,
        sort_keys=True,
        default_flow_style=False,
        allow_unicode=False,
        width=2**31 - 1,
        indent=2,
        line_break="\n",
    )
    return text.encode("utf-8")


def recipe_id(recipe_bytes: bytes) -> str:
    """Return the 16-hex-char recipe id derived from the serialized bytes."""
    return hashlib.sha256(recipe_bytes).hexdigest()[:16]


def emit(out_dir: pathlib.Path = HERE) -> tuple[pathlib.Path, pathlib.Path, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    recipe_path = out_dir / RECIPE_FILENAME
    id_path = out_dir / RECIPE_ID_FILENAME
    data = canonical_bytes(RECIPE)
    recipe_id_value = recipe_id(data)
    recipe_path.write_bytes(data)
    id_path.write_text(recipe_id_value + "\n", encoding="ascii")
    return recipe_path, id_path, recipe_id_value


def verify(out_dir: pathlib.Path = HERE) -> tuple[bool, str, str]:
    recipe_path = out_dir / RECIPE_FILENAME
    id_path = out_dir / RECIPE_ID_FILENAME
    on_disk_recipe = recipe_path.read_bytes()
    on_disk_id = id_path.read_text(encoding="ascii").strip()
    fresh_id = recipe_id(on_disk_recipe)
    return on_disk_id == fresh_id, on_disk_id, fresh_id


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in {"emit", "verify"}:
        print(__doc__, file=sys.stderr)
        return 2
    if argv[1] == "emit":
        recipe_path, id_path, value = emit()
        print(f"wrote {recipe_path}")
        print(f"wrote {id_path}")
        print(f"recipe_id = {value}")
        return 0
    ok, on_disk, fresh = verify()
    if ok:
        print(f"OK: {on_disk}")
        return 0
    print(f"MISMATCH: on_disk={on_disk} fresh={fresh}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
