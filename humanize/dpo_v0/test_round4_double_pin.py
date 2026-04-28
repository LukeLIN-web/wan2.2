"""Tests for round-4 training_config + pair_ids double-pin (rl2 spec b98b72b1).

Validates the three pin-assert helpers in train_dpo_i2v.py:
  - assert_recipe_pin (existing, regression check)
  - assert_training_config_pin (new)
  - assert_pair_ids_pin (new)

Also smoke-tests the trainer CLI parser + canonicality of the round-4
training_config.
"""

import hashlib
import pathlib
import subprocess
import sys
import tempfile

import pytest
import yaml

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import train_dpo_i2v as trainer  # noqa: E402

RECIPES_DIR = HERE / "recipes"


def test_assert_recipe_pin_passes_round2_immutable():
    rid = trainer.assert_recipe_pin(RECIPES_DIR, expected="6bef6e104cdd3442")
    assert rid == "6bef6e104cdd3442"


def test_assert_recipe_pin_drift_raises():
    with pytest.raises(AssertionError, match="recipe pin drift"):
        trainer.assert_recipe_pin(RECIPES_DIR, expected="0000000000000000")


def test_assert_training_config_pin_round4():
    config_path = RECIPES_DIR / "training_config_round4.yaml"
    pin_file = RECIPES_DIR / "training_config_sha256_pin"
    expected = pin_file.read_text(encoding="ascii").strip()
    config = trainer.assert_training_config_pin(config_path, expected)
    assert config["lr"] == 5.0e-5
    assert config["max_steps"] == 200
    assert config["max_pairs"] == 1000
    assert config["beta"] == 0.1
    assert config["lora_rank"] == 16
    assert config["lora_alpha"] == 16
    assert config["seed_namespace"] == "round4-tier_b-1k"
    assert config["sampling_band"] == [901, 999]
    assert config["round_tag"] == "round-4"
    assert config["subset_pair_ids_sha256_hex16"] == "cf5d3e5fd528a3e0"


def test_assert_training_config_pin_drift_raises():
    config_path = RECIPES_DIR / "training_config_round4.yaml"
    with pytest.raises(AssertionError, match="training_config pin drift"):
        trainer.assert_training_config_pin(config_path, "0000000000000000")


def test_assert_training_config_pin_modified_yaml_raises(tmp_path):
    config_path = RECIPES_DIR / "training_config_round4.yaml"
    real_pin = (RECIPES_DIR / "training_config_sha256_pin").read_text(encoding="ascii").strip()
    # Corrupt the YAML by appending a comment (whitespace-only would not change hash).
    altered = config_path.read_bytes() + b"# tampered\n"
    altered_path = tmp_path / "training_config_round4.yaml"
    altered_path.write_bytes(altered)
    with pytest.raises(AssertionError, match="training_config pin drift"):
        trainer.assert_training_config_pin(altered_path, real_pin)


def test_assert_pair_ids_pin_canonical_form():
    pair_ids = ["a__1_gt_2", "b__3_gt_4", "c__5_gt_6"]
    canonical = ("\n".join(pair_ids) + "\n").encode("utf-8")
    expected = hashlib.sha256(canonical).hexdigest()[:16]
    fresh = trainer.assert_pair_ids_pin(pair_ids, expected)
    assert fresh == expected


def test_assert_pair_ids_pin_drift_raises():
    pair_ids = ["a__1_gt_2", "b__3_gt_4"]
    with pytest.raises(AssertionError, match="pair_ids pin drift"):
        trainer.assert_pair_ids_pin(pair_ids, "0000000000000000")


def test_assert_pair_ids_pin_order_sensitive():
    """Reordering pair_ids changes the canonical hash (intentional — the seed-shuffled
    order from build_round4_tier_b_1k.py is part of the contract)."""
    a = ["x__1_gt_2", "y__3_gt_4"]
    b = ["y__3_gt_4", "x__1_gt_2"]
    pin_a = hashlib.sha256(("\n".join(a) + "\n").encode("utf-8")).hexdigest()[:16]
    # b should not match a's pin
    with pytest.raises(AssertionError, match="pair_ids pin drift"):
        trainer.assert_pair_ids_pin(b, pin_a)


def test_round4_yaml_sampling_band_matches_trainer_constants():
    """training_config_round4.yaml sampling_band must match trainer hardcoded SAMPLING_T_*."""
    config = yaml.safe_load((RECIPES_DIR / "training_config_round4.yaml").read_bytes())
    assert config["sampling_band"] == [trainer.SAMPLING_T_LOW, trainer.SAMPLING_T_HIGH]


def test_round4_yaml_subset_pin_matches_build_script_output():
    """training_config_round4.yaml subset_pair_ids_sha256_hex16 must match what
    the round-1/2 #19 build script produced. This is a static cross-check that
    catches drift between the recipe and the actual subset that was built."""
    config = yaml.safe_load((RECIPES_DIR / "training_config_round4.yaml").read_bytes())
    assert config["subset_pair_ids_sha256_hex16"] == "cf5d3e5fd528a3e0"


def test_trainer_cli_rejects_partial_round4_args(tmp_path):
    """--training-config-path without --training-config-sha256-pin must fail loud,
    so an operator cannot accidentally bypass the pin."""
    fake_yaml = tmp_path / "fake.yaml"
    fake_yaml.write_text("lr: 1.0e-5\n")
    fake_subset = tmp_path / "fake_subset.json"
    fake_subset.write_text("{}")
    fake_pair_json = tmp_path / "fake_pair.json"
    fake_pair_json.write_text("[]")
    fake_image_manifest = tmp_path / "fake_image_manifest.json"
    fake_image_manifest.write_text("{}")
    fake_latent_manifest = tmp_path / "fake_latent_manifest.jsonl"
    fake_latent_manifest.touch()

    cmd = [
        sys.executable, str(HERE / "train_dpo_i2v.py"),
        "--latent-manifest", str(fake_latent_manifest),
        "--post-t2-pair", str(fake_pair_json),
        "--t2-image-manifest", str(fake_image_manifest),
        "--training-config-path", str(fake_yaml),
        # intentionally NOT passing --training-config-sha256-pin
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    combined = (result.stderr + result.stdout).lower()
    assert "training-config-sha256-pin" in combined or "both required together" in combined


def test_trainer_cli_rejects_partial_subset_args(tmp_path):
    fake_subset = tmp_path / "fake_subset.json"
    fake_subset.write_text('{"tier_b_round4_1k": {"pair_ids": []}}')
    fake_pair_json = tmp_path / "fake_pair.json"
    fake_pair_json.write_text("[]")
    fake_image_manifest = tmp_path / "fake_image_manifest.json"
    fake_image_manifest.write_text("{}")
    fake_latent_manifest = tmp_path / "fake_latent_manifest.jsonl"
    fake_latent_manifest.touch()

    cmd = [
        sys.executable, str(HERE / "train_dpo_i2v.py"),
        "--latent-manifest", str(fake_latent_manifest),
        "--post-t2-pair", str(fake_pair_json),
        "--t2-image-manifest", str(fake_image_manifest),
        "--subset-pair-ids-json", str(fake_subset),
        # intentionally NOT passing --pair-ids-sha256-pin
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    combined = (result.stderr + result.stdout).lower()
    assert "pair-ids-sha256-pin" in combined or "both required together" in combined
