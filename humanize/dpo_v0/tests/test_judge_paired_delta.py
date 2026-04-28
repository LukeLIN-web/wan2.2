"""Unit tests for the PhyJudge-9B paired-delta scoring module.

Covers the round-1 contract: probe-time axis mapping (identity match,
operator alias, halt-on-missing), per-leg result loading (halt on
missing axis / non-numeric value / asymmetric secondary set), pair
join (inner-join semantics; surfaced unmatched ids), paired bootstrap
(determinism under fixed seed; different axes get independently
reproducible CIs; sign correctness on a contrived delta), composite
arithmetic, manifest stamping, and CLI failure-record writing.

Numerical tolerances on the bootstrap stats are loose because the
contract under test is *deterministic for a fixed seed and direction
correctness*, not a specific numeric value.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # humanize/dpo_v0/
from eval import judge_paired_delta as jpd  # noqa: E402


# --- helpers ----------------------------------------------------------------

def _phyjudge_record(
    video_id: str,
    sa: float,
    ptv: float,
    persistence: float,
    physical_avg: float = 5.0,
    extra: dict | None = None,
) -> dict:
    """Construct a per-result object shaped like the PhyJudge-9B harness output."""
    record = {
        "video": video_id,
        "SA": sa,
        "PTV": ptv,
        "persistence": persistence,
        "general_avg": (sa + ptv + persistence) / 3.0,
        "physical": {
            "laws": {"collision": {"score": physical_avg, "status": "scored"}},
            "missing_laws": [],
            "coverage": 1.0,
            "avg": physical_avg,
        },
        "raw_response_general": "",
        "raw_response_physical": "",
        "rationale_general": "",
        "rationale_physical": "",
        "usage": {"general": 0, "physical": 0, "total": 0},
        "prompt": "",
        "physical_laws": ["collision"],
    }
    if extra:
        record.update(extra)
    return record


def _phyjudge_payload(records: list[dict]) -> dict:
    """Wrap records in the harness's top-level eval-JSON envelope."""
    return {
        "meta": {
            "type": "eval_result",
            "prompt_config": "subq+human.yaml",
            "prompt_mode": "eval_prompts",
            "scheme": "subq_hint",
            "evaluator": "qwen9b",
            "subset": "humaneval_set",
            "video_model": "wan22-i2v-a14b-orig-init",
            "eval_timestamp": "20260427_220000",
            "source_filename": "test.json",
        },
        "num_videos": len(records),
        "general_dimensions": ["SA", "PTV", "persistence"],
        "results": records,
        "judge": "phyjudge_9b",
        "visual_config": {},
        "inference_config": {},
        "prompt_config": "subq+human.yaml",
        "prompt_mode": "eval_prompts",
    }


def _write_json(tmp_path: pathlib.Path, name: str, payload: dict) -> pathlib.Path:
    p = tmp_path / name
    p.write_text(json.dumps(payload))
    return p


# --- probe tests ------------------------------------------------------------

def test_probe_identity_mapping_resolves(tmp_path):
    payload = _phyjudge_payload([_phyjudge_record("v0", 4, 3, 5)])
    probe = _write_json(tmp_path, "probe.json", payload)
    field_names, mapping = jpd.probe_phyjudge_axes(probe)
    assert "SA" in field_names
    assert mapping.axis_key("SA") == "SA"
    assert mapping.axis_key("PTV") == "PTV"
    assert mapping.axis_key("persistence") == "persistence"


def test_probe_alias_mapping_resolves(tmp_path):
    record = _phyjudge_record("v0", 4, 3, 5)
    record["alignment_score"] = record.pop("SA")
    payload = _phyjudge_payload([record])
    probe = _write_json(tmp_path, "probe.json", payload)
    field_names, mapping = jpd.probe_phyjudge_axes(
        probe, axis_mapping_override={"SA": "alignment_score"}
    )
    assert mapping.axis_key("SA") == "alignment_score"
    assert "alignment_score" in field_names


def test_probe_halts_on_missing_axis(tmp_path):
    record = _phyjudge_record("v0", 4, 3, 5)
    del record["persistence"]
    payload = _phyjudge_payload([record])
    probe = _write_json(tmp_path, "probe.json", payload)
    with pytest.raises(jpd.JudgeAxisMissingError, match="persistence"):
        jpd.probe_phyjudge_axes(probe)


def test_probe_halts_on_alias_pointing_to_nonexistent_key(tmp_path):
    payload = _phyjudge_payload([_phyjudge_record("v0", 4, 3, 5)])
    probe = _write_json(tmp_path, "probe.json", payload)
    with pytest.raises(jpd.JudgeAxisMissingError, match="alignment_score"):
        jpd.probe_phyjudge_axes(
            probe, axis_mapping_override={"SA": "alignment_score"}
        )


def test_probe_halts_on_non_numeric_axis(tmp_path):
    record = _phyjudge_record("v0", 4, 3, 5)
    record["SA"] = "high"
    payload = _phyjudge_payload([record])
    probe = _write_json(tmp_path, "probe.json", payload)
    with pytest.raises(jpd.JudgeAxisMissingError, match="not numeric"):
        jpd.probe_phyjudge_axes(probe)


def test_probe_halts_on_empty_results(tmp_path):
    payload = _phyjudge_payload([])
    probe = _write_json(tmp_path, "probe.json", payload)
    with pytest.raises(jpd.JudgeAxisMissingError, match="no results"):
        jpd.probe_phyjudge_axes(probe)


def test_secondary_axis_keys_excludes_primaries_and_general_avg():
    record = _phyjudge_record("v0", 4, 3, 5)
    record["physical_score"] = 4.5
    mapping = jpd.AxisMapping(mapping={
        "SA": "SA", "PTV": "PTV", "persistence": "persistence",
    })
    keys = jpd.secondary_axis_keys(record, mapping)
    assert "SA" not in keys
    assert "PTV" not in keys
    assert "persistence" not in keys
    assert "general_avg" not in keys
    assert "physical_score" in keys
    # Non-numeric / dict / list fields are excluded.
    assert "physical" not in keys
    assert "physical_laws" not in keys
    assert "video" not in keys


# --- load tests -------------------------------------------------------------

def test_load_normal_path(tmp_path):
    records = [
        _phyjudge_record("v0", 4, 3, 5),
        _phyjudge_record("v1", 2, 1, 3),
    ]
    payload = _phyjudge_payload(records)
    eval_path = _write_json(tmp_path, "eval.json", payload)
    mapping = jpd.AxisMapping(mapping={
        "SA": "SA", "PTV": "PTV", "persistence": "persistence",
    })
    loaded = jpd.load_eval_results(eval_path, mapping, secondary_axis_keys_list=[])
    assert len(loaded) == 2
    assert loaded[0].video_id == "v0"
    assert loaded[0].composite == 4 + 3 + 5
    assert loaded[1].composite == 2 + 1 + 3


def test_load_halts_on_duplicate_video_id(tmp_path):
    records = [
        _phyjudge_record("v0", 4, 3, 5),
        _phyjudge_record("v0", 5, 5, 5),  # dup
    ]
    payload = _phyjudge_payload(records)
    eval_path = _write_json(tmp_path, "eval.json", payload)
    mapping = jpd.AxisMapping(mapping={
        "SA": "SA", "PTV": "PTV", "persistence": "persistence",
    })
    with pytest.raises(jpd.JudgeAxisMissingError, match="duplicate"):
        jpd.load_eval_results(eval_path, mapping, secondary_axis_keys_list=[])


def test_load_halts_on_missing_secondary_axis(tmp_path):
    records = [_phyjudge_record("v0", 4, 3, 5)]
    payload = _phyjudge_payload(records)
    eval_path = _write_json(tmp_path, "eval.json", payload)
    mapping = jpd.AxisMapping(mapping={
        "SA": "SA", "PTV": "PTV", "persistence": "persistence",
    })
    with pytest.raises(jpd.JudgeAxisMissingError, match="bogus"):
        jpd.load_eval_results(eval_path, mapping, secondary_axis_keys_list=["bogus"])


# --- pair tests -------------------------------------------------------------

def _record_obj(video_id: str, sa: float, ptv: float, pers: float) -> jpd.JudgedRecord:
    return jpd.JudgedRecord(
        video_id=video_id,
        primary_axes={"SA": sa, "PTV": ptv, "persistence": pers},
        secondary_axes={},
        raw={},
    )


def test_pair_inner_join_and_unmatched():
    baseline = [_record_obj("a", 1, 1, 1), _record_obj("b", 2, 2, 2),
                _record_obj("c", 3, 3, 3)]
    trained = [_record_obj("a", 4, 4, 4), _record_obj("b", 2, 2, 2),
               _record_obj("d", 5, 5, 5)]
    deltas, b_only, t_only = jpd.pair_records_by_video_id(baseline, trained)
    assert sorted(d.video_id for d in deltas) == ["a", "b"]
    assert b_only == ["c"]
    assert t_only == ["d"]
    pa = next(d for d in deltas if d.video_id == "a")
    assert pa.composite_delta == (4 + 4 + 4) - (1 + 1 + 1)
    pb = next(d for d in deltas if d.video_id == "b")
    assert pb.composite_delta == 0


def test_pair_halts_on_asymmetric_secondary_axes():
    b = jpd.JudgedRecord(video_id="a", primary_axes={"SA": 1, "PTV": 1, "persistence": 1},
                         secondary_axes={"x": 1.0}, raw={})
    t = jpd.JudgedRecord(video_id="a", primary_axes={"SA": 1, "PTV": 1, "persistence": 1},
                         secondary_axes={"y": 1.0}, raw={})
    with pytest.raises(jpd.JudgeAxisMissingError, match="asymmetric secondary"):
        jpd.pair_records_by_video_id([b], [t])


def test_video_paired_list_loads_flat_paths_and_prompt_source(tmp_path):
    b = tmp_path / "b.mp4"
    t = tmp_path / "t.mp4"
    b.write_bytes(b"b")
    t.write_bytes(b"t")
    paired = tmp_path / "pairs.json"
    paired.write_text(json.dumps({
        "pairs": [{
            "scene_id": "collision_1",
            "baseline_video_path": str(b),
            "trained_video_path": str(t),
        }]
    }))
    prompt_source = _write_json(tmp_path, "prompts.json", {
        "prompts": [{
            "video": "collision_1",
            "prompt": "A ball hits a wall.",
            "physical_laws": ["collision"],
        }]
    })

    pairs = jpd.load_video_paired_list(paired, prompt_source_json=prompt_source)
    assert len(pairs) == 1
    assert pairs[0].video_id == "collision_1"
    assert pairs[0].prompt == "A ball hits a wall."
    assert pairs[0].physical_laws == ["collision"]
    assert pairs[0].baseline_video_path == b
    assert pairs[0].trained_video_path == t


def test_video_paired_list_loads_heldout_regen_nested_out_dirs(tmp_path):
    base_dir = tmp_path / "baseline"
    train_dir = tmp_path / "trained"
    base_dir.mkdir()
    train_dir.mkdir()
    (base_dir / "video.mp4").write_bytes(b"b")
    (train_dir / "video.mp4").write_bytes(b"t")
    paired = tmp_path / "run_manifest.json"
    paired.write_text(json.dumps({
        "results": [{
            "prompt_id": "abc123",
            "prompt": "A toy car rolls downhill.",
            "baseline": {"out_dir": str(base_dir)},
            "trained": {"out_dir": str(train_dir)},
            "physical_laws": "gravity,inertia",
        }]
    }))

    pairs = jpd.load_video_paired_list(paired)
    assert pairs[0].video_id == "abc123"
    assert pairs[0].physical_laws == ["gravity", "inertia"]
    assert pairs[0].baseline_video_path == base_dir / "video.mp4"
    assert pairs[0].trained_video_path == train_dir / "video.mp4"


def test_prepare_video_pair_eval_inputs_writes_prompts_and_video_dirs(tmp_path):
    b = tmp_path / "baseline_src.mp4"
    t = tmp_path / "trained_src.mp4"
    b.write_bytes(b"b")
    t.write_bytes(b"t")
    pairs = [jpd.VideoPair(
        video_id="scene_1",
        prompt="A block falls.",
        baseline_video_path=b,
        trained_video_path=t,
        physical_laws=["gravity"],
        dataset="wmb",
    )]

    prompts_json, baseline_dir, trained_dir = jpd.prepare_video_pair_eval_inputs(
        pairs, tmp_path / "work"
    )
    payload = json.loads(prompts_json.read_text())
    assert payload["prompts"][0]["video"] == "scene_1"
    assert payload["prompts"][0]["physical_laws"] == ["gravity"]
    assert (baseline_dir / "scene_1.mp4").exists()
    assert (trained_dir / "scene_1.mp4").exists()


# --- bootstrap tests --------------------------------------------------------

def test_bootstrap_zero_deltas_zero_mean():
    """All-zero deltas → mean exactly 0; CI both ends near 0."""
    deltas = [0.0] * 10
    ci = jpd.paired_bootstrap_ci(deltas, n_iters=200, seed_value="test")
    assert ci.mean == 0.0
    assert ci.std == 0.0
    assert ci.ci_low == 0.0
    assert ci.ci_high == 0.0


def test_bootstrap_positive_direction():
    """Strictly-positive deltas → mean > 0 and both CI ends > 0."""
    deltas = [0.5] * 5 + [1.0] * 5
    ci = jpd.paired_bootstrap_ci(deltas, n_iters=2000, seed_value="dir-test")
    assert ci.mean == pytest.approx(0.75)
    assert ci.ci_low > 0
    assert ci.ci_high > 0
    assert ci.ci_low <= ci.mean <= ci.ci_high


def test_bootstrap_seed_determinism():
    """Same seed + same deltas → identical CI."""
    deltas = [0.1, -0.2, 0.3, 0.5, -0.1, 0.0]
    a = jpd.paired_bootstrap_ci(deltas, n_iters=1000, seed_value="abc")
    b = jpd.paired_bootstrap_ci(deltas, n_iters=1000, seed_value="abc")
    assert a == b


def test_bootstrap_different_seeds_differ():
    """Different seeds for the same data should usually yield different CIs."""
    deltas = [0.1, -0.2, 0.3, 0.5, -0.1, 0.0]
    a = jpd.paired_bootstrap_ci(deltas, n_iters=500, seed_value="abc")
    b = jpd.paired_bootstrap_ci(deltas, n_iters=500, seed_value="xyz")
    # Mean is invariant (same data); CIs come from sample-mean percentiles
    # which depend on seed.
    assert a.mean == b.mean
    assert (a.ci_low, a.ci_high) != (b.ci_low, b.ci_high)


def test_bootstrap_rejects_too_few_pairs():
    with pytest.raises(ValueError, match=">=2 pairs"):
        jpd.paired_bootstrap_ci([1.0], n_iters=10, seed_value="x")


def test_bootstrap_per_axis_seed_independence():
    """Composite + secondary CIs are computed against per-axis seeds, so two
    secondary axes with identical per-pair deltas still get reproducible
    (and structurally identical) CIs without leaking RNG state."""
    deltas = [
        jpd.PairedDelta(video_id=f"v{i}", composite_delta=float(i),
                        secondary_deltas={"alpha": float(i), "beta": float(i)})
        for i in range(8)
    ]
    composite_ci, secondary_cis = jpd.bootstrap_all_axes(
        deltas, n_iters=400, seed_value="topseed",
    )
    # Both secondary axes have identical underlying delta vectors but
    # use different per-axis seeds; sanity-check that they produce
    # finite, ordered CIs.
    assert set(secondary_cis) == {"alpha", "beta"}
    for ci in secondary_cis.values():
        assert ci.ci_low <= ci.mean <= ci.ci_high


# --- composite tests --------------------------------------------------------

def test_composite_uses_three_axes():
    rec = _record_obj("v", 1.0, 2.0, 3.0)
    assert rec.composite == 6.0


def test_composite_halts_on_missing_axis():
    rec = jpd.JudgedRecord(video_id="v", primary_axes={"SA": 1.0, "PTV": 2.0},
                           secondary_axes={}, raw={})
    with pytest.raises(jpd.JudgeAxisMissingError, match="persistence"):
        _ = rec.composite


# --- manifest + md tests ----------------------------------------------------

def test_manifest_stamps_required_fields():
    mapping = jpd.AxisMapping(mapping={"SA": "SA", "PTV": "PTV",
                                       "persistence": "persistence"})
    composite_ci = jpd.BootstrapCI(mean=1.0, std=0.5, ci_low=0.0, ci_high=2.0)
    manifest = jpd.stamp_manifest(
        field_names=["SA", "PTV", "persistence", "video"],
        axis_mapping=mapping,
        secondary_axis_keys_list=["physical_score"],
        composite_ci=composite_ci,
        secondary_cis={},
        n_pairs=42,
        baseline_only_ids=[],
        trained_only_ids=[],
        bootstrap_iters=10000,
        bootstrap_seed="0xdpo_judge",
        baseline_ckpt_path="/path/to/baseline",
        trained_ckpt_path="/path/to/trained",
        generation_config={"seed": 0, "inference_steps": 50},
        comparator_pair="orig vs orig+dpo",
        compute_envelope="single_gpu",
        commit_id="deadbeef",
        machine_ip_tail="42",
    )
    assert manifest["recipe_id"] == jpd.KNOWN_GOOD_RECIPE_ID
    assert manifest["aggregation_rule"] == "cross_group_rater_union"
    assert manifest["judge"]["composite_definition"] == "SA+PTV+persistence"
    assert manifest["judge"]["axis_mapping"] == {
        "SA": "SA", "PTV": "PTV", "persistence": "persistence",
    }
    assert manifest["bootstrap"]["iters"] == 10000
    assert manifest["bootstrap"]["seed"] == "0xdpo_judge"
    assert manifest["bootstrap"]["alpha"] == 0.05
    assert manifest["compute_envelope"] == "single_gpu"
    assert manifest["commit_id"] == "deadbeef"
    assert manifest["machine_ip_tail"] == "42"
    assert manifest["results"]["primary_composite"]["mean"] == 1.0
    # Generation config sha is stable for identical inputs.
    sha = manifest["generation_config"]["sha256_short"]
    again = jpd.stamp_manifest(
        field_names=["SA", "PTV", "persistence", "video"],
        axis_mapping=mapping,
        secondary_axis_keys_list=["physical_score"],
        composite_ci=composite_ci,
        secondary_cis={},
        n_pairs=42,
        baseline_only_ids=[],
        trained_only_ids=[],
        bootstrap_iters=10000,
        bootstrap_seed="0xdpo_judge",
        baseline_ckpt_path="/path/to/baseline",
        trained_ckpt_path="/path/to/trained",
        generation_config={"inference_steps": 50, "seed": 0},  # reordered
        comparator_pair="orig vs orig+dpo",
        compute_envelope="single_gpu",
        commit_id="deadbeef",
        machine_ip_tail="42",
    )
    assert again["generation_config"]["sha256_short"] == sha


def test_write_results_md_contains_required_blocks(tmp_path):
    mapping = jpd.AxisMapping(mapping={"SA": "SA", "PTV": "PTV",
                                       "persistence": "persistence"})
    composite_ci = jpd.BootstrapCI(mean=1.5, std=0.5, ci_low=0.5, ci_high=2.5)
    secondary = {"physical_score": jpd.BootstrapCI(mean=0.2, std=0.1,
                                                    ci_low=-0.1, ci_high=0.5)}
    manifest = jpd.stamp_manifest(
        field_names=["SA", "PTV", "persistence", "video", "physical_score"],
        axis_mapping=mapping,
        secondary_axis_keys_list=["physical_score"],
        composite_ci=composite_ci,
        secondary_cis=secondary,
        n_pairs=42,
        baseline_only_ids=[],
        trained_only_ids=["x"],
        bootstrap_iters=10000,
        bootstrap_seed="0xdpo_judge",
        baseline_ckpt_path="/path/to/baseline",
        trained_ckpt_path="/path/to/trained",
        generation_config={"seed": 0},
        comparator_pair="orig vs orig+dpo",
        compute_envelope="single_gpu",
        commit_id="deadbeef",
        machine_ip_tail="42",
    )
    out_md = tmp_path / "out.md"
    jpd.write_results_md(out_md, manifest)
    text = out_md.read_text()
    assert "SA + PTV + persistence" in text
    assert "orig vs orig+dpo" in text
    assert jpd.KNOWN_GOOD_RECIPE_ID in text
    assert "cross_group_rater_union" in text
    assert "physical_score" in text
    assert "1.5000" in text  # composite mean rendered to 4dp


# --- end-to-end (CLI-shaped) tests ------------------------------------------

def test_main_writes_failure_manifest_on_missing_axis(tmp_path):
    bad_record = _phyjudge_record("v0", 4, 3, 5)
    del bad_record["persistence"]
    probe = _write_json(tmp_path, "probe.json", _phyjudge_payload([bad_record]))
    baseline = _write_json(tmp_path, "baseline.json",
                           _phyjudge_payload([_phyjudge_record("v0", 4, 3, 5)]))
    trained = _write_json(tmp_path, "trained.json",
                          _phyjudge_payload([_phyjudge_record("v0", 5, 5, 5)]))
    gen_cfg = _write_json(tmp_path, "gen.json", {"seed": 0})
    out_manifest = tmp_path / "out_manifest.json"
    out_md = tmp_path / "out.md"

    rc = jpd.main([
        "--probe-results-json", str(probe),
        "--baseline-results-json", str(baseline),
        "--trained-results-json", str(trained),
        "--baseline-ckpt-path", "/b", "--trained-ckpt-path", "/t",
        "--generation-config-json", str(gen_cfg),
        "--out-manifest", str(out_manifest), "--out-md", str(out_md),
        "--bootstrap-iters", "100",
    ])
    assert rc == 3
    failure = json.loads(out_manifest.read_text())["failure"]
    assert failure["kind"] == "judge-axis-missing"
    assert "persistence" in failure["message"]


def test_main_happy_path(tmp_path):
    """End-to-end skeleton run with synthetic 3-pair data."""
    records_baseline = [_phyjudge_record(f"v{i}", 2, 2, 2) for i in range(3)]
    records_trained = [_phyjudge_record(f"v{i}", 3, 3, 3) for i in range(3)]
    probe = _write_json(tmp_path, "probe.json",
                        _phyjudge_payload([_phyjudge_record("vp", 4, 4, 4)]))
    baseline = _write_json(tmp_path, "baseline.json",
                           _phyjudge_payload(records_baseline))
    trained = _write_json(tmp_path, "trained.json",
                          _phyjudge_payload(records_trained))
    gen_cfg = _write_json(tmp_path, "gen.json",
                          {"seed": 0, "inference_steps": 50})
    out_manifest = tmp_path / "out_manifest.json"
    out_md = tmp_path / "out.md"

    rc = jpd.main([
        "--probe-results-json", str(probe),
        "--baseline-results-json", str(baseline),
        "--trained-results-json", str(trained),
        "--baseline-ckpt-path", "/baseline_ckpt",
        "--trained-ckpt-path", "/trained_ckpt",
        "--generation-config-json", str(gen_cfg),
        "--out-manifest", str(out_manifest), "--out-md", str(out_md),
        "--bootstrap-iters", "200",
        "--bootstrap-seed", "happy-path",
        "--commit-id", "abc1234",
        "--machine-ip-tail", "99",
    ])
    assert rc == 0
    manifest = json.loads(out_manifest.read_text())
    assert manifest["bootstrap"]["n_pairs_used"] == 3
    assert manifest["results"]["primary_composite"]["mean"] == pytest.approx(3.0)  # +1 SA + +1 PTV + +1 persistence per pair = +3
    assert manifest["recipe_id"] == jpd.KNOWN_GOOD_RECIPE_ID
    assert manifest["aggregation_rule"] == "cross_group_rater_union"
    assert manifest["commit_id"] == "abc1234"
    assert manifest["machine_ip_tail"] == "99"
    md_text = out_md.read_text()
    assert "3.0000" in md_text
    assert "abc1234" in md_text
