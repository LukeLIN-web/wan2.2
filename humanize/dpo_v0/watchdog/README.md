# Trainer-side runtime watchdog

Three independent live tails for the I2V DPO trainer:

- **VRAM peak per step + per stage** — `vram_probe.py`. Wraps
  `torch.cuda.reset_peak_memory_stats()` + `max_memory_allocated/reserved`
  with stage tags between forwards (`step_start`, `ref_forward_winner`,
  `ref_forward_loser`, `policy_forward_winner`, `policy_forward_loser`,
  `backward`, `step_end`). One JSON line per step.
- **DPO loss components per step** — `loss_components.py`. Records the
  four MSE components, policy / reference advantage, raw DPO logit, and
  implied policy-over-reference winner probability returned by
  `flow_matching_dpo_loss(..., return_components=True)`. NaN-aware. One
  JSON line per step.
- **Routing counter live tail** — `routing_counter.py`. Mirrors the
  in-trainer `RoutingCounter` but writes a JSONL per call so monitors can
  see each routing event the moment it happens. Halts on any low-noise
  routing per AC-5.U3.

Plus `__init__.py` exposes `Watchdog`, a thin aggregator over the three
tails so the trainer only attaches one object.

## Output layout

```
<run_dir>/
  watchdog/
    rank0/
      vram.jsonl       # one record per step, with stage breakdown
      loss.jsonl       # one record per step, all DPO components
      routing.jsonl    # one record per forward, cumulative counters
      summary.json     # written at end of run by Watchdog.flush_summary()
    rank1/
      ...
    summary.json       # written by aggregate.py after run, cross-rank
```

Use `tail -f <run_dir>/watchdog/rank0/loss.jsonl` (or the others) to
monitor live. Use `python -m humanize.dpo_v0.watchdog.aggregate <run_dir>`
once at end of run to cross-aggregate per-rank summaries.

## Trainer hook

The integration into `train_dpo_i2v.py` is a five-line diff. Drop in:

```python
# top of file
from watchdog import Watchdog

# inside main() after run_dir is created and after `dtype` / `device`
# / `rank` are known:
watchdog = Watchdog(
    run_dir=run_dir,
    rank=rank,
    halt_on_low_noise=args.halt_on_low_noise,
    device=device,
)
```

Inside the per-step loop, replace the existing routing-counter call and
add the lifecycle stages around the forwards:

```python
watchdog.start_step(step, pair_id=pid)
t_raw, eps = sample_per_pair_t_eps(pid, tuple(wlat.shape), device, dtype, namespace=args.seed_namespace)
watchdog.log_routing(step=step, sampled_timestep_id=step, raw_timestep=t_raw, pair_id=pid)
# (keep the in-trainer routing_counter.log call too — that one is the
# source of truth for the run-manifest summary; the watchdog tail is the
# live-tail mirror.)

# ... build y, context, t_tensor, z_w_t / z_l_t (unchanged) ...

# Reference forwards
with torch.no_grad():
    ...
    v_ref_w = reference([z_w_t], t_tensor, context, seq_len, y=[y])[0]
    watchdog.stage("ref_forward_winner")
    v_ref_l = reference([z_l_t], t_tensor, context, seq_len, y=[y])[0]
    watchdog.stage("ref_forward_loser")
    ...

# Policy forwards
with torch.amp.autocast("cuda", dtype=dtype):
    v_pi_w = policy([z_w_t], t_tensor, context, seq_len, y=[y])[0]
    watchdog.stage("policy_forward_winner")
    v_pi_l = policy([z_l_t], t_tensor, context, seq_len, y=[y])[0]
    watchdog.stage("policy_forward_loser")

# DPO loss — switch to return_components=True so the watchdog sees the
# component breakdown.
loss, components = flow_matching_dpo_loss(
    v_policy_winner=v_pi_w.unsqueeze(0).float(),
    v_policy_loser=v_pi_l.unsqueeze(0).float(),
    v_reference_winner=v_ref_w.unsqueeze(0).float(),
    v_reference_loser=v_ref_l.unsqueeze(0).float(),
    v_target_winner=v_w.unsqueeze(0).float(),
    v_target_loser=v_l.unsqueeze(0).float(),
    beta=args.beta,
    return_components=True,
)

optimizer.zero_grad()
loss.backward()
watchdog.stage("backward")
optimizer.step()

watchdog.log_loss(step=step, pair_id=pid, t_raw=t_raw, loss=loss, beta=args.beta, components=components)
watchdog.end_step()
```

At end of run, splice the watchdog summary into the existing manifest
write (the in-trainer `routing_counter.summary()` and run-manifest
fields are unchanged):

```python
manifest["watchdog"] = watchdog.flush_summary()
```

## Why a separate live tail when the trainer already records per-step?

The trainer's existing `print(...)` step log is line-oriented stdout —
fine for one rank attached to a terminal, awkward for multi-rank,
DeepSpeed-Zero-2 4×A100 monitoring (DEC-6 default envelope under
AC-6). The watchdog gives per-rank JSONL files at known paths so:

- Monitors (`tail -f`, `wb-watch.py`, etc.) don't fight stdout multiplex.
- Loss curves are machine-parseable without scraping print lines.
- The routing counter has a per-event audit trail, not a one-shot
  end-of-run summary, so a low-noise hit shows up the instant it
  happens (in addition to the existing in-trainer hard halt).
- VRAM peak captures stage-level breakdown (forward chosen / loser /
  ref-on / ref-off / backward) rather than only end-of-step roll-up,
  so OOM root-cause analysis after an AC-6 single-rank fallback knows
  *which* forward blew the budget.

The in-trainer `RoutingCounter` and the in-trainer end-of-step VRAM
print remain the source of truth for the existing run-manifest fields.
The watchdog is additive.

## Scope and contracts

- **Read-only** w.r.t. trainer state: only `torch.cuda.*_memory_*`
  queries and Python-side tensor `.detach().cpu().item()` reductions.
- **No graph allocation**: all loss-component reads are out of the
  autograd graph (the `detach()`/`item()` chain is pure host-side).
- **Halt-on-low-noise** mirrors the in-trainer assertion. The watchdog
  raises `LowNoiseRoutingError` so a low-noise hit halts the run loudly
  even if a future trainer refactor relaxes the in-trainer assertion;
  the operator can disable this only on M5 eval-harness smoke (where
  the *frozen low-noise sampling tail* is the expected path) by
  passing `halt_on_low_noise=False`.
- **Recipe / aggregation_rule / heldout encoding**: the watchdog does
  not read or write any of these. It is observability only.
