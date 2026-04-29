# WandB 监控

## Token

`./wandbtoken`（仓库根，已 gitignore），单行，格式 `wandb_v1_...`。

本机 `~/.netrc` 默认是 `armandm`，**读不到** `lukelin/wanrl/*`。每次调用前显式覆盖：

```bash
export WANDB_API_KEY=$(cat /shared/user60/worldmodel/rlvideo/videodpoWan/wandbtoken)
```

## 一行查 run 状态

```bash
export WANDB_API_KEY=$(cat /shared/user60/worldmodel/rlvideo/videodpoWan/wandbtoken)
python3 - <<'PY'
import os, wandb
run = wandb.Api(api_key=os.environ['WANDB_API_KEY']).run('lukelin/wanrl/<RUN_ID>')
print(run.state, run.heartbeatAt, '_step=', run.summary.get('_step'))
hist = run.history(samples=2000)
print(hist.tail(8)[['_step','loss','margin','accuracy','acc_win50','grad_norm','grad_finite']].to_string())
PY
```

`<RUN_ID>` 从 URL 末段拿：`wandb.ai/lukelin/wanrl/runs/<RUN_ID>`。

## 健康信号优先级

DPO i2v（β=1000，r5）盯这几列，**按顺序**：

1. `grad_finite` — 必须恒 1。出现 0 = raw grad inf/NaN，那一步被跳过；连续多次出现 → kill。
2. `margin` / `accuracy` / `acc_win50` — 真正的学习信号。margin 应正向爬升，acc_win50 从 0.5 往 1 走。
3. `loss` — 起点 ln 2 ≈ 0.6931。健康轨迹：margin 一旦正向，loss 会**断崖**塌到 ~0（β=1000 sigmoid 在 |Δ|≈0.005 就饱和）。
4. `grad_norm` — raw 值，看走势不看绝对值。β=1000 下 step-0 ~200 是预期，值由 σ(-β·Δ) 决定：margin 正向走 → grad_norm 掉到个位数；margin 走负 → 爬到 ~400 并稳住，**这是红灯**。

## 自动周期 check

会话内用 `CronCreate` 每 10 min 触发 `/training-check`（详见 `~/.claude/skills/training-check`）：

```
cron: 7,17,27,37,47,57 * * * *
prompt: /training-check lukelin/wanrl/<RUN_ID> ...
```

健康信号稳定后逐级放宽到 20 → 30 → 60 min。任何一次异常 → 重置回 10 min。

## 常见坑

- `Could not find run` / `Could not find project wanrl` — 没设 `WANDB_API_KEY`，落到了 armandm 默认账号。
- summary 全空 / Step=None — 还在 cold boot（FSDP shard + T5/VAE encode），14B 通常 5+ min 才出 step 0，正常等。
- `loss=0.6931 margin=0` 一直不动 — 不是没学，看 step 数是不是 ≤2。step 0 必然是 ln 2，step 1-2 还在小幅扰动是正常的。
