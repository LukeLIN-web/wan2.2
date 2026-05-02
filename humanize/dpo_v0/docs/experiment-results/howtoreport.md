# 如何写 round-N eval 报告（强制改造）

写这份的原因：`round5_v3_lr1e5_warm_step100.md` 那种写法**已经被叫停**——n=24 上读 ±0.04 的 Δ、把 collision class n=1 当一个独立信号、决策门槛写成 "≥ +0.114" 这种点估计比较，全部不要再写。

下面是新模板。**所有 round-6+ 的 eval md 必须按这个结构写，不允许加章节，不允许加叙事段落。**

---

## 一、强制规则（违反任何一条 = 退回重写）

1. **永远不在 n<42 的样本上下结论**。n<42 只允许写一句话："rolling read, direction-of-travel only, no decisions taken." 之后什么都不写。决策必须等 lora_final n=42。
2. **永远不直接比较不同 n 的 axes-avg**。round-4 lora_final n=42 vs round-5 step-100 n=24 的对照是无效推断。要比就用 round-5 的相同 24 子集回算 round-4。
3. **collision class n=1 这个分桶整个废弃**。所有 class 分析必须用 `docs/eval/evalprompt.md` 的 7 类 (A-G) 或至少 "A 类 (n=12) vs 其他 (n=30)" 的二分。Newton's cradle 单 prompt 只能进 per-prompt 表，不能进 per-class 表。
4. **所有 Δ 数字必须配 95% CI 或 sign-test p**。整数 1-4 PhyJudge × n=42、σ≈0.55 → 单轴 95% CI 半宽 ≈ 0.17。Δ 绝对值小于 0.17 的一律标 "within noise"，不要再写"weakening / flipped negative / drifting" 这种叙事词。
5. **决策门槛事前定，不事后改**。round 开始前在 plan 文档里写死 "winner 判据 = X"，eval 完只填 pass/fail，不允许临时新增标准（如 round-5 那个临时冒出来的 "PTV sign-test 作为统计确认"）。
6. **TL;DR 只能有 1 个 verdict + 1 个判据 + 1 个 next action**。不允许放 3 行 metric table 当 TL;DR。
7. **Hypothesis update 段落删除**。假设是 plan 阶段的事，eval 阶段只输出 "criterion X: pass/fail"。"hypothesis was half-right" 这种话写出来就是没读过本规范。
8. **同一个数字在文档里只准出现一次**。round-5 那篇 +0.127 出现 5 次，+0.114 出现 4 次，是噪声。引用就用锚点 `[ref:r4_lora_final_n42]`。
9. **Caveats 段不允许超过 3 条**。caveat 多到要列 4+ 条说明实验设计本身有问题，去修实验，不要写 caveat。
10. **没有 baseline-from byte-equal 校验通过的 run，整篇报告作废**。这条不写在文档正文，写在自动化检查里；md 里只放一行 `baseline_sha256_match: true`。

---

## 二、强制模板

```markdown
# Round-N <variant> <ckpt> PhyJudge eval (n=<N>)

**Verdict**: <pass | fail | rolling-read-only>
**Criterion** (set in plan `<plan-doc-id>`): <one line>
**Next action**: <one line>

## Run identity

| field | value |
|---|---|
| ckpt | <name> (<size>) |
| ckpt sha256 | <full> |
| trainer step health | loss=<>, gnorm=<>, margin=<>, vram_peak=<> |
| eval gen out | <path> |
| baseline_sha256_match | true |
| n | <N> |
| baseline ref | <ref-id> |

## Headline (n=<N>)

| metric | value | 95% CI | vs criterion |
|---|---|---|---|
| axes-avg Δ | <±X.XXX> | [<lo>, <hi>] | pass / fail / within-noise |
| A 类 axes-avg Δ (n=12) | <±X.XXX> | [<lo>, <hi>] | pass / fail / within-noise |
| 其他类 axes-avg Δ (n=30) | <±X.XXX> | [<lo>, <hi>] | pass / fail / within-noise |

## Per-axis Δ (n=<N>)

| axis | Δ | 95% CI | sign-test p (vs 0) |
|---|---|---|---|
| SA | | | |
| PTV | | | |
| persistence | | | |
| inertia | | | |
| momentum | | | |

## Per-class Δ axes-avg (用 evalprompt.md 的 A-G)

| class | n | Δ | 95% CI |
|---|---|---|---|
| A 多体碰撞 | 12 | | |
| B 破坏/形变 | 9 | | |
| C 流体 | 6 | | |
| D 阴影/反射 | 5 | | |
| E 链式 | 3 | | |
| F 滚动/滑动 | 4 | | |
| G 抛掷/弹道 | 3 | | |

> n<5 的 class 只显示 raw Δ，不计算 CI（无意义）。

## Stuck-prompt watch (per-prompt scores, base=2)

只列 base 模型 ≤2 的 prompt + ckpt 对应分数。不写散文。

| pid | class | base axes-mean | ckpt axes-mean | 5 轴细分 |
|---|---|---|---|---|

## Caveats (≤3 条)

- ...

## Source data

- aggregate JSON: <path>
- per-prompt scores: <path>
- gen videos: <path>
- baseline reuse: <path>
```

**整篇报告硬上限：上面这套表 + 不超过 3 条 caveat + 来源路径。**

---

## 三、Verdict 的 3 个合法值（也是仅有的 3 个）

| verdict | 触发条件 |
|---|---|
| `pass` | n=42 lora_final，且事前判据全部满足 |
| `fail` | n=42 lora_final，至少一个事前判据未满足 |
| `rolling-read-only` | n<42 或非 lora_final ckpt。整篇只输出表格，不允许出现"trend / encouraging / concerning / weakening"等叙事形容词 |

`rolling-read-only` 状态下，**Verdict 行后面接一句 "no decisions taken at this checkpoint" 就结束**，不要再写 "hypothesis update" "trajectory hint"。这些都是为了制造确定感的伪信息。

---

## 四、统计方法（事前固定）

- **CI**: bootstrap percentile，n_resamples=10000，按 prompt 重采样。
- **sign-test**: per-prompt Δ 的符号 vs 0，二项检验双侧。
- **比较 round 间 winner**: 按相同 prompt id 集合配对 sign-test，**不允许**直接比 axes-avg 点估计。
- **多重比较校正**: 5 轴 × N class 时使用 Bonferroni（α=0.05/(5×N)）。

实现一次写在 `eval/stats.py`，所有报告调用同一份函数生成数字，不允许在 md 里手算。

---

## 五、明确禁止的写法（来自 round-5 那篇的反面教材）

| 禁止 | 替换为 |
|---|---|
| "essentially flat" / "essentially zero" | Δ=X.XXX, [lo, hi], within-noise |
| "PTV flipped negative" | PTV Δ=−0.042, sign-test p=0.78, within-noise |
| "warm-start retraced round-4 trajectory" (在 n=1 上) | 不写。n=1 不构成证据 |
| "this is NOT a warmup transient" | 不写。warmup transient 是 hypothesis，事后宣布它真假是循环论证 |
| "encouraging but only represents 1 prompt" | 不写。任何 "encouraging but" 都是偷渡 narrative |
| "the key signal" | 不写。signal 看 CI，不看你的形容词 |
| "redistributing axis-level performance, not uniformly improving" | 不写。这句话在 5 轴 × σ≈0.55 × n=24 上无统计意义 |
| 决策门槛 "axes-avg ≥ +0.114" | "round-5 与 round-4 lora_final 配对 sign-test on n=42, p<0.05 favoring round-5" |

---

## 六、为什么这么严

PhyJudge 1-4 整数尺度 + 5 轴 + n=42 是**低分辨率测量工具**。它能稳定区分的最小效应大约是 axes-avg ±0.10 量级（n=42 时 95% CI 半宽 ≈ 0.05-0.08）。在 n=24 上谈 ±0.04 的 Δ、在 n=1 上谈 class 行为，是把仪器精度之外的随机扰动当成科学结论。

round-4 之所以拿到 +0.127 这个真信号，是因为它**是 n=42 上跨 5 轴累计的稳定偏移**，不是因为某个 step 的某个 axis 翻正。round-5 的报告把这种点估计当成精确门槛去比较，方法论上是退步。

报告的目的是让读的人 30 秒内知道 "这个 ckpt 该 ship 还是 revert"，不是让你展示自己读出了多少 nuance。verdict + 判据 + next action，三行写完。

---

## 七、迁移 checklist（给改造旧报告的人）

- [ ] TL;DR 改成 verdict + criterion + next action 三行
- [ ] 删除所有 "hypothesis update" 段
- [ ] 删除所有 collision (n=1) vs non-collision (n=41) 表，换成 A-G 类 或 A vs 其他
- [ ] 所有 Δ 加 95% CI
- [ ] |Δ| < CI 半宽的轴/类标 within-noise，删除对应散文
- [ ] caveats 砍到 ≤3
- [ ] 重复出现的数字改成锚点引用
- [ ] 移除所有 "encouraging / concerning / weakening / drifting / flipped" 等叙事形容词
- [ ] n<42 的报告整篇加 `rolling-read-only` 标识，移除所有决策性语言
