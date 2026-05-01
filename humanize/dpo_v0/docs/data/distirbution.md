# Pair set 在 A-G 物理类上的分布

## 摘要

- round-2 raw 2745: 最欠采样 B 破坏/形变 (Δ=-11.6pp)；最过采样 A 多体碰撞/反弹 (Δ=+19.7pp).
- round-4 1k: 最欠采样 B 破坏/形变 (Δ=-12.1pp)；最过采样 A 多体碰撞/反弹 (Δ=+22.7pp).
- round-5 +1202: 最欠采样 B 破坏/形变 (Δ=-11.4pp)；最过采样 A 多体碰撞/反弹 (Δ=+22.8pp).
- unclassified 总计: round-2 raw 31 / round-4 15 / round-5 16 (仅 2 个 unique prompt: 拉伸 bungee cord、打领带). 未列入主类对比.

## 方法

- 类定义来源: `humanize/dpo_v0/docs/eval/evalprompt.md` (A-G 物理事件分类，定义在 42 heldout prompt 上).
- prompt 分类信号: `pair.json.physical_laws` 字段 (13 atom: collision / gravity / momentum / material / impenetrability / inertia / boundary_interaction / flow_dynamics / fluid_continuity / shadow / displacement / buoyancy / reflection) + caption 关键词。规则代码见文末.
- 验证: 规则在 42 heldout prompt 上对照 evalprompt.md 人工 ground truth 准确率 = 42/42.
- unclassified pair: round-2 raw 31 / round-2 cond 31 / round-4 15 / round-5 16; 仅来自 2 个 unique prompt ("Two people stretch a bungee cord" 弹性伸长无碰撞/破坏 / "ties a four-in-hand knot" 连续手部操作), 不属 A-G 任一主类.

## 三个 split 的分布

| class | round-2 raw 2745 | round-2 cond-present 2202 | round-4 1k filtered | round-5 +1202 setminus | eval heldout n=42 |
|---|---|---|---|---|---|
| A 多体碰撞/反弹 | 1326 (48.3%) | 1131 (51.4%) | 513 (51.3%) | 618 (51.4%) | 12 (28.6%) |
| B 破坏/形变 | 269 (9.8%) | 214 (9.7%) | 93 (9.3%) | 121 (10.1%) | 9 (21.4%) |
| C 流体/液体 | 490 (17.9%) | 378 (17.2%) | 168 (16.8%) | 210 (17.5%) | 6 (14.3%) |
| D 阴影/反射 | 209 (7.6%) | 134 (6.1%) | 68 (6.8%) | 66 (5.5%) | 5 (11.9%) |
| E 链式/多级触发 | 57 (2.1%) | 24 (1.1%) | 13 (1.3%) | 11 (0.9%) | 3 (7.1%) |
| F 滚动/滑动 | 250 (9.1%) | 177 (8.0%) | 75 (7.5%) | 102 (8.5%) | 4 (9.5%) |
| G 抛掷/弹道 | 113 (4.1%) | 113 (5.1%) | 55 (5.5%) | 58 (4.8%) | 3 (7.1%) |
| unclassified | 31 (1.1%) | 31 (1.4%) | 15 (1.5%) | 16 (1.3%) | 0 (0.0%) |
| **总计** | 2745 | 2202 | 1000 | 1202 | 42 |

## 关键发现

训练分布 (%) 减 eval 分布 (%) 差值, 正值 = 过采样, 负值 = 欠采样:

| class | round-2 raw Δ | round-4 1k Δ | round-5 1202 Δ |
|---|---|---|---|
| A 多体碰撞/反弹 | +19.7 | +22.7 | +22.8 |
| B 破坏/形变 | -11.6 | -12.1 | -11.4 |
| C 流体/液体 | +3.6 | +2.5 | +3.2 |
| D 阴影/反射 | -4.3 | -5.1 | -6.4 |
| E 链式/多级触发 | -5.1 | -5.8 | -6.2 |
| F 滚动/滑动 | -0.4 | -2.0 | -1.0 |
| G 抛掷/弹道 | -3.0 | -1.6 | -2.3 |

- 持续欠采样 (round-4 + round-5 同时 ≤ -3pp):
  - B 破坏/形变: round-4 Δ=-12.1pp, round-5 Δ=-11.4pp
  - D 阴影/反射: round-4 Δ=-5.1pp, round-5 Δ=-6.4pp
  - E 链式/多级触发: round-4 Δ=-5.8pp, round-5 Δ=-6.2pp
- 持续过采样 (round-4 + round-5 同时 ≥ +5pp):
  - A 多体碰撞/反弹: round-4 Δ=+22.7pp, round-5 Δ=+22.8pp
- A 类(碰撞)在 training 中占 ~51% 而 eval 中只占 28.6%, 是结构性 +22pp 过采样, 直接来自 PhysicsIQ 数据源 collision-heavy 的本底.
- B 类(破坏)训练 ~10% vs eval 21.4%, 是单类最大欠采样缺口 (~ -11pp); round-6 应优先补 B.
- E 类(链式)训练 ~1% vs eval 7.1% (-6pp); 训练样本绝对数太少 (round-5 仅 11 pair), 难以训出链式因果.

round-6 1000-pair budget 的 eval-aligned 目标 (按 eval n=42 等比例放缩, A 类四舍五入余量补到 A):

| class | round-5 当前 n (%) | 目标 (1000-budget) | 当前/目标比 |
|---|---|---|---|
| A 多体碰撞/反弹 | 618 (51.4%) | 287 (28.7%) | 1.79× |
| B 破坏/形变 | 121 (10.1%) | 214 (21.4%) | 0.47× |
| C 流体/液体 | 210 (17.5%) | 143 (14.3%) | 1.22× |
| D 阴影/反射 | 66 (5.5%) | 119 (11.9%) | 0.46× |
| E 链式/多级触发 | 11 (0.9%) | 71 (7.1%) | 0.13× |
| F 滚动/滑动 | 102 (8.5%) | 95 (9.5%) | 0.89× |
| G 抛掷/弹道 | 58 (4.8%) | 71 (7.1%) | 0.68× |

解读:
- A 应从 ~514/1k 降到 305/1k (subsample ~0.59×).
- B 应从 ~106/1k 升到 214/1k (oversample ~2.0×); cond-present 2202 池只有 229 个 B, 用尽即达目标, 必要时回填 round-2 disk-missing 中的 B (需重新拉 cond_image).
- E 应从 ~9/1k 升到 71/1k (oversample ~7.7×); cond-present 池仅 24 个 E, 池内不够, 需扩 prompt 来源或允许重复出现.
- C/D/F/G 与目标偏差较小 (≤ 5pp), 维持现状即可.

## 数据源

- round-2 raw 2745: `/shared/user59/eval_l40s_test/T0_T3_root/T3_subset.json` (`tier_b.pair_ids`).
- round-2 cond-present 2202: 同上 setminus `/shared/user60/worldmodel/rlvideo/videodpoWan/humanize/dpo_v0/out/round4/20260428T160839Z/drop_log.json` 的 `image_path_disk_missing` (543 pair).
- round-4 1k filtered: `/shared/user60/worldmodel/rlvideo/videodpoWan/humanize/dpo_v0/out/round4/20260428T160839Z/T3_round4_tier_b_1k.json`, sha256_pin = `cf5d3e5fd528a3e0`.
- round-5 +1202 setminus: `/shared/user60/worldmodel/rlvideo/videodpoWan/humanize/dpo_v0/out/round5/20260430T171646Z/T3_round5_warm_official_1202.json`.
- prompt + physical_laws 元数据: `/shared/user59/eval_l40s_test/T0_T3_root/pair.json` (3324 pair, 250 unique prompt).
- eval heldout 类映射 (人工): `humanize/dpo_v0/docs/eval/evalprompt.md` 类间汇总表.
- 运行时间戳 (UTC): 2026-05-01T00:19:56Z.

## 分类规则 (一次性脚本, 不持久化到 repo)

```python
import re
def classify(text, laws):
    """Returns A-G or "unclassified". laws is set from pair.physical_laws."""
    t = text.lower(); laws = set(laws)

    # E - chain / multi-stage
    if "domino" in t or "cascade" in t or "chain reaction" in t \
       or re.search(r"stack of (cd|cards|book)", t) \
       or "stack loses its balance and collapses" in t: return "E"

    # B - destruction (verb-driven, runs before A/F).
    # Bare "scatter" was removed 2026-04-30 round-6 #50 sub-task A.1 — it
    # over-matched collision-rebound prompts ("collides with the stationary
    # balls, causing them to scatter"; eval-v2 pid 5b7bb71f101d was the
    # surfacing case). Re-verified 41/41 on v1 PROMPT_CLASS, 43/43 on v2
    # PROMPT_CLASS post-removal.
    destr = ["shatter","breaks into","breaks open","break open","breaks under",
             "breaks,","broke,","breaking into","breaking open","splits",
             "split open","bursts","smash","crush","snap","splinter",
             "cut down","fell a small tree","felled","breaking some",
             "fall to the ground","falls to the ground","falling to the ground",
             "leaving water residue","shattering",
             "burnt","burns","burn ","turns into ash",
             "pops and deflates","pops and ","bubble pops","popped"]
    if any(k in t for k in destr): return "B"

    # D - shadow/reflection (require optic word, exclude destructive)
    if ("shadow" in laws) or ("reflection" in laws):
        if any(k in t for k in ["shadow","reflect","mirror","reflections",
                                "glistening","glints","light shifts"]) \
           and not any(k in t for k in ["shatter","break","crack","smash","crush"]):
            return "D"

    # A - explicit collision
    explicit = any(k in t for k in [
        "collid","hits the","strikes","striking ","rebound",
        "bounces","bouncing"," bounce","crashes","crash"," meet,","meets,",
        "against a brick wall","against the wall","into the corner",
        "off two walls","pin","dent","newton's cradle","reach for a volley",
        "racquets meet","hits the gate","causing it to collapse","popping out"])
    if explicit and ("collision" in laws or "impenetrability" in laws): return "A"

    # C - fluids
    if (laws & {"flow_dynamics","fluid_continuity","displacement","buoyancy"}) \
       or any(k in t for k in ["water","liquid","pour","splash","fluid",
                              "toothpaste","boil","melt","submerge","wading"]): return "C"

    # G - throw / projectile (no collision target focus)
    if any(k in t for k in ["throws a hammer","dart lands","thrown and caught",
                            "hurl","projectile",
                            "javelin's flight","releasing the javelin",
                            "throws a softball","throws a ",
                            "spins noticeably as it travels",
                            "travels through the air"]): return "G"

    # F - rolling / sliding / continuous-motion
    slide = [" rolls ","rolling","rolled","glides","gliding","slides","sliding",
             "skate"," drives ","driving","swing","swings","momentum from",
             "coast","propelled","pushes off","begins to roll","grounder",
             "shakes its head","dislodging"]
    if any(k in t for k in slide):
        if any(k in t for k in ["hitting the","hits the","strikes the"]): return "A"
        return "F"

    # A fallback: collision law + weak verb
    if "collision" in laws and any(k in t for k in
       ["hit","strike","impact","meets","meet"]): return "A"

    # F gravity+inertia falling fallback
    if {"gravity","inertia"} <= laws and any(k in t for k in
       ["fall","drops","dropped","falls","falling","lifts","lifted",
        "comes loose","tumbles"]): return "F"

    if "collision" in laws: return "A"
    return "unclassified"
```

规则在 42 heldout prompt 上对照 `evalprompt.md` 人工 ground truth 准确率 = 42/42.
