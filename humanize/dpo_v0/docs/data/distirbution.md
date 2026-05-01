# Pair set 在 A-G 物理类上的分布

## 摘要

- round-2 raw 2745: 最欠采样 class B (训练 8.5% vs eval 21.4%)；最过采样 class A (训练 48.6% vs eval 28.6%).
- round-4 1k: 最欠采样 class B (训练 8.3% vs eval 21.4%)；最过采样 class A (训练 52.8% vs eval 28.6%).
- round-5 +1202: 最欠采样 class B (训练 9.2% vs eval 21.4%)；最过采样 class A (训练 52.1% vs eval 28.6%).
- 三个 split 合计 unclassified: 233 pair (round-2 raw 140 / round-4 1k 45 / round-5 1202 48).

## 方法

- 类定义来源: `humanize/dpo_v0/docs/eval/evalprompt.md` (A-G 物理事件分类，权威划在 42 heldout 上).
- prompt 分类方式: `pair.json` 自带 `physical_laws` 字段 (13 个 atom: collision/gravity/momentum/material/impenetrability/inertia/boundary_interaction/flow_dynamics/fluid_continuity/shadow/displacement/buoyancy/reflection) + caption 关键词组合规则。规则见文末附 `classify(text, laws)`。
- 验证: 在 42 heldout prompt 上规则与人工 ground truth 一致 (42/42).
- 无法分类的 pair 数: round-2 raw 140 / round-2 cond-present 93 / round-4 1k 45 / round-5 1202 48.

## 三个 split 的分布

| class | round-2 raw 2745 | round-2 cond-present 2202 | round-4 1k filtered | round-5 +1202 setminus | eval heldout n=42 |
|---|---|---|---|---|---|
| A 多体碰撞/反弹 | 1335 (48.6%) | 1154 (52.4%) | 528 (52.8%) | 626 (52.1%) | 12 (28.6%) |
| B 破坏/形变 | 233 (8.5%) | 194 (8.8%) | 83 (8.3%) | 111 (9.2%) | 9 (21.4%) |
| C 流体/液体 | 490 (17.9%) | 378 (17.2%) | 168 (16.8%) | 210 (17.5%) | 6 (14.3%) |
| D 阴影/反射 | 209 (7.6%) | 134 (6.1%) | 68 (6.8%) | 66 (5.5%) | 5 (11.9%) |
| E 链式/多级触发 | 51 (1.9%) | 24 (1.1%) | 13 (1.3%) | 11 (0.9%) | 3 (7.1%) |
| F 滚动/滑动 | 239 (8.7%) | 177 (8.0%) | 75 (7.5%) | 102 (8.5%) | 4 (9.5%) |
| G 抛掷/弹道 | 48 (1.7%) | 48 (2.2%) | 20 (2.0%) | 28 (2.3%) | 3 (7.1%) |
| unclassified | 140 (5.1%) | 93 (4.2%) | 45 (4.5%) | 48 (4.0%) | 0 (0.0%) |
| **总计** | 2745 | 2202 | 1000 | 1202 | 42 |

## 关键发现

训练分布 (%) vs eval 分布 (%) 差值 (正值=过采样, 负值=欠采样):

| class | round-2 raw Δ | round-4 1k Δ | round-5 1202 Δ |
|---|---|---|---|
| A 多体碰撞/反弹 | +20.1 | +24.2 | +23.5 |
| B 破坏/形变 | -12.9 | -13.1 | -12.2 |
| C 流体/液体 | +3.6 | +2.5 | +3.2 |
| D 阴影/反射 | -4.3 | -5.1 | -6.4 |
| E 链式/多级触发 | -5.3 | -5.8 | -6.2 |
| F 滚动/滑动 | -0.8 | -2.0 | -1.0 |
| G 抛掷/弹道 | -5.4 | -5.1 | -4.8 |

- **持续欠采样 (round-4 + round-5 同时 < eval -3pp):**
  - class B (B 破坏/形变): round-4 Δ=-13.1pp, round-5 Δ=-12.2pp
  - class D (D 阴影/反射): round-4 Δ=-5.1pp, round-5 Δ=-6.4pp
  - class E (E 链式/多级触发): round-4 Δ=-5.8pp, round-5 Δ=-6.2pp
  - class G (G 抛掷/弹道): round-4 Δ=-5.1pp, round-5 Δ=-4.8pp
- **持续过采样 (round-4 + round-5 同时 > eval +5pp):**
  - class A (A 多体碰撞/反弹): round-4 Δ=+24.2pp, round-5 Δ=+23.5pp
- **round-6 1000-pair 重平衡目标 (按 eval n=42 分布等比例放缩):**

| class | 当前 round-5 n | 目标 (1000-budget) | oversample 倍率 (target/round5_pct·1000) |
|---|---|---|---|
| A 多体碰撞/反弹 | 626 (52.1%) | 287 (28.6%) | 0.55× |
| B 破坏/形变 | 111 (9.2%) | 214 (21.4%) | 2.32× |
| C 流体/液体 | 210 (17.5%) | 143 (14.3%) | 0.82× |
| D 阴影/反射 | 66 (5.5%) | 119 (11.9%) | 2.17× |
| E 链式/多级触发 | 11 (0.9%) | 71 (7.1%) | 7.81× |
| F 滚动/滑动 | 102 (8.5%) | 95 (9.5%) | 1.12× |
| G 抛掷/弹道 | 28 (2.3%) | 71 (7.1%) | 3.07× |

## 数据源

- round-2 raw 2745: `/shared/user59/eval_l40s_test/T0_T3_root/T3_subset.json` (`tier_b.pair_ids`).
- round-2 cond-present 2202: 上面 setminus `/shared/user60/worldmodel/rlvideo/videodpoWan/humanize/dpo_v0/out/round4/20260428T160839Z/drop_log.json` 的 `image_path_disk_missing` (543 pair).
- round-4 1k filtered: `/shared/user60/worldmodel/rlvideo/videodpoWan/humanize/dpo_v0/out/round4/20260428T160839Z/T3_round4_tier_b_1k.json`, sha256_pin = `cf5d3e5fd528a3e0`.
- round-5 +1202 cond-present setminus: `/shared/user60/worldmodel/rlvideo/videodpoWan/humanize/dpo_v0/out/round5/20260430T171646Z/T3_round5_warm_official_1202.json`.
- prompt + physical_laws 元数据: `/shared/user59/eval_l40s_test/T0_T3_root/pair.json` (3324 pair, 250 unique prompt).
- eval heldout 类映射 (人工): `humanize/dpo_v0/docs/eval/evalprompt.md`.
- 运行时间戳 (UTC): 2026-05-01T00:16:37Z.

## 分类规则 (Python, 一次性脚本)

```python
def classify(text, laws):
    """Return A-G or "unclassified". `laws` is a set from pair.physical_laws."""
    t = text.lower(); laws = set(laws)

    # E - chain reaction
    if "domino" in t or "cascade" in t or "chain reaction" in t \
       or re.search(r"stack of (cd|cards|book)", t): return "E"

    # B - destruction (verb-driven; outranks A/F when both fire)
    destr = ["shatter","breaks into","breaks open","break open",
             "breaks under","breaks,","broke,","breaking into","breaking open",
             "splits","split open","bursts","scatter","smash","crush",
             "snap","splinter","cut down","fell a small tree","felled",
             "breaking some","fall to the ground","falls to the ground",
             "falling to the ground","leaving water residue","shattering"]
    if any(k in t for k in destr): return "B"

    # D - shadow/reflection (require optic word AND no destruction)
    if ("shadow" in laws) or ("reflection" in laws):
        optic_word = any(k in t for k in ["shadow","reflect","mirror",
                                         "reflections","glistening","glints","light shifts"])
        if optic_word and not any(k in t for k in ["shatter","break","crack","smash","crush"]):
            return "D"

    # A - explicit collision (also catches surfer/cradle/skater-collide)
    explicit = any(k in t for k in ["collid","hits the","strikes","striking ",
                                    "rebound","bounces","bouncing"," bounce",
                                    "crashes","crash"," meet,","meets,",
                                    "against a brick wall","against the wall",
                                    "into the corner","off two walls","pin","dent",
                                    "newton's cradle","reach for a volley","racquets meet"])
    if explicit and "collision" in laws: return "A"

    # C - fluids (law set OR caption)
    if (laws & {"flow_dynamics","fluid_continuity","displacement","buoyancy"}) \
       or any(k in t for k in ["water","liquid","pour","splash","fluid","toothpaste",
                               "boil","melt","submerge","wading"]): return "C"

    # G - throw/projectile (no collision target)
    if any(k in t for k in ["throws a hammer","dart lands","thrown and caught",
                            "hurl","projectile"]): return "G"

    # F - rolling / sliding / coasting
    slide = [" rolls ","rolling","rolled","glides","gliding","slides","sliding",
             "skate"," drives ","driving","swing","swings","momentum from",
             "coast","propelled","pushes off","begins to roll","grounder"]
    if any(k in t for k in slide):
        if any(k in t for k in ["hitting the","hits the","strikes the"]): return "A"
        return "F"

    # A fallback (collision law + weak verb)
    if "collision" in laws and any(k in t for k in ["hit","strike","strikes","striking",
                                                    "impact","meets","meet"]): return "A"

    # F gravity+inertia falling fallback
    if {"gravity","inertia"} <= laws and any(k in t for k in ["fall","drops","dropped",
                                                              "falls","falling","lifts","lifted",
                                                              "comes loose","tumbles"]): return "F"

    if "collision" in laws: return "A"
    return "unclassified"
```

规则在 42 heldout prompt 上对照 evalprompt.md 人工 ground truth 准确率 = 42/42.
