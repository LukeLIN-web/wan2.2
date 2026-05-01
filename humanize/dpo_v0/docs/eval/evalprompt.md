# Eval prompt classes (heldout n=42)

源: `<T0_T3_ROOT>/splits/heldout.json` — 579 pair / 245 group / **42 unique prompt**。
`prompt_id = sha256(prompt)[:12]`。

## 实验当前使用的二分类（experiment-results 里唯一被引用的）

| class | n | 代表 prompt_id | 用法 |
|---|---|---|---|
| **collision**（多体碰撞，运动量交换为核心物理约束） | 1 | `2455740c4d45` (Newton's cradle) | round-4/5 stuck-prompt watch；axes-mean ≥1.4 是 round-5 step-100 协议的恢复门槛 |
| **non-collision**（其余全部） | 41 | — | round-4 lora_final 的 +0.127 reference 就在这一桶 |

> 当前实验文档中 "per-prompt-class breakdown" 就是这一刀切，没有用更细的语义分类。"collision class n=1" 是个长期被吐槽的局限——多体碰撞的真实判据需要至少 3-5 个 prompt 才不致退化为 Newton's cradle 单点指标。

## 按物理现象的细分（建议用于 round-6+ 的 class composite）

按视频里被 PhyJudge 5 轴（SA / PTV / persistence / inertia / momentum）实际考察的核心物理事件分组。每条 prompt 只归一个主类（按事件最难/最显著的物理约束）。

### A. 多体碰撞 / 反弹（collision & rebound, n=11）
碰撞瞬间的动量传递 + 反弹角/速度

| pid | caption 摘要 |
|---|---|
| `2455740c4d45` | Newton's cradle 末端球弹出 |
| `24d86e4e0339` | bowling 球击倒球瓶 |
| `e0dae745a2a3` | kickball 撞砖墙留痕 |
| `e90b3f54bffb` | softball 撞墙反弹 |
| `2559ab47b909` | racquetball 撞两面墙连续反弹 |
| `1a0d4f1d8b1a` | 双人 volley 球拍相撞 |
| `3719b41ec796` | 球拍击中 ping-pong |
| `70d3b1b89e19` | 钢珠撞金属板留凹痕 |
| `5f68f5951b6b` | 跨栏被脚踢响 |
| `cb4c5cd47231` | 两滑冰者相撞 |
| `ad664fa349ef` | 冲浪者相撞 |
| `31cd7275ca92` | 锤子掷出弹一次 |

### B. 破坏 / 形变（fracture & destruction, n=9）
刚体或脆性材料发生不可逆形变/断裂

| pid | caption 摘要 |
|---|---|
| `1b1c06c5ff1c` | 30lb 哑铃压碎陶瓷杯 |
| `eef5be6cabd2` | 陶罐落地碎裂 |
| `d858e0d67470` | 玩具锤敲碎塑料蛋 |
| `242e01f46c08` | 南瓜压坏小凳子 |
| `61345a00dfb5` | 大锤砸石堆 |
| `8f8b14d04c41` | 水气球砸车窗破裂 |
| `8d44a2958eb4` | 链锯锯倒小树 |
| `f6cad0ea56a8` | 斧头砍倒小树 |
| `58db668bc142` | 刀切胶状甜品 |

### C. 流体 / 液体动力学（fluid dynamics, n=6）
连续流、浮力/置换、相变

| pid | caption 摘要 |
|---|---|
| `8b8d6d0a9919` | 绿色液体倾倒入碗 |
| `2be476eeac0d` | 牙膏挤出连续流 |
| `e38a4396df92` | 皮靴浸入水中置换液体 |
| `fa37196314bf` | 桶装料倒入金属罐 |
| `252b84def499` | 蛋黄入沸水凝固（相变） |
| `1a44aba35343` | 提桶过及胸深水 |

### D. 阴影 / 反射 / 光学一致性（lighting & reflection, n=5）
光线/阴影/镜面随物体运动应一致变化

| pid | caption 摘要 |
|---|---|
| `75f6acbf5ba7` | 沙漠车影随车移动 |
| `261fccfc811f` | 人在沙漠投长影行走 |
| `7977e8df650c` | 摩托车阴影随抬起变化 |
| `488e8d91cff5` | 白船在峡谷水面镜面反射 |
| `6b48a3f28874` | 玻璃后机器人受灯光反光变化 |

### E. 链式 / 多级触发（chain reaction & cascading, n=3）
A 触发 B 触发 C 的多级因果

| pid | caption 摘要 |
|---|---|
| `e7815fab19d6` | 旋转杆 → 鸭子 → 多米诺 |
| `36e42af19937` | 两排多米诺 + 旋转杆隔空触发 |
| `9d500eec2188` | 手指戳 CD 堆隔出缝隙 |

### F. 滚动 / 滑动 / 持续动量（rolling & gliding, n=4）
摩擦/势能转动能/匀速滑行

| pid | caption 摘要 |
|---|---|
| `5fdbe9f87762` | 滚地球在内场滑过 |
| `48255a441729` | 滑冰者推墙获得动量 |
| `80cc85fa7fa7` | 高空秋千传球 |
| `31ea17615154` | 棍掉，引擎盖坠落 |

### G. 抛掷 / 弹道（throwing & ballistic, n=3）
仅抛物线轨迹 + 落点

| pid | caption 摘要 |
|---|---|
| `2db7ce10fffb` | 飞镖落在分隔线上 |
| `daed47f0fab3` | 假肢运动员抛锤 |
| `b69b73d7b65f` | 比萨饼皮多次抛接边缘变圆 |

### 类间汇总

| class | n | 占比 |
|---|---|---|
| A 多体碰撞/反弹 | 12 | 28.6% |
| B 破坏/形变 | 9 | 21.4% |
| C 流体/液体 | 6 | 14.3% |
| D 阴影/反射 | 5 | 11.9% |
| E 链式触发 | 3 | 7.1% |
| F 滚动/滑动 | 4 | 9.5% |
| G 抛掷/弹道 | 3 | 7.1% |
| 合计 | 42 | 100% |

> 注：A 类有 12 条而非上文表格的 11 条，含 `31cd7275ca92`（锤子弹一次），它兼具弹道+碰撞，归 A。

## 与 PhyJudge 5 轴的对应直觉

| 轴 | 哪些 class 最敏感 |
|---|---|
| SA (semantic alignment) | 全部 |
| PTV (physical temporal violation) | A, E, F |
| persistence (object identity over time) | B, C, D |
| inertia (Newton 1st law) | A, F, G |
| momentum (conservation) | A, E, G |

⇒ **A 类是 inertia / momentum 的核心信号源**；round-4/5 一直盯 Newton's cradle 是因为它是 A 类里唯一一个有"精确动量守恒"的强约束 prompt（其余 A 类大多只考验"撞了之后会反弹/损坏"的弱约束）。

## 用法建议

1. **rolling n=24 / n=42 报告**: 至少分别给出 A vs (B∪C∪D∪E∪F∪G) 两桶的 axes-avg，比当前 collision vs non-collision (1 vs 41) 信噪比高得多。
2. **stuck-prompt watch** 不要只盯 Newton's cradle；A 类里 `2559ab47b909`（连续两次反弹）和 `e7815fab19d6`（链式）也是 base 模型 1-2 分常驻户。
3. **round-6 round 设计**: 训练 pair 的 class 分布应至少覆盖到 A/B/C 三桶，避免再次出现 round-5 "碰撞类回来了但 non-collision 蒸发" 的失衡。

## 数据来源校验

```
heldout.json: /shared/user59/eval_l40s_test/T0_T3_root/splits/heldout.json
records=579, unique_prompts=42, unique_groups=245
prompt_id 生成: hashlib.sha256(prompt.encode()).hexdigest()[:12]
```
