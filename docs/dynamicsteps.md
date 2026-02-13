# Dynamic Denoising Steps Plan

## 动机

PhyT2V 迭代 refinement 中，前几轮主要用于探索 prompt 方向（caption → mismatch → refine），视频质量不是关键，速度更重要。后面轮次 prompt 已收敛，应该用更多 steps 出高质量视频。

## 现状

- `generate_video()` 的 `num_inference_steps` 固定为 50（argparse default）
- `video_generation()` 从 CLI 读取，不区分 round
- 主循环 `for i in range(1, round_num+1)` 每轮调用同样的 `video_generation()`

## 改动方案

### 1. 添加 step schedule 函数

在 `inference.py` 中新增：

```python
def get_steps_for_round(current_round, total_rounds, min_steps=20, max_steps=50):
    """线性递增：早期 round 少 steps，最后 round 用满 steps"""
    current_round = max(1, min(current_round, total_rounds))
    min_steps = int(min_steps)
    max_steps = int(max_steps)
    if min_steps > max_steps:
        min_steps, max_steps = max_steps, min_steps
    if total_rounds <= 1:
        return max_steps
    ratio = (current_round - 1) / (total_rounds - 1)
    return int(min_steps + ratio * (max_steps - min_steps))
```

定义统一：
- `round_num` = refinement 轮数（主循环次数）
- `total_rounds = round_num + 1`（含初始视频）

例如 `round_num=4`，则 `total_rounds=5`（生成 `output1~output5`）：
- Round 1 (初始): 20 steps → 快速出图
- Round 2: 27 steps
- Round 3: 35 steps
- Round 4: 42 steps
- Round 5: 50 steps → 最终高质量

### 2. 修改 `video_generation()` 接受 steps 参数

当前签名: `video_generation(prompt_path, output_path)`

改为: `video_generation(prompt_path, output_path, num_inference_steps=50)`

将 `num_inference_steps` 直接传入 `generate_video()`，不再从 argparse 读取。

### 3. 修改主循环传入动态 steps

```python
# Round 1 初始视频 — 用最少 steps
steps = get_steps_for_round(1, args.round_num + 1)
video_generation(PROMPT_PATH, video_dir + "/output1.mp4", num_inference_steps=steps)

for i in range(1, args.round_num + 1):
    # ... caption, mismatch, score, refine prompt ...

    # Round i+1 视频 — 动态 steps
    steps = get_steps_for_round(i + 1, args.round_num + 1)
    print(f"Using {steps} denoising steps for round {i+1}")
    video_generation(PROMPT_PATH, video_output_path, num_inference_steps=steps)
```

### 4. 添加 CLI 参数控制

```
--min_steps  早期 round 的最少 steps (default: 20)
--max_steps  最终 round 的最多 steps (default: 50)
```

边界约定：
- 若 `min_steps > max_steps`，自动交换，避免配置错误导致异常。
- 若 `round_num <= 0`，等价单次生成，使用 `max_steps`。
- 通过 `int()` 向下取整，因此中间轮次会是 `27/35/42` 这类整数。

## 涉及文件

- `inference.py` — 唯一需要改的文件
  - 新增 `get_steps_for_round()` 函数
  - 修改 `video_generation()` 签名
  - 修改 `generate_video()` 调用链（直接传 steps，绕过 argparse）
  - 修改主循环和 Round 1 调用

## 预期效果

以 4 轮为例 (round_num=4，共 5 次生成):

| 视频 | Steps | 用途 |
|------|-------|------|
| output1.mp4 | 20 | 初始快速预览 |
| output2.mp4 | 27 | refinement 探索 |
| output3.mp4 | 35 | refinement 探索 |
| output4.mp4 | 42 | 接近收敛 |
| output5.mp4 | 50 | 最终高质量输出 |

总 steps: 174 vs 原来 250 (5×50)，**节省 ~30% 生成时间**，且最终质量不变。
