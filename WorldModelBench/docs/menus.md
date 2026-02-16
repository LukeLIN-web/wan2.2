# WorldModelBench 目录说明

| 目录 | 用途 |
|---|---|
| `images/` | 350 张 first-frame JPG，`worldmodelbench.json` 中 `first_frame` 字段引用 |
| `outputs/` | Wan2.2-TI2V-5B baseline 生成的 350 个视频（一次性 I2V，无 refinement） |
| `iterative_outputs/` | Phyrefine 迭代 refinement 输出（含 `round{1,2,3}_videos/`、`checkpoints/`、`eval/`） |
| `VILA/` | VILA 源码（VLM judge 依赖） |
| `vila-ewm-qwen2-1.5b/` | VILA judge 模型权重 |
| `docs/` | 项目文档（勿删除） |
| `eval_v2_<timestamp>/` | v2 评估运行产出（8-GPU 分片结果 + 日志 + 合并后 `results.json`） |
