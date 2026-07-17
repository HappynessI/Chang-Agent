# Change-Agent 开发工作报告

更新时间：2026-07-17
代码仓库：`git@github.com:HappynessI/Chang-Agent.git`
当前分支：`main`
当前提交：`d2002a1 fix: enable single-GPU smoke execution`

本文记录本轮开发实际完成的代码、模型、测试、资源测量和 Git 管理工作。模型权重和数据集没有提交到 Git，只保留在本地路径。

## 1. 最终代码结构

### 1.1 Agent 运行闭环

- `change_agent/state.py`
  - 定义 `AgentAction`、`AgentObservation`、`ChangeState`、`VerifierOutput` 等状态对象。
  - Agent 观察只包含 T1、T2、当前 change mask、query、Verifier 反馈和历史摘要。
  - 隐藏的 T1/T2 mask、实例、匹配关系和模型证据保留在 Environment 内部。
- `change_agent/action_parser.py`
  - 从普通文本或 Markdown JSON fence 中提取一个 JSON action。
  - 校验 `target_view`、动作类型、点坐标、box 坐标和图像边界。
  - 将 `[0,1000]` 归一化坐标转换成像素坐标。
- `change_agent/environment.py`
  - 实现 GT-free 的 `reset` / `step` 运行时环境。
  - `reset` 不接受 GT；运行时只依赖 T1、T2 和 query。
  - 支持 point、box、finish 三类动作，维护最大步数和终止条件。
- `change_agent/executor.py`
  - 将 point 动作交给 SimpleClick 边界，将 box 动作交给 SAM3/OmniOVCD 边界。
- `change_agent/trajectory.py`
  - 保存每一步 raw action、解析后的 action、Verifier 输出、state 快照和执行证据。
  - 支持 JSON trajectory 与每步 mask 文件落盘。
- `change_agent/runner.py`、`change_agent/agent.py`
  - 提供脚本 Agent 和统一 rollout runner。
  - 支持历史最佳 state 选择，而不是只返回最后一步。

### 1.2 Mask、实例和 change state

- `change_agent/adapters/omniovcd_adapter.py`
  - 通过回调隔离 OmniOVCD/SAM3 的重模型依赖。
  - 实现 T1/T2 mask 初始化、box segmentation、8 邻域 connected components、实例匹配和 change mask 重建。
  - matching 使用确定性的一对一 greedy overlap 匹配。
- `change_agent/adapters/sam3_adapter.py`
  - 接入 OmniOVCD 的公开 `set_image`、`set_text_prompt`、`add_geometric_prompt` API。
  - 将 SAM3 logits/masks 转成 NumPy bool mask 和 confidence map。
  - 新增 `segment_text`，支持低内存场景按视图分阶段调用。
- `change_agent/adapters/segagent_adapter.py`
  - 包装 SegAgent 的 SimpleClick 推理模型。
  - 将 Environment 当前目标视图 mask 作为 SimpleClick `prev_mask` 传入。
- `change_agent/perturbations.py`
  - 仅供离线 Verifier 监督使用，生成 erode、dilate、local add/delete 等候选 mask 和误差标签。

### 1.3 Qwen3-VL Adapter

文件：`change_agent/adapters/qwen3vl_adapter.py`

完成内容：

1. 使用 `AutoProcessor` 和 `Qwen3VLForConditionalGeneration` 加载本地 Qwen3-VL-2B-Instruct。
2. 通过现代 `apply_chat_template` API 构造多模态消息。
3. 明确标记三类图像：T1 earlier image、T2 later image、current change mask。
4. 强制模型只返回一个结构化 action JSON。
5. 将生成文本交给统一 `ActionParser`，避免 Adapter 自己重复坐标和动作校验。

模型本地路径：

```text
/Data/wyh/CD-SegAgent/models/Qwen3-VL-2B-Instruct
```

模型文件不在 Git 中。

## 2. Verifier 实现

### 2.1 规则 Verifier

文件：`change_agent/verifier.py`

实现了可直接运行的 rule-based baseline：

- 根据 change mask 比例、模型 evidence 和动作类型计算质量分数。
- 对 finish 动作进行接受/拒绝判断。
- 输出 quality score、error type、target view、recommended action 和 error map。

该 Verifier 是运行闭环基线，不代表最终研究模型。

### 2.2 可训练 Verifier head

文件：`change_agent/verifier_model.py`

实现了冻结视觉特征上的轻量训练头：

- 两层卷积 encoder。
- quality regression head。
- error-map segmentation head。
- error-type classification head。
- target-view classification head。
- action classification head。
- 组合 MSE、BCE、Dice 和多项 cross entropy loss。

训练入口：`tools/train_verifier.py`

支持两种输入：

- `--smoke`：确定性合成数据，用于代码和 optimizer smoke。
- `--samples path.npz`：真实离线样本。

### 2.3 真实 Verifier 样本构建

新增：`tools/build_verifier_samples.py`

该工具读取离线数据集的：

- `A/<name>.png`
- `B/<name>.png`
- `label_cvt/<name>.png`

利用 GT 只在离线样本构建阶段生成候选 mask 和监督标签，输出特征不包含原始 GT mask。当前特征由以下内容组成：

- T1 RGB：3 通道
- T2 RGB：3 通道
- T1/T2 absolute difference：3 通道
- candidate change mask：1 通道

输出字段：

```text
features
quality
error_map
error_type
target_view
action
```

已用 LEVIR-CD `test_256` 的 2 个样本完成构建和训练 smoke。

## 3. 真实模型和适配器验证

由于 SAM3、SimpleClick 和 Qwen 使用不同 Python 依赖栈，验证采用独立进程；没有强行把三个大模型塞进同一个 Python 环境。

### 3.1 Qwen3-VL CPU 验证

入口：`tools/qwen3vl_smoke.py`

结果：

```text
加载时间：约 5.1 秒
CPU RSS 峰值：约 982 MB
生成 JSON：成功
ActionParser：成功
```

### 3.2 Qwen3-VL GPU 验证

使用单张 L20：`CUDA_VISIBLE_DEVICES=7`

结果：

```text
PyTorch：2.9.1+cu128
可见 CUDA device：1
模型加载：6.893 秒
CUDA allocated：约 4,067 MB
CUDA reserved：约 4,174 MB
CUDA peak allocated：约 4,147 MB
```

### 3.3 Qwen3-VL → Environment GPU rollout

入口：`tools/rollout_smoke.py --agent qwen3vl`

结果：

```text
Environment rollout：成功
步数：1
Verifier/finish：成功
总耗时：14.188 秒
CUDA allocated：约 4,067 MB
CUDA peak allocated：约 4,147 MB
trajectory：outputs/qwen_gpu_rollout/trajectory.json
```

### 3.4 SAM3 GPU 验证

入口：`tools/sam3_smoke.py --device cuda --resolution 1008`

本地 checkpoint：

```text
/Data/wyh/CD-SegAgent/models/sam3/sam3.pt
```

验证了：

- text prompt
- box prompt
- processor state 转 NumPy mask
- SAM3 mask evidence 提取

资源结果：

```text
耗时：13.540 秒
CUDA allocated：约 3,525 MB
CUDA reserved：约 4,166 MB
CUDA peak allocated：约 3,909 MB
CPU RSS 峰值：约 7,488 MB
```

### 3.5 SimpleClick GPU 验证

入口：`tools/simpleclick_smoke.py --device cuda`

本地 checkpoint：

```text
/Data/wyh/CD-SegAgent/models/SimpleClick/cocolvis_vit_large.pth
```

验证了：

- checkpoint 加载
- external `prev_mask` 传递
- positive click
- 256x256 mask 结果回传

资源结果：

```text
耗时：7.167 秒
CUDA allocated：约 1,249 MB
CUDA reserved：约 1,450 MB
CUDA peak allocated：约 1,390 MB
CPU RSS 峰值：约 3,230 MB
```

## 4. SAM3 CPU 兼容性修复

为了让 OmniOVCD/SAM3 在 CPU 和 GPU 两种上下文都能工作，在独立的 `OmniOVCD` 仓库中提交了以下源码修复：

### 4.1 Decoder coordinate cache

文件：`OmniOVCD/sam3/model/decoder.py`

- cache 创建不再硬编码 CUDA。
- cache 记录 feature size 和 device。
- CPU/CUDA 切换后自动重建或迁移坐标 cache。
- 修复 cached coordinates 与 reference boxes device 不一致的问题。

### 4.2 Positional encoding cache

文件：`OmniOVCD/sam3/model/position_encoding.py`

- 原始实现预计算时固定 `device="cuda"`。
- 现在根据 `torch.cuda.is_available()` 选择 CPU 或 CUDA。

### 4.3 Geometry encoder

文件：`OmniOVCD/sam3/model/geometry_encoders.py`

- CUDA 使用 pinned memory 和 non-blocking copy。
- CPU 使用普通 tensor copy，避免没有 CUDA runtime 时调用 `pin_memory()`。

OmniOVCD 本地提交：

```text
c9b306e fix: make SAM3 decoder coordinate cache device aware
4746803 fix: allow SAM3 positional cache on CPU
816a1ec fix: avoid hardcoded CUDA decoder cache
c2f6115 fix: support SAM3 geometry encoding on CPU
```

OmniOVCD 中原有的用户修改没有被覆盖，也没有把它们混入这些提交。

## 5. GPU 可见性问题定位

最初 Python smoke 显示 CUDA 不可用，但 `nvidia-smi` 后来确认主机有 8 张 L20。根因是两种执行上下文不同：

- 受限沙箱：看不到 `/dev/nvidia*`，Python 会得到 `cuda.is_available() == false`。
- 设备可见执行上下文：能访问 NVML 和 CUDA device，PyTorch 正常工作。

确认命令：

```bash
nvidia-smi -L
CUDA_VISIBLE_DEVICES=7 \
  /Data/wyh/CD-SegAgent/omniovcd-env/bin/python \
  -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

新增单卡配置：

```text
configs/runtime_gpu_l20.json
```

内容要点：

- `cuda_visible_devices: "7"`
- `device_map: "auto"`
- 单 GPU smoke policy
- 记录受限沙箱可能误报 CUDA 的说明

## 6. 测试和验证清单

### 6.1 单元测试

命令：

```bash
PYTHONPATH=. \
  /Data/wyh/CD-SegAgent/omniovcd-env/bin/python \
  -m unittest discover -s tests -v
```

结果：

```text
13 tests passed
```

覆盖内容：

- action JSON/fence 解析
- 坐标和 box 转换
- Environment reset/step
- 隐藏 state 不泄露
- finish 接受逻辑
- trajectory 保存
- connected components 和 instance matching
- Qwen 消息格式
- SAM3 public processor API
- SimpleClick external prev mask forwarding

### 6.2 Verifier smoke

```bash
PYTHONPATH=. \
  /Data/wyh/CD-SegAgent/omniovcd-env/bin/python \
  tools/train_verifier.py --smoke --count 8 --epochs 2
```

结果：loss 从约 `5.3406` 降至约 `5.3378`，CPU RSS 约 `630 MB`。

### 6.3 真实 LEVIR-CD Verifier smoke

```bash
PYTHONPATH=. \
  /Data/wyh/CD-SegAgent/omniovcd-env/bin/python \
  tools/build_verifier_samples.py \
  --dataset-root /Data/wyh/CD-SegAgent/OmniOVCD/dataset/LEVIR-CD/test_256 \
  --max-samples 2 \
  --output outputs/verifier_levir_smoke.npz

PYTHONPATH=. \
  /Data/wyh/CD-SegAgent/omniovcd-env/bin/python \
  tools/train_verifier.py \
  --samples outputs/verifier_levir_smoke.npz \
  --epochs 1 --batch-size 1 \
  --output outputs/verifier_levir_smoke.pt
```

结果：2 个样本、10 通道特征、1 epoch 训练成功，CPU RSS 约 `864 MB`。

## 7. Git 管理记录

Change-Agent 仓库的主要提交：

```text
2cf4ee2  初始 GT-free change-agent loop
da935f6  v0-v3 架构和验证文档
456002c  合并远端仓库基线
eb826f5  模型、Verifier、rollout、runtime smoke 工具
7b2597d  SAM3 processor adapter
60ca391  记录 Qwen3-VL CPU smoke
355b84e  真实 Qwen3-VL 接入 rollout environment
9863472  真实 adapter 和 Verifier 数据准备
d2002a1  单卡 GPU smoke、GPU 配置和 CUDA 资源记录
```

远端推送目标：

```text
git@github.com:HappynessI/Chang-Agent.git
```

Git 忽略规则确保以下大文件不进入仓库：

- `models/`
- 本地训练 checkpoint
- smoke 输出和日志
- Python cache

## 8. 当前边界和下一步

已经完成的是代码闭环、真实权重加载、独立适配器运行、单卡 GPU rollout 和 Verifier 训练准备。以下属于正式实验阶段，而不是基础代码缺失：

1. 用完整 LEVIR-CD/Change-Agent 数据集进行多 epoch Verifier 训练。
2. 对 point、box、finish 策略运行正式 ablation 和数据集级指标统计。
3. 将 SAM3、SimpleClick、Qwen 拆成服务或分阶段 pipeline，以便在单卡内完成完整双视图真实模型 rollout。
4. 在固定 GPU、固定随机种子和正式数据 split 下生成最终实验报告。

当前推荐策略是继续使用单卡 `GPU 7` 做短 smoke，正式大规模训练前再根据显存余量决定是否切换到多卡或服务化部署。
