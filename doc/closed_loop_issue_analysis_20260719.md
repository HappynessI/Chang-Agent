# Chang-Agent 三样本闭环测试问题分析与解决方案

## 1. 分析对象与结论

分析对象是成功运行目录：

`outputs/change_agent_levir_gpu_smoke_20260719_030624`

固定样本为 `test_20_15`、`test_78_13`、`test_85_16`。在线闭环使用 fresh
SAM3、Qwen3-VL-2B Agent、SimpleClick/SAM3 工具、Qwen zero-shot Verifier，最多
3 步，最终采用 `conservative_best`。GT 只在 `rollout_complete.json` 写入后用于
离线评估，本文中的 IoU、FP/FN 和 oracle 对比都属于闭环后的诊断，不能反馈给
在线决策。

核心结论如下：

1. **工程链路成功，安全机制有效，但优化闭环没有真正改善最终结果。** 三个样本
   最终都选择 step 0，实际被接受的 refinement 数为 0。最终汇总 IoU 从上次的
   `0.69690657` 变为 `0.69744116`，差值来自 fresh SAM3 初始预测的变化，不能归功
   于 Agent 闭环。
2. **Agent、Verifier、工具三层同时失效。** Agent 共 10 次遗漏必需坐标，导致前
   两个样本没有工具动作；Verifier 的 6 次定位全部是整图，且分数对候选没有区分
   能力；SimpleClick 的单点更新是全局 mask 替换，前两个候选发生大面积漂移。
3. **回滚策略避免了明显退化，但也暴露出 Verifier 的漏选。** `test_85_16` 的
   step 1/2 是灾难性候选，被拒绝是正确的；step 3 的离线 IoU 从 `0.30658070`
   提升到 `0.38875878`，仍因 Verifier 分数保持 0 而被拒绝。
4. **当前测试只能证明“不会轻易破坏初始结果”，不能证明“闭环能够优化结果”。**
   下一轮工作的重点不应是增加步数或放松回滚，而应是先修复动作参数生成、候选
   局部性和 Verifier 的相对比较能力。

## 2. 量化复盘

### 2.1 样本级结果

| 样本 | 初始 T1/T2 前景像素 | 初始 IoU | 初始 Verifier | 无效 Agent 输出 | 工具动作 | 最佳生成 step/IoU | 最终选择 |
| --- | ---: | ---: | --- | ---: | ---: | --- | --- |
| `test_20_15` | 0 / 5390 | 0.84638554 | 0.00 / false negative | 3 | 0 | step 0 / 0.84638554 | step 0 |
| `test_78_13` | 566 / 14186 | 0.75787116 | 0.85 / mixed error | 3 | 0 | step 0 / 0.75787116 | step 0 |
| `test_85_16` | 0 / 4342 | 0.30658070 | 0.00 / false negative | 4 | 3 | step 3 / 0.38875878 | step 0 |

汇总现象：

- 结构化动作失败 10 次，均为 point 缺 `coordinate` 或 box 缺 `box`。
- 只有 `test_85_16` 执行了工具；3 个 candidate 全部被回滚。
- 6 个 Verifier entry 的 `error_region` 全是 `[0,0,1000,1000]`。
- accepted refinement 数为 0，selected prediction 与 initial prediction 完全一致。
- 如果在闭环后用 GT 对每个样本生成过的 step 做 oracle 选择，汇总 IoU/F1 为
  `0.75206355`/`0.85848889`，高于实际的 `0.69744116`/`0.82175592`。这只说明
  “候选集合中存在更优结果但选择器没有识别”，不构成在线使用 GT 的理由。

### 2.2 `test_20_15`

T1 是基本没有建筑的林地，T2 出现多栋新建筑。SAM3 给出空 T1 mask 和 5390 像素
的 T2 mask，初始 change mask 与 GT 已高度一致，IoU 为 `0.84638554`。

Verifier 却给出 0 分和 `false_negative`，理由是“T1 mask 全黑，不能代表真实场景”。
这与图像内容和 change 结果不一致。离线错误中 FP=613、FN=254，错误像素的
70.7% 实际是 FP，并非以 FN 为主。错误诊断随后要求在 T1 做 positive point；Agent
连续 3 次只输出 action 类型而不输出坐标，最终安全停止。这里没有执行工具反而
保护了一个已经较好的初始结果。

### 2.3 `test_78_13`

初始 IoU 为 `0.75787116`。Verifier 同时输出 `quality_score=0.85` 和
`mixed_error`，定位仍为整图。0.85 已高于配置的 accept threshold 0.82，但运行时
只有 `error_type=none` 才可能 accept，因此该分数与错误语义缺少一致的标尺。

离线错误为 FP=3246、FN=430，88.3% 的错误像素是 FP。Agent 被要求执行 box，但
连续 3 次只输出 `{"target_view":"t1","action":"box"}`，没有 box 坐标，样本
同样以零工具动作结束。

### 2.4 `test_85_16`

这是本轮最能暴露闭环问题的样本：

- SAM3 在 T1 检出 0 个对象，在 T2 返回 21 个检测 mask，二值合并后为 19 个连通
  组件；T2 score 均值仅 `0.61458`，明显低于 `test_20_15` 的 `0.73047` 和
  `test_78_13` 的 `0.82109`。
- T2 场景里有大量屋顶状目标、车辆或临时结构，`building` 文本提示把许多非 GT
  建筑变化一起纳入。GT 有 1535 个前景像素，初始预测有 4342 个；FP=2963、
  FN=156，95.0% 的错误像素是 FP。
- Verifier 仍诊断为 `false_negative`，程序据此推导 `positive_point`。也就是说，
  工具方向与主要误差方向相反。

三个真实动作如下：

| step | 归一化点 / 像素点 | candidate 面积比例 | 离线 IoU | Verifier 分数 | 结果 |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | `[420,600]` / `[107,153]` | 0.474716 | 0.00107326 | 0.0 | 分数未提升、面积跳变，拒绝 |
| 2 | `[427,427]` / `[109,109]` | 0.265015 | 0.00026458 | 0.0 | 分数未提升，拒绝 |
| 3 | `[500,300]` / `[128,76]` | 0.012772 | 0.38875878 | 0.0 | 分数未提升，拒绝 |

由于每次拒绝后都会回滚到初始状态，这三个候选是从同一个 4342 像素的 T2 mask
分别生成的。step 1/2 的 SimpleClick 输出分别膨胀到 31111 和 17368 像素；step 3
收缩到 837 像素，显著提高精度但损失召回。Verifier 对跨度极大的四个状态始终给
0 分、整图定位和 `false_negative`，说明它没有形成可用于排序的质量信号。

## 3. 根因定位

### 3.1 指标解释错误：本轮提升来自初始化，不来自闭环

`Trajectory.best_entry` 只在 accepted entries 中按 Verifier 分数选择。本轮没有任何
candidate 被接受，所以三个样本全部选择初始 step 0。将本轮汇总指标与上次直接
比较会造成“闭环略有提升”的错觉；实际上两次 fresh SAM3 初始 mask 略有差异。

此外，runner 只设置了 Python 和 NumPy seed（`tools/run_levir_change_agent.py:82-83`），
没有设置父进程与 worker 的 PyTorch/CUDA seed，也没有声明 deterministic 策略。
trajectory 中 `git_commit` 还是 `null`，因为 `git rev-parse` 依赖启动目录而没有显式
指定项目 cwd。这些都会削弱跨次结果可比性。

### 3.2 初始分割：通用 `building` 提示对遥感小目标存在语义过检

SAM3 public processor 已用 0.5 detection confidence threshold 过滤实例，但不同
样本的 score 分布差异明显。`test_85_16` 的低置信、多实例结果把屋顶状车辆和
临时结构大量视作建筑变化。当前 `t12_min_instance_area=0`、
`cd_min_instance_area=0`，所有小连通组件都会进入 change mask。

基于已存档实例做的**闭环后诊断消融**显示：

| 二次最小 detection score | `20_15` IoU | `78_13` IoU | `85_16` IoU |
| ---: | ---: | ---: | ---: |
| 0.55 | 0.858175 | 0.774137 | 0.408680 |
| 0.65 | 0.772727 | 0.797128 | 0.566463 |
| 0.70 | 0.569996 | 0.797128 | 0.630369 |

较高阈值能显著改善 `85_16`，但会损害 `20_15`，因此不能从三个样本直接硬编码
0.65 或 0.70。对实际 step-0 change mask 做连通域过滤时，最小面积 256 的离线
IoU 分别变成 `0.846836`、`0.794120`、`0.365967`，说明面积过滤值得在独立验证集
上调参，但它也无法单独解决 `85_16` 的大块语义误检。

### 3.3 Agent：需要编辑 T1/T2 mask，却看不到这两个 mask

Agent 输入在 `change_agent/adapters/qwen3vl_adapter.py:42-64` 只有 T1 原图、T2
原图和 current change mask；它看不到自己实际要编辑的 predicted T1/T2 object
mask。Verifier 虽然看五张图，但传给 Agent 的只是一段文本反馈。本轮 feedback
又全部是整图区域，因此 Agent 缺少可落到局部对象的视觉依据。

动作生成还把“选择 view/action”和“输出精确坐标”放在同一次自由文本生成中，
没有 JSON grammar 或 schema constrained decoding。retry 只把校验错误追加到原
prompt（`tools/run_levir_change_agent.py:261-279`）；greedy generation 在相近输入上
容易重复同一缺字段 JSON。增加 retry 次数不会从根本上解决问题。

### 3.4 Verifier：没有比较基准，却承担 candidate 排序

当前 Verifier 每次只看到 candidate 的五张图。它收到 previous score 和 previous
action 文本，但看不到 previous change mask、previous T1/T2 mask 或 candidate delta。
`score_delta` 是“新一次绝对分数减旧分数”，不是对两个状态做视觉配对比较。因此
它很难判断一次工具动作究竟改善还是破坏了状态。

本轮的具体失真包括：

- 高 IoU 的 `20_15` 得 0 分，较低 IoU 的 `78_13` 得 0.85 分，绝对分数未校准。
- `85_16` 从 IoU 0.00026 到 0.38876 的候选都得 0 分，样本内也无法排序。
- `20_15` 和 `85_16` 的错误主要是 FP，却被判断为 false negative，程序进而派生
  出相反方向的 positive point。
- 第二阶段 localization 返回整图仍通过 `_parse_region` 的语法校验；代码没有检查
  region 是否局部、是否与白色 change mask 或 candidate delta 满足诊断语义。
- Agent 和 Verifier 共享同一个 Qwen3-VL-2B 权重，节省显存但会产生相关错误，
  不能把同一模型的自我评价当作独立证据。

### 3.5 SimpleClick：名义上的点修正实际是全局 mask 替换

`SimpleClickAdapter.refine` 把 initial mask 作为输入，但直接返回完整 prediction；
`ActionExecutor` 随后用它替换整个目标时相 mask。单个 point 因而可以把 6.6% 的
mask 改成 47.5%，并非局部编辑。

当前安全门只有：Verifier 有效、分数提升、change-mask 绝对面积变化不超过 0.25。
面积门只能拦住 step 1；step 2 的面积变化约 0.199，虽未超过 0.25，离线 IoU 却
接近 0。它没有检查变化是否集中在点击/box 附近，也没有检查连通域或拓扑是否发生
异常全局变化。

### 3.6 可观测性不足

runner 在内存中的 `rollout_records` 保存了 `episode_stop_reason`，但最终没有把完整
记录持久化。`invalid_agent_outputs.json` 也是扁平列表，没有 loop step、attempt
index、对应 feedback、prompt hash 等信息。因此可以确认前两个样本重试耗尽，却
无法从最终文件精确还原每个无效输出属于哪一轮动作。

## 4. 解决方案

### 4.1 P0：先补齐可复现和审计，不改变模型决策

1. 新增每样本 `episode_summary.json`，持久化 stop reason、accepted/rejected 数、
   action retry 的 loop index/attempt index、原始输出、校验错误和 prompt hash。
2. 显式从项目路径执行 `git -C <repo> rev-parse HEAD`，记录 dirty diff 状态、模型
   文件 checksum、PyTorch/CUDA/Transformers 版本。
3. 父进程及 SAM3/SimpleClick worker 都设置 Python、NumPy、PyTorch、CUDA seed；
   明确记录是否启用 deterministic algorithms，而不是只写 `seed=42`。
4. 离线报告新增 action-valid rate、accepted-refinement rate、degenerate-localization
   rate、Verifier ranking regret 和 oracle-step 指标。oracle 只用于闭环后诊断。

验收标准：同配置复跑能解释所有差异；每个 stop 都有持久化原因；无效输出能映射
到具体 step/attempt；trajectory 的 commit 不再是 null。

### 4.2 P1：把 Agent 改成“离散决策 + 受约束定位”

1. Agent 视觉输入与 Verifier 对齐，增加带标签的 predicted T1 object mask 和
   predicted T2 object mask。它必须看到自己要编辑的状态。
2. 将一次生成拆成两阶段：
   - 阶段 A 只输出 `target_view` 和 `action`；
   - 阶段 B 只完成缺失的 point/box 参数。
3. 优先避免让 2B 模型自由生成数值坐标。系统从以下证据生成有限 region proposal：
   change-mask 连通域、SAM3 低置信实例、previous/candidate delta 连通域、Verifier
   error map。模型只选择 `region_id` 和正负动作，系统再把 region 转为中心点或 box。
   这属于“模型选择的、证据驱动的参数化动作”，不是恢复任意 runner fallback。
4. 若仍使用 JSON 数值，接入 constrained decoding/JSON schema；repair prompt 只问
   缺失字段，并回显无效 payload 和一个同 action 的最小合法例子。
5. 参数补全失败继续安全 no-op，不恢复自动大框。

验收标准：固定三样本 action schema 有效率 100%；当 Verifier 给出 actionable
feedback 时，不再因缺字段导致整样本零工具动作。

### 4.3 P1：Verifier 从绝对打分改为 pairwise candidate gate

candidate commit 不应依赖零样本绝对分数差。推荐拆成两个任务：

1. **Pairwise gate**：同时展示 previous 与 candidate 的 T1 mask、T2 mask、change
   mask，以及高亮的 delta mask，输出 `better`、`worse` 或 `uncertain`。commit 只
   依赖 pairwise 结论和硬安全门。
2. **Absolute stop head**：只判断当前状态是否足够好，可以 finish；它不承担相邻
   candidate 排序。
3. localization 不再完全自由生成 box。优先在 candidate delta 或可疑实例 proposal
   中选择；若仍返回整图，除非 delta 本身覆盖大部分图像，否则标记
   `localization_valid=false` 并重试。
4. 为 FP/FN 加程序一致性检查：false positive region 应主要覆盖当前白色 change；
   false negative region 应主要位于当前 change 外。诊断文本、error type、target
   view 与 region 不一致时不能派生工具动作。
5. 不把共享 Qwen 的单一结论视为充分证据。至少加入 mask area、SAM3 score、局部
   delta、时相 overlap 等独立 proxy；中期接入专门训练的 verifier head。

验收标准：在存档候选上能把 `85_16` step 1/2 排在 initial 之后，并识别 step 3
优于 initial；整图 localization 比例从 100% 降到 0；FP/FN 方向与离线审计在该
三样本上不再出现明显相反。验收使用 GT 仅做离线评分，不进入在线输入。

### 4.4 P1：把 point/box 工具变成真正的局部编辑

1. positive point：只合并 prediction 中包含点击点的连通组件，或只在选定 ROI 内
   更新；不能用整张 prediction 替换旧 mask。
2. negative point：从旧 mask 中移除点击组件或 ROI 内被工具判定为负的区域，保留
   其他组件。
3. box：只允许 box 内替换，box 外保持旧 mask。
4. 新增 locality gate：记录 target-mask XOR，检查变化像素落在 action ROI 外的比例、
   最大变化连通域、组件数变化和相对面积变化。绝对面积阈值 0.25 只能作为最后一
   道粗门。
5. 每个动作可生成少量 candidate（例如局部 merge、局部 replace），交给 pairwise
   gate 选择；不要通过增加 max_steps 盲目试错。

验收标准：单点动作的绝大多数变化位于约定 ROI；不再出现一次 point 把 mask 从
6.6% 扩到 47.5%；拒绝候选不会影响 live state。

### 4.5 P2：校准初始化，而不是硬编码本次最优阈值

1. 保留 SAM3 每实例 mask 与 score，构造多阈值/稳定性候选，而不是先永久合并成
   一个二值 mask。对跨阈值稳定的实例提高置信，对只在低阈值出现的实例降权。
2. 在独立验证集上联合调 `confidence_threshold`、`t12_min_instance_area`、
   `cd_min_instance_area` 和 overlap threshold。不能依据这三个测试样本选择 0.65、
   0.70 或面积 256。
3. 对遥感场景增加更明确的语义提示或候选集，例如永久建筑、屋顶、临时结构，
   再由时相一致性和专用 verifier 判别；避免单一 `building` 把车辆/临时棚屋全部
   当作目标。
4. 训练 verifier 时使用真实 SAM3/SimpleClick rollout candidate，而不仅是从 GT
   做腐蚀/膨胀的合成扰动。按地理区域/场景划分 train/validation/test，避免样本
   泄漏；运行时仍不读取 GT。

建议评价指标包括 pairwise accuracy、Spearman/Kendall 排序相关、quality
calibration error、error-type macro F1、localization IoU，以及最终 accepted action
对离线 IoU 的正/负影响比例。

## 5. 推荐实施顺序

1. **审计与复现**：episode summary、attempt 索引、git/seed/version 记录。
2. **Agent 可执行性**：增加 T1/T2 object mask 输入，两阶段动作和 region ID 协议。
3. **Verifier 可比较性**：previous/candidate pairwise gate、delta proposal、整图区域
   失效规则。
4. **工具局部性**：point/box 局部 merge 与 locality gate。
5. **初始化校准**：在更大验证集上做 score/面积/匹配消融。
6. **再跑闭环**：先复用这三个固定样本做回归，再扩展到至少 30--100 个独立样本
   统计稳定性，不在三个样本上决定正式阈值。

下一轮三样本回归至少应满足：

- 3/3 样本结构化动作可执行；
- degenerate full-image localization 为 0；
- accepted harmful candidate 为 0；
- 至少 1 个样本 selected step 优于 initial step；
- 汇总 selected 指标不低于同一次运行的 initial 指标；
- 在线链路继续保持 `gt_loaded=false`。

## 6. 不建议采用的“快捷修复”

- 只增加 action retries：当前近似 greedy 的错误会重复。
- 恢复自动大 box fallback：上次已经证明会稳定扩大错误。
- 将 `selection_epsilon` 调成负数或接受同分 candidate：会同时放行
  `85_16` step 1/2 的灾难性候选。
- 直接选择 last state：本轮 step 1/2 已证明最后状态不具备安全性。
- 根据三个样本硬编码 SAM3 score 0.7 或面积 256：存在明显样本依赖和测试集过拟合。
- 把闭环后的 GT 指标交给 Agent/Verifier：这会破坏 GT-free 评估边界。
- 在 Verifier 尚无排序能力时增加 max steps：只会增加 GPU 开销和错误候选数量。

## 7. 证据路径与代码定位

- 运行参数：`outputs/change_agent_levir_gpu_smoke_20260719_030624/run_manifest.md`
- 指标：`outputs/change_agent_levir_gpu_smoke_20260719_030624/per_sample_metrics.json`
- 轨迹：`outputs/change_agent_levir_gpu_smoke_20260719_030624/trajectories/*/trajectory.json`
- 无效动作：`outputs/change_agent_levir_gpu_smoke_20260719_030624/trajectories/*/invalid_agent_outputs.json`
- SAM3 初始化证据：`outputs/change_agent_levir_gpu_smoke_20260719_030624/tool_runs/*/sam3_initialization/initialize_000/report.json`
- Agent 视觉输入与 prompt：`change_agent/adapters/qwen3vl_adapter.py`
- Verifier 视觉输入、定位和评分：`change_agent/adapters/qwen3vl_verifier.py`
- candidate commit/rollback：`change_agent/environment.py`
- SimpleClick 全 mask 返回：`change_agent/adapters/segagent_adapter.py`
- retry 与 GT 隔离：`tools/run_levir_change_agent.py`

本文初版分析没有修改运行逻辑，也没有使用 GT 重新选择或覆盖本次成功输出；所有
阈值消融只用于解释问题和制定下一阶段实验方案。下述第 8 节记录后续实现。

## 8. 2026-07-19 第一轮修复记录

本轮只修改 Prompt、Verifier 输入/评分协议和 Agent 动作格式提示，不启动 GPU 闭环：

1. Verifier Prompt 明确定义四种时相关系：新增建筑、消失建筑、不变建筑和不变
   背景。只有新增/消失属于最终 change mask；不再使用“仅存在于一张图”作为未
   区分方向的判断依据。
2. Prompt 明确 predicted T1/T2 mask 是辅助模型输出而非 GT，空 T1 或空 T2 mask
   不自动代表错误；Verifier 必须优先评价最终 candidate change mask，并回看两张
   原图判断空 mask 是否合理。
3. candidate Verifier 固定接收 T1/T2 原图，同时接收上一有效状态的 T1/T2/change
   mask、当前 candidate 的 T1/T2/change mask，以及产生 candidate 的归一化动作。
   被拒绝候选之后仍从上一有效状态比较，不把 rejected candidate 当作基准。
4. 第一阶段只输出 `quality_score`、`progress_score`、`error_type` 和 `feedback`。
   `quality_score` 是 candidate 的绝对质量；`progress_score` 位于 `[-1,1]`，独立
   表示 candidate 相对上一有效状态的改善或退化。初始状态固定为 0。动作候选门
   使用 `progress_score`，旧 `score_delta` 仅保留为审计/兼容字段。
5. actionable error 的第二阶段定位同样看到 previous/candidate 成对 mask 和当前
   动作，只输出 `target_view` 与 `error_region`；运行时继续派生 suggested action。
6. Agent 正常 Prompt 根据 Verifier 的 `suggested_action` 动态注入唯一一个 point、
   box 或 finish JSON 格式示例。point 示例必含 `coordinate`，box 示例必含 `box`，
   finish 不含二者，并继续禁止模型输出 `coordinate_frame`。
7. action retry 现在同时回传上一条无效 JSON。若 point 缺少/写错 `coordinate`，
   或 box 缺少/写错 `box`，repair prompt 会保留原 `target_view` 和 `action`，给出
   同结构精确模板并要求用 `[0,1000]` 数值替换占位符。

相关实现：

- `change_agent/adapters/qwen3vl_verifier.py`
- `change_agent/adapters/qwen3vl_adapter.py`
- `change_agent/environment.py`
- `change_agent/state.py`
- `change_agent/verifier.py`
- `change_agent/trajectory.py`
- `tools/run_levir_change_agent.py`

本轮 CPU 单元测试覆盖成对视觉输入、四关系 Prompt、独立 progress 解析、动态单示例、
缺字段定向重试、previous-state 传递和回滚兼容。下一步应先用固定三样本做闭环回归，
再根据轨迹中的 `quality_score`/`progress_score` 与离线 IoU 进行审计。

## 9. 2026-07-19 第二轮闭环安全修复记录

本轮继续完成分析报告中优先级最高的四项工程修复，仍未启动 GPU 闭环：

1. **局部工具组合与 locality gate**
   - positive point 只合并 SimpleClick prediction 中包含点击点的连通组件；其他
     全局 prediction 组件不会进入 live mask。
   - negative point 只移除当前 target mask 中包含点击点的连通组件。
   - box 只替换 action box 内像素，box 外严格保留上一 target mask。
   - 每次动作记录 target-mask XOR、ROI 外变化比例、target-mask 变化比例、最大
     变化组件、动作前后组件数；环境增加对应硬拒绝门。
2. **Verifier localization 与 FP/FN 一致性**
   - near-full-image region 在 candidate delta 不够大时标记为退化定位。
   - false-positive region 必须覆盖足够的当前白色 change；false-negative region
     必须主要位于白色 change 外。
   - 定位失败不再立即结束，而是携带具体校验错误重试；所有 localization attempts
     和检查统计进入 Verifier evidence。
3. **Agent 可见 predicted T1/T2 masks**
   - `AgentObservation` 和 Qwen3-VL messages 新增当前 predicted T1/T2 object mask，
     并明确它们是待编辑的模型输出而非 GT。
4. **审计与复现**
   - 无效动作记录 `loop_index`、`attempt_index`、原始输出、错误、prompt SHA-256
     和时间戳。
   - 每样本新增 `episode_summary.json`，记录 stop reason、候选接受/拒绝数量、工具
     数量、无效尝试、耗时和最终选择。
   - trajectory 显式从仓库路径解析 commit/dirty；run manifest 增加 Python、平台、
     parent seed、模型元数据 checksum 和 deterministic-policy 状态。
   - 隔离工具通过 seed wrapper 在模型构造前设置 Python/NumPy/PyTorch seed，并将
     seed runtime 写入 worker report。

新增/更新的 CPU 回归测试覆盖 clicked-component merge/remove、box ROI preservation、
global point rejection、full-image localization retry、FP region consistency、Agent mask
输入、prompt hash、step/attempt audit、Git metadata、模型 metadata checksum 和父子进程
seed 记录。正式有效性仍需下一轮固定三样本 GPU 闭环及闭环后 GT 审计确认。

## 10. 2026-07-19 SimpleClick 会话与 initial mask 根因修复

复查第二轮实现后确认，局部组合和 locality gate 只阻止了整图 prediction 直接写入
live mask，但当时仍存在两个底层问题：每次 point 都新建 `Clicker`，且 SegAgent 的
`SegmentationModel.get_prediction(..., mask=...)` 实际忽略 `mask` 参数。因此该版本
不能称为真正的连续 SimpleClick 编辑。

本轮从会话协议和 predictor 调用链修复：

1. Environment 分别维护 T1/T2 的 point session initial mask 与已接受点击历史。
2. 每次隔离 worker 从 session initial mask 开始，按顺序逐次重放历史点击，再执行
   当前候选点击。这保留了隔离进程设计，同时等价重建 SimpleClick Clicker 会话。
3. 候选只有通过 progress、area、locality 和 Verifier 门后才提交点击；被拒绝的点击
   不会污染后续历史。
4. accepted box 会以 box 编辑后的 mask 重开对应视图的 point session；另一视图的
   session 不受影响。
5. `SimpleClickAdapter` 绕过会吞掉 `mask` 的 SegAgent 包装方法，直接调用底层
   `predictor.get_prediction(clicker, prev_mask=...)`，并设置 external-mask 所需的
   `click_indx_offset=1`。
6. worker report 与 action evidence 记录 session click count、accepted click history
   和 session initial-mask 像素数，便于闭环后审计。

local clicked-component composition 与 locality gate 继续保留为最终安全边界。CPU
回归覆盖 initial mask 的实际逐次传递、点击顺序重放、reject 不提交、T1/T2 会话隔离
以及 accepted box 重置会话；真实模型效果仍需下一轮固定样本 GPU 闭环确认。

## 11. 2026-07-19 Delta-effect Verifier 与重复候选修复

对 `outputs/change_agent_levir_gpu_closed_loop_20260719_173613` 的复盘确认：
`test_20_15` 的 149 个 candidate-added 白色像素被错误标成 false negative，
`test_78_13` 的逐像素相同候选先后得到 worse/invalid/better，而 verbose 六区域 JSON
仍可能在后部截断。本轮据此完成以下结构性修改：

1. 初始区域输出压缩为 `region_id -> [verdict,target_view]`，不再要求每区域自然语言；
   普通白色 change component 不允许标成 false negative，该标签必须来自
   `temporal_difference_missing` proposal。
2. 候选阶段只展示本次 candidate delta：新增像素聚合成一个 panel、删除像素聚合成
   一个 panel。Qwen 只输出 added/removed supported/unsupported/uncertain 效果标签。
3. `better/worse/uncertain` 完全由程序从效果标签推导，删除第二次 Qwen pairwise 自由
   判断；任何 harmful 标签即 worse，任何 uncertain 即 uncertain，全部 beneficial 才
   能 better。
4. Verifier 按 previous masks、candidate masks 和 action 的 SHA256 缓存候选裁决；
   Environment 同时禁止在未变化 live state 上再次执行完全相同的已拒绝动作。
5. replay challenge 按线上局部 point/box composition 重建 temporal candidate mask；
   Verifier 默认输出预算降为 256。Git 审计新增 tracked diff 与 untracked contents 的
   `git_worktree_sha256`，避免再次只留下 `commit + dirty=true`。

CPU 全量回归通过；没有在未重新确认迁移后 GPU 配额与通知策略的情况下启动 GPU
任务。下一步应先 replay 上述 9 个存档候选，再跑固定三样本闭环，重点检查
`test_85_16` 的 T1-empty 语义是否仍被模型误解。
