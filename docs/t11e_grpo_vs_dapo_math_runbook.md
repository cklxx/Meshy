# T11e 执行单：GRPO vs DAPO，MATH，单卡 V100，2×N 窗（N 由 §3.2 smoke 量定，目标 40）

> **两道硬门，未过不开正式窗**：①§3.0 先量基座 MATH500 500×4，lenient pass@1
> <15% 直接停（任务对 0.6B 过难，DAPO 丢组会主导）；②§3.2 DAPO 2 窗计量，
> 按实测补样倍率 R 定 N=40 或两臂同降 20。本文档以 40 为默认窗数书写。

状态：可执行单（不是提案）。owner=rl。前置依赖 T7（env，wip），见 §6。
所有路径/默认值都已对齐当前 `v100/math`（含 HendrycksMATH、math_verify 判分、
tuple 修复、空 gold 过滤）与 v100/rl 单卡 colocate 基建。

> **冻结代码（v100/t11 @ origin）：`e692d26`**。含 T11i（T7 prompt 游标、T5h
> fail-loud DCP、T5i slim trajectory、T11b tuple 判分修复）、T11f flash-decode
> （`MESHY_SM70_FLASH_DECODE=1`、`XRL_DCP_CKPT_INTERVAL=3`，含 flash 接线/内核
> 三件套的 sentinel 实证修复）、`_rollout_metrics` staticmethod NameError 修复、
> T11 前置修复（DCP 目录按 recipe 解析、末尾 DCP 全量可 resume）与存储迁本地
> `/data00/meshy/store`（3FS 退役）。冒烟部署树按
> docs/sm70_flash_decode_integration.md 以 merge 方式更新到此 hash，不用 rsync。
> partial rollout 刻意不在树上，两臂均不开。

---

## 0. 一句话实验

在**完全相同**的模型/数据顺序/窗数/批大小/lr/长度/显存-图配置下，只切换
RL 算法族 GRPO→DAPO 的两件套（零方差组动态补样、soft-overlong 惩罚），优势函数
两臂**相同**（组内 `(r−mean)/std`；“去 std”是 Dr.GRPO 的做法，不是 DAPO 原文，
见 §1.2），用 MATH500 500×4 lenient 主指标判断 DAPO 在 MATH 上是否带来真实增益。

---

## 1. 两份 recipe 与受控量

新增两份 recipe，都 fork 自 `recipe/grpo_gsm8k_v100.py`（单卡 colocate 骨架），
数据集与判分换成 Hendrycks MATH：

- `recipe/math_grpo_v100.py`（对照臂，GRPO）
- `recipe/math_dapo_v100.py`（实验臂，DAPO）

共用一个带参数的基类/工厂（建议在 `recipe/v100_math_common.py` 放
`build(arm: "grpo"|"dapo")`），两文件只传 arm 标志，**杜绝两份文件漂移**。

### 1.1 必须逐字相同的量（受控，硬编码在 common，不允许 env 覆盖出不同值）

| 量 | 固定值 | 来源/理由 |
|---|---|---|
| 基座模型 | `/data00/meshy/models/Qwen3-0.6B`（同一份权重、同 hash） | 冷启动两臂都从它起 |
| 启动点 | 都从 **window 0 冷启动**（`XRL_START_WINDOW=0`，无 DCP） | 这次不热启动，规避 T7；两臂 genesis 完全一致 |
| 数据集 | `meshy.dataset.hendrycks_math:HendrycksMATH`，split=train | 已过滤 2 行空 gold（7498 行） |
| 数据顺序 seed | `seed=42`（两臂同一 seed → 逐窗 prompt id 序列相同） | 配对比较的核心 |
| 窗数 | **40 窗**（`RL_STEPS=40`，LR scheduler horizon=40） | 一一对应 |
| 每窗 prompt 数 | **`XRL_ROLLOUT_BATCH=64`（env 覆盖值，不是 recipe 默认 8）** | clean40b 实测口径；启动命令必须显式带，见 §7 |
| 每 prompt 采样数 | `GROUP_SIZE=8` | completions/prompt |
| 每窗样本数 | **64 prompts × 8 = 512 样本/窗**（`BATCH_SIZE = ROLLOUT_BATCH*GROUP_SIZE = 512`） | clean40b trajectories 实测每 weight_version 恰 512 行（基线，未触发补样的窗） |
| trainer batch | `BATCH_SIZE=512`，`mini_batch=64`（512/64=**8 个优化更新/窗**），micro=1，per-token 微批 `max_tokens_per_micro=4096`，`seq_bucket=64`→seq_align=64 | 钉成 clean40b 实跑 init（mtpm4096/mini64/seq64），两臂相同；v100_run_rl.sh 在 MATH recipe 下强制 |
| lr / schedule | `lr=1e-6`，weight_decay=0.1，max_norm=1.0，**warmup=0，lr_decay_ratio=0（常数）** | 40 窗太短，常数 lr，两臂一致 |
| 训练精度 | `XRL_TRAIN_DTYPE=float32` master + fp16 FSDP compute + GradScaler | sm70 必需，勿用 uniform fp16 |
| seq_len | `XRL_SEQ_LEN=5120` | 同 gsm8k |
| 生成上限 | `MAX_NEW_TOKENS=4096`，temp=1.0 / top_p=1.0 / top_k=-1 | RL 采样口径，两臂一致 |
| reward 任务分 | `meshy.dataset.hendrycks_math:HendrycksMATH.reward`（math_verify，0/1） | 同一判分 |
| KV / graph | mem_fraction_static=0.60、**flash attn（`MESHY_SM70_FLASH_DECODE=1`，T3h sm70 warp-key flash-decoding）**、pytorch sampling、decode graph on（默认）、enable_memory_saver | 推理配置逐字一致 |
| colocate 时序 | 同一 ColocationRing（infer FALLBACK / train ON_DEMAND）、pacing_window=1、poll_interval=2.0 | 同一 GPU 手序 |
| 评测点 | 同一 hook、同一窗集合、同一题集、同一采样（见 §3） | 评测本身也要配对 |
| checkpoint | DCP 间隔/保留数一致，dump 到**不同 run tag 子目录** | 隔离，不互相 resume |

### 1.2 唯一的自变量（两列 diff 表，只列不同行）

DAPO 在本树由两个 env 开启（`recipe/grpo_math_v100.py` 唯一一份，两臂共用，
受控量从该文件读同一组值）：`XRL_DYNAMIC_SAMPLING=1 XRL_OVERLONG_SHAPING=1`。

| 配置行 | GRPO 臂 | DAPO 臂 |
|---|---|---|
| 动态补样 | 关（每组跑完即止，R=1） | **开** `dynamic_sampling=True`：丢零方差（全对/全错）组并补样到固定有效组数 |
| 补样硬上限 | 不适用 | `dynamic_max_prompts=192`（每窗 kept+dropped prompt 上限，=3×64 目标；env `XRL_DYNAMIC_MAX_PROMPTS`）。**真实生效的是它，不是 `oversample_factor`**（后者在 dynamic 路径不参与预算，见 rollout.py replacement_budget） |
| `advantage` | `meshy.worker.rollout:grpo_advantage`（组内 `(r−mean)/std`） | **完全相同**（见下说明） |
| soft-overlong 惩罚 | 关（reward 直通 0/1） | 开：reward_shaping=`meshy.reward:dapo_overlong_penalty`，见下 |
| length-reward 塑形 | 关 | 关——本实验不引入 length reward，只隔离 动态补样 + overlong |
| clip ε | `ppo_clip 0.2/0.28` | **完全相同** |
| TIS | 无 | **完全相同**（clip-higher/TIS 不放本次自变量） |
| PPO loss 聚合 | per-sequence（`advantage=column`，`calculate_per_token_loss=False`） | **完全相同**（同一份 trainer，非 token-level 平均） |

**为什么两臂 advantage 相同（不去 std）：** DAPO 原文（Yu et al. 2025）的优势仍是
组内 `(r−mean)/std`；“去掉 std、只减均值”是 **Dr.GRPO** 的做法，不是 DAPO。另外
树里的 `meshy.advantage:_2_6_math_reshaped_advantage` 自身再做一次 overlong，
若与 `reward_shaping=dapo_overlong_penalty` 同用会**双重计数**超长惩罚。故两臂都用
默认 `grpo_advantage`，overlong 只经 reward shaping 注入一次。

**overlong 的 L_max / 缓冲（仅 DAPO 臂，`dapo_overlong_penalty`）：**
- `max_response_len = L_max = 4096`（响应长度上限）。
- `cache_len = 1024` 是**末尾软惩罚缓冲宽度**（不是起点）：软罚区为
  `(L_max−cache_len, L_max) = (3072, 4096)`。长度 ≤3072 不罚；进入 (3072,4096)
  线性加罚 `(len−3072)/1024`，0→1；长度 ≥4096 或被截断（`sample.truncated`）硬罚 1
  （被截的回答无论 boxed 是否可解析都按错塑形）。
- 经 `reward_shaping_kwargs` 传：`max_response_len=4096`、`cache_len=1024`
  （env `XRL_OVERLONG_L_MAX` / `XRL_OVERLONG_L_CACHE`）。

> 落地（已在 recipe 实现）：`XRL_DYNAMIC_SAMPLING=1` →
> `dynamic_sampling=True` + `dynamic_max_prompts`（env `XRL_DYNAMIC_MAX_PROMPTS`，
> 默认 192）；`XRL_OVERLONG_SHAPING=1` →
> `reward_shaping="meshy.reward:dapo_overlong_penalty"` + 上述 kwargs。GRPO 臂两个
> env 都留 0。优势、clip、loss、训练几何两臂从同一 recipe 取同值。

---

## 2. 主对照轴：按**窗数**对齐，另外两轴记账备查

**结论：主轴 = 窗数（40 vs 40，逐窗配对）。** 理由：

1. 优化语义按窗定义——每一窗是一次「64 prompt × 8 采样 = 512 样本 → 一个 lr
   scheduler step」。两臂 lr schedule 步数相同（=N），第 k 次梯度更新在算法上
   应对齐；GRPO 第 k 步与 DAPO 第 k 步比，才是"同一点的两种算法"。
2. 评测 hook 在 colocate grant 回调里按 version 触发，天然按窗对齐（§3）。
3. 数据 seed 相同 → 第 k 窗两臂看到**同一批 64 个锚 prompt**；DAPO 丢组补的是
   额外 prompt，不改变"这一窗对应这 64 个锚 prompt"的配对关系。

**动态采样导致的样本消耗不等，不抵消、不重采样对齐**，而是**逐窗记录备查**：

- `prompts_consumed`（实际发起的 prompt 数 = 锚 8 + 补样）
- `groups_seen` / `groups_filtered`（零方差丢弃数，rollout 已有计数器）
- 实际生成样本数、有效训练样本数（进 TQ 的行数）
- 累计 GPU 秒（rollout 段 + train 段，timer 已开 `timer_enabled=True`）

为什么另两轴不作主轴但必须记：

- **按消耗样本数**对齐会系统性偏向 DAPO——它多花 2× prompt 才凑满有效组，
  在同 token 预算下 DAPO 完成的"优化窗"更少；拿它当主轴等于奖励浪费。但它是
  **成本轴**，要用来回答"DAPO 多花多少样本/GPU 小时换这点分"。
- **按 GPU 小时**对齐同理是效率轴：DAPO 补样增加 rollout 墙钟。报告在
  "每 GPU 小时解题率"上谁高，但不作为算法优劣的主判据。

交付时给三张对齐图/表：①按窗（主）②按累计消耗样本 ③按累计 GPU 小时，每张
都画两臂的 MATH500 lenient 曲线。只有①用于判胜负，②③是效率旁证。

---

## 3. 评测点、题集、采样、卡时

**题集：MATH500（test split，500 题），不用 train holdout。** 理由：T11c 已定
500×4 与 GSM8K 的 200×4 同口径可并排放；MATH500 与训练集实测 0 重叠，无泄漏；
train holdout 会和训练 prompt 撞。

**评测协议（两臂逐字相同）：**
- 采样：Qwen3 thinking，temp 0.6 / top_p 0.95 / top_k 20，**每题 4 个样本**
  （500×4 = 2000 请求），`max_new_tokens=4096`（in-loop 口径，不是 endpoint
  的 8192；**4096 只覆盖到约 p62**——基座 500×4 实测截断率 37.9%、p90 已顶
  4096，并非“覆盖 p95”）。
- 判分：lenient（math_verify 等价）为主，strict（规范化字符串）为辅，都报；
  同时报 boxed format rate 与 truncation rate。
- 主指标：**lenient 逐题平均正确率**（200×4/500×4 同一口径）；pass@1..8 也出
  （4 样本只支持 pass@1/2/4，pass@8 需 ≥8 样本，in-loop 不出）。
- hook：复用 `recipe/v100_inloop.py` 的 version_hook（已支持
  `XRL_EVAL_DATASET=math500`），它在权重 grant 后、trainer 阻塞时跑，保证评测
  用的就是该 version。

**评测点（40 窗内取 5 个点，避免太密拖慢）：**

| 点 | version（窗） | 题数×样本 |
|---|---|---|
| 基座 v0 | 0（首窗 grant 时） | 500×4 |
| 早期 | 10 | 500×4 |
| 中 | 20 | 500×4 |
| 后 | 30 | 500×4 |
| 终 | 40 | 500×4 |

即每臂 5 次 × 2000 请求，两臂共 10 次。

> 若 500×4 单次墙钟超预算，**降级方案两臂必须同时降**：统一改 200×4（仍是
> GSM8K 主指标口径），5 个点不变。不得只给一臂降级。设
> `XRL_EVAL_N`（500 或 200），两臂同一值。

**评测卡时估算（单卡 V100，spec-off 实测聚合吞吐 1060–2340 tok/s）：**
- 500×4 = 2000 请求，思考输出按均 500–1000 tok：约 **15–31 分钟/次**（中位
  ~22 分钟）；保守含 prefill/排队按 **~0.4–0.5 小时/次**。
- 每臂 5 次 ≈ **2.0–2.5 GPU 小时评测**；两臂 **≈4–5 GPU 小时**纯评测。
- 评测与训练卡时分离估算：评测不按每窗样本数外推，上面就是 10 次独立 500×4
  的实测口径。训练卡时见 §3.1，**两臂总卡时的决定项是训练，不是评测**。
- 评测点选择 5 个而非每窗评，把 40 次评测压到 5 次，省 ~87% 评测墙钟。

### 3.0 开跑第一硬门：先量基座 MATH500，再决定要不要做这个对照

**顺序在 §3.2 两窗计量之前**，是更靠前的 go/no-go。原因：DAPO 的零方差丢组
在任务对当前模型过难时会主导一切——`filter_zero_std_groups` 丢的是组内 8 个
奖励全同的组；0.6B 在 MATH 上早期会有大量"全错组"，被丢后动态采样一路撞 3×
上限仍可能凑不满有效组，有效批量不升反缩。这时两臂比出来的不是 GRPO 与
DAPO 的算法差，而是"超采样能否救一个对模型太难的任务"（答案大概率不能）。

**步骤 1（占卡，排 kern/rl 之后，不自行 hold）：** 跑基座 Qwen3-0.6B 的
MATH500 **500×4**（temp 0.6/top_p0.95/top_k20，4096），15–30 分钟，命令：

```bash
python scripts/eval_math.py --base-url <已起好的基座server> --data math500 \
  --samples 4 --n 500 --max-new-tokens 4096
# 无在跑 server 时让脚本自启：去掉 --base-url（单卡 sm70 配置脚本内置）
```

取输出的 **lenient pass@1**（逐题 4 样本平均正确率）作为门槛量。

**步骤 2（go/no-go，阈值写死）：**

| 基座 lenient pass@1 | 决定 |
|---|---|
| **≥ 15%** | **go**：继续 §3.1/§3.2（两窗计量→定 N→两臂对照）。15% 意味着组内有足够混合（全错组不是绝大多数），动态采样有可作用空间。 |
| **< 15%** | **no-go，不开两臂 40 窗**。把基座分数和下面两个替代方案一起报主控裁决：① 换 MATH **level 1–3 子集**（更易、全错组比例低）重做 §3.0 再判；② 直接在 **GSM8K** 上做 GRPO vs DAPO——已有 clean40b 完整曲线与 0.835 终点做现成锚点，对照更干净，代价是丢掉"MATH 换数据集"这个卖点。 |

> 15% 是经验门槛而非精算：它对应"平均每题 4 样本里至少有信号"。低于此，
> 2000 个基座采样中绝大多数组 8 个全错，DAPO 的丢组率会逼近上限，§3.2 的 R
> 几乎必然 >2，提前在数据侧挡掉比跑完 40 窗再发现"任务太难"省一整个实验。

**关于 clean40b 那个 1.09× 的澄清（它不是 R 的经验锚点）：**
clean40b 是 **GRPO 臂、`filter_zero_std_groups=False`，代码里根本没有丢组补样
路径**。40 个 version 里 38 个 trajectories 恰 512 行；v10（1536 行）和 v20
（1336 行）这两个高行 round 的真因是**两次崩溃重启导致的窗口重发**（clean40b
未开 partial rollout，不是 defer/partial 机制）：

1. **2026-09-30 00:19 推理侧**：权重交接 pause_generation /
   release_memory_occupation / resume_memory_occupation /
   update_weights_from_disk 都返回 200，紧接 torch_memory_saver 的 `cuMemCreate`
   报 `CUDA_ERROR_OUT_OF_MEMORY`（core.cpp resume:243 抛 OOM），scheduler_0
   退出码 1，SIGQUIT 清理。
2. **2026-09-30 01:07 训练侧**：torchtitan
   `F.scaled_dot_product_attention` 申请 282 MB 失败（全卡 31.74 GB 仅剩
   22.38 MB），TitanWorker fatal、01:10 重启。这是 step20 之前那套激进配置
   （graph bs128 + KV 扩容 + 关 AC）的后果；之后回退到 **AC full / mem
   fraction 0.60 / graph bs64** 正是为修它。

崩溃前已生成、崩溃后窗口重发又生成一遍，同一批 64 个 prompt 留下 2–3 批轨迹
（v10 每 prompt 24 条=3 批、v20 每 prompt 16–40 条不齐），**只有一批进梯度
更新**——TB `num_mini_batches` 在 v10/v20 恒为 8（=512/64）正好印证。

所以它**既不是 DAPO 补样、也不是 partial rollout，而是崩溃重启的重复生成
浪费**，不能用来预判 MATH 的 R。两个推论：①GSM8K 尚且 40 窗内 2 次崩溃，
MATH 全错组只会让 DAPO 真实 R 更高（§3.0 不能省）；②**崩溃重复生成是实测
频率（2 次/40 窗），两臂排期都要预留这部分重复墙钟**（见 §3.1 风险项）。



### 3.1 训练卡时（triton 口径上界；flash 落地后按实测重算）

> **本节所有小时数是当前 triton attention 路径的上界，先不要据此定最终窗数。**
> kern T3h 已实测 flash 对 triton 在 MATH 吞吐上是真增益：**mean +74.6% /
> median +128%**（平均上下文 1358 token）。生成段占每窗 20.5 分钟里的 14.5
> 分钟，是大头；flash 落地 rl 树后这段会显著缩短。**最终每窗墙钟与 N 一律以
> §3.2 两窗计量在"开了 flash 的生产路径"上的实测为准**，本节 16–20h / 29h
> 数字保留仅作 triton 口径的预算上界，量到 flash 实测后再改。

clean40b（GSM8K，512 样本/窗，单卡同配置，triton）实测**每窗墙钟 ≈ 生成
14.5 分钟 + 训练 6 分钟 ≈ 20.5 分钟**。这是 GRPO 臂（无补样）的 triton 基线：

- **GRPO 臂（triton 上界）**：约 20.5 分/窗 × 40 ≈ **13.7 GPU 小时**（MATH
  输出更长，生成段按 1.2–1.5× 估 ≈ **16–20 GPU 小时**；flash 落地后下修）。
- **DAPO 臂（triton 上界）**：`dynamic_max_prompts=192` 允许每窗最多发起 3×
  prompt 补零方差组，最坏生成段 14.5×3 ≈ 43.5 分/窗，40 窗 ≈ **29+ GPU
  小时**；加 GRPO 臂与两臂评测 4–5h，triton 口径总墙钟逼近或超 40 GPU 小时。
- clean40b 轨迹里 2/40 round（v10/v20）记了 512×2~3 行，已查实是**两次崩溃
  重启的窗口重发**（filter 关闭、无补样、未开 partial，训练仍只吃 512，见
  §3.0），**不是 DAPO 的补样倍率，不能锚 R**。MATH 的真实 R 完全未知，必须
  经 §3.0 → §3.2 实测。

**崩溃重复生成的风险项（要预留墙钟）：** clean40b 在 40 窗内实测发生 **2 次
崩溃重启**（推理侧 cuMemCreate OOM、训练侧 SDPA OOM 各一，配置回退后未再复
现），每次崩溃当窗的生成白做一遍。**两臂排期都按 ~2 次/40 窗的频率预留重复
生成墙钟**（约 +1 个生成段/20 窗）；§3.2 smoke 同时确认本配置（AC full /
mem 0.60 / graph bs64）在 MATH 长序列下不再 OOM，若 smoke 就崩，先停下解决
稳定性，不带着崩溃率开 40 窗。

因此顺序是 §3.0（基座分 go/no-go）→ §3.2（2 窗量 R 与 flash 实测墙钟）→ 定 N
→ 正式两臂；任一前置没过都不开正式窗。

### 3.2 开跑前置：DAPO 2 窗计量（gating，未通过 rl 不启动 40 窗）

依赖 env T11d（已完成，commit `8ec7b61`，需先把该提交合入运行分支；
v100/math 当前基于的 rl 快照不含它）。它把每窗动态/partial 统计写进
`<XRL_RUNTIME_DIR>/rollout_window_stats.jsonl` 并转发训练日志/TensorBoard，
无需额外 env 开关。

**跑法：** 只跑 DAPO 臂配方，`RL_STEPS=2`，独立 smoke run tag，512 样本/窗
（`XRL_ROLLOUT_BATCH=64 XRL_GROUP_SIZE=8`），其余全部 §1 受控量照最终配置
（含 overlong L_max=4096/buffer=1024）。用 MATH train 真实数据、seed=42、
window 0 冷启动。不开 in-loop 评测（设 `XRL_EVAL_EVERY` 大于窗数，如
`XRL_EVAL_EVERY=99`），只量训练。

**前提：在"开了 flash 的生产路径"上跑**（kern T3h flash 已合 rl 树后）。若
smoke 时 flash 尚未落地，§3.1 小时数只能先记 triton 上界，N 不作最终锁定，
flash 落地后需补一次同 2 窗确认墙钟。同时这 2 窗是稳定性验证：确认 AC full /
mem 0.60 / graph bs64 在 MATH 长序列下不出现 clean40b 那类 cuMemCreate/SDPA
OOM；一旦 smoke 崩溃，先停下来修稳定性（§3.1 风险项），不带崩溃率定 N。

**看哪几个指标（`rollout_window_stats.jsonl` 里 `kind="dynamic"` 每行）：**
- `prompts_drawn`：该窗实际发起的 prompt 数（基线 64）。
- `valid_groups` / `valid_samples`：通过零方差过滤、进训练的组/样本（理想每窗
  64 组 / 512 样本；少了就是被丢）。
- `groups_dropped_zero_variance`（=`refill_count`）：丢组数。
- **`filtered_ratio` = 丢弃组 / (有效组 + 丢弃组)**：核心量。
- 另外从日志/timer 读该窗**生成墙钟**与**训练墙钟**（分两段），以及 GPU 秒。

**补样倍率定义：** `R = prompts_drawn / 64`（每窗），取两窗的较大值与均值。
（等价于生成 token 量的放大倍数，因为每组都是 8 个采样。）

**阈值 → 决定（写死，量完照此执行，不再拍窗数）：**

| 实测 R（两窗） | filtered_ratio | 决定 |
|---|---|---|
| **R ≤ 1.5**（均值，且单窗 ≤1.5） | 约 ≤1/3 | **40 窗可行**，`XRL_DYNAMIC_MAX_PROMPTS=192`（R 上限 3）保持，上限用不满没关系；按 §3.1 重算总卡时确认 ≤ 预算后开跑 |
| **1.5 < R ≤ 2.0** | 约 1/3–1/2 | 两臂**同时降到 20 窗**（保持 512 样本/窗、同评测点比例 v0/5/10/15/20），或保持 40 窗但确认总卡时可接受；二选一在量完当场定，倾向 20 窗 |
| **R > 2.0**（接近 3 上限） | >1/2 | 把 **DAPO 单侧**的 `XRL_DYNAMIC_MAX_PROMPTS` 从 192 降到 **128**（R 上限 3→2），重跑 2 窗复量；若仍 >2，再降到 20 窗 |

> 注意：动态补样/overlong 是 **DAPO 单侧自变量**，GRPO 臂永远
> `XRL_DYNAMIC_SAMPLING=0`、R=1.0，不补样。表里“两臂同时改”只针对**窗数**
> 这种受控量；补样上限 `XRL_DYNAMIC_MAX_PROMPTS` 只调 DAPO。

**判定纪律：**
- 2 窗样本极小，早期窗又最难（基座全错组最多），R 可能是全程上界，所以阈值
  宁可保守。若两窗 R 一高一低，取高的那窗对照阈值（上界口径）。
- 同时确认 T11d 已验证的 partial/defer 行为：2 window × 512 下没有卡死、没有
  全量 defer（`kind="partial"` 行的 `groups_deferred` 不应等于全部组）。
- smoke 结论（R、filtered_ratio、每窗生成/训练墙钟、推荐窗数）落一条记录，
  追加到本执行单末尾或 awb news，rl 据此启动；**该结论未出，40 窗不开跑**。

#### 3.2.1 冒烟实测结果（2026-10-01，冻结树 5dc3605，已完成）

DAPO 配方 `XRL_DYNAMIC_SAMPLING=1 XRL_OVERLONG_SHAPING=1`，3 窗（前两窗 +
崩溃续跑第三窗），每窗 valid 64 组/512 样本：

| 窗 | prompts_drawn | R=drawn/64 | dropped zero-var | filtered_ratio |
|---|---|---|---|---|
| 1 (v0) | 160 | 2.50 | 96 | 60.0% |
| 2 (v1) | 168 | 2.63 | 104 | 61.9% |
| 3 (v2，续跑) | 150 | 2.34 | 86 | 57.3% |

R 稳定 **2.34–2.63（均 >2）**，零方差组约 60%（MATH 截断 38% + L4/L5 难题）。

分项墙钟（单窗全循环）：
- **生成**：纯 rollout 约 **61–87 min/窗**（窗1 72min 含 v0 inloop 5.4min；随
  R≈2.5 放大；4096 长尾）。
- **训练**（8 个优化更新，colocation grant→HF gather）：约 **23 min/窗**
  （21m56 / 23m17 / 23m09）。
- **DCP 落盘**（gather→step completed，含 HF 导出 ~4–5s）：约 **50 s/份**，
  每份 7.3 GiB 本地盘（含 fp32 master + Adam exp_avg/exp_avg_sq + lr_scheduler
  + train_state，`.metadata` 实证）。interval=3 时 40 窗 14 次 DCP 仅摊销
  ~12 min，可忽略。
- 峰值：host 最低可用 11.5 GB（31 GB 盒，未触发 earlyoom）；infer graph capture
  后 GPU 最低可用 11.2 GB。`/data00` 增量：3 份 ckpt 22 GB + 单 run rollout
  3.4 GB。

infer 侧（加 `max_running_requests` 前）：DAPO 补样队列使 running-req 峰值
**172**，超过 flash graph max_bs=64 的时间占 decode **28.7%**，超限回退 triton
eager，吞吐 1254→957 tok/s（−24%），并伴 1 次可恢复 prefill OOM。→ 两臂统一
`max_running_requests=64`（`XRL_MAX_RUNNING`，MATH recipe 默认 64），要求全程
decode `cuda graph: True`。

崩溃续跑演练（同 tag、STEPS=3、不设 START_WINDOW）通过：checkpointer 从 DCP
folder 载入（非 HF/非 0）、gate version 0→2、infer 从 v2 权重 restore、colocate
resume 后 graph capture 重建全部 12 个 flash plan（flash 仍 called）、step3 落
全量 DCP、ignitor 报 Run completed cleanly。

#### 3.2.2 最终拍板与 20 窗预注册规则（用户已拍板）

**全量 MATH（不切 L1–3 子集），两臂各 20 窗冷启动**，DCP interval=3 可续到 40。

- 两臂：`XRL_STEPS=20 seed=42` 冷启动 `start_window=0`；评测点 **v0/5/10/15/20**
  各 MATH500 **500×4（max_new=4096，in-loop）**；`XRL_DCP_CKPT_INTERVAL=3`
  且 **step20 必落**（interval 3 + 末尾步）。
- GRPO 臂：`XRL_RUN_TAG=math-grpo20`，两个 DAPO env 留 0。
- DAPO 臂：`XRL_RUN_TAG=math-dapo20`，`XRL_DYNAMIC_SAMPLING=1
  XRL_OVERLONG_SHAPING=1`。
- 两臂同 `XRL_MAX_RUNNING=64`、同几何（512/8，mtpm4096/seq64）、同评测点。
- 串行：先 GRPO，放卡再 DAPO，同一张卡。

**v20 判胜规则（跑完 v20 才按此判，不改阈值）：**
1. **定胜负**：`Δ = lenient(v20_DAPO) − lenient(v20_GRPO) ≥ +3.0pp` **且**
   DAPO 臂 v15→v20 不回落 → DAPO 胜（Δ≤−3pp 且 v30 同向为负时为 DAPO 负，
   但 20 窗无 v30，故 20 窗只判正向胜/平/续跑）。
2. **续到 40**：若任一臂 v15→v20 仍 **≥+1.5pp**，或 Δ 落在 **[+1.5,+3.0)pp**
   → 两臂同 tag 改 `XRL_STEPS=40` 续跑（LR 恒定、warmup=0，续跑与一次性 40 窗
   等价），补 v25/30/35/40 评测点再判。
3. **判平**：其余情况（|Δ|<1.5pp 且两臂都已收敛）。

**终点 8192 重评（评测侧，独立于 in-loop 4096）：** v0 基座、两臂终点各跑
MATH500 500×4、`max_new_tokens=8192`，报 lenient/strict 与**截断拆分**（截断率
骤降后未截断 acc 是否仍 v40/训练臂更高——用于区分能力 vs 被 4096 截断掩盖）。

> 卡时提示：按实测生成 60–90min/窗 + 训练 23min/窗 + DCP~50s，单臂 20 窗约
> 28–38 GPU 小时；两臂 + 评测点（5×500×4 flash 约 1h/次）合计约 70–90 GPU 小时。


---

## 4. 判胜条件（提前写死，跑完不改阈值）

主指标：MATH500 500×4 **lenient 正确率**，比较两臂在 **v20 / v30 / v40** 三个
点（早期 v10 噪声大只看不判）。记 DAPO−GRPO 的绝对差 Δ（百分点）。

**评测噪声估计（阈值依据）：**
- 500×4 的逐题平均，每题 4 个伯努利样本。单题正确率的标准误最大在 p=0.5：
  SE_单题 ≈ √(0.25/4)=0.25；500 题平均后均值 SE ≈ √(p(1−p)/(4·500))，
  在 p≈0.3–0.5（小模型 MATH 区间）取 ≈ **0.009–0.011，即约 0.9–1.1 个百分点**
  （这是同一策略重采样的噪声下界）。
- 两次独立 run 的 run 间噪声更大。GSM8K 既有曲线在上升段出现过 0.775→0.76→
  0.795→0.835 的 ±1.5–2 点来回（200×4）；500×4 样本量是其 2.5×，单点波动按
  **σ≈1.5 个百分点**保守取值（含 prompt 难度抽样与训练随机性）。
- 两臂配对（同 seed、同 prompt 序列）做**配对差**可消掉题目难度方差，Δ 的 SE
  用配对二项/自助法估，按 Δ_SE ≈ **1.0–1.3 点**取。

**判据（在 v20/v30/v40 三个点上分别算，以 v40 终判，v20/v30 看趋势一致性）：**

- **DAPO 赢**：v40 处 Δ ≥ +3.0 个百分点（≈ ≥2.3× Δ_SE），**且** v30→v40 不回落
  （v40 ≥ v30 − 1 点），**且** v20/v30 至少一点 Δ ≥ +1.5（方向一致，不是终点
  偶然跳变）。同时报告成本：若 Δ 达标但 DAPO 多耗 >50% GPU 小时，结论记为
  "赢但不划算"，分开写。
- **平**：|Δ(v40)| < 3.0 个百分点，或方向在 v20/v30/v40 反复变号。结论：DAPO
  两件套（动态补样 + overlong；优势两臂相同）在 MATH 40 窗上无可测增益，不默认开启。
- **DAPO 输**：v40 处 Δ ≤ −3.0 个百分点（且非单点噪声：v30 同向）。需排查是否
  overlong 惩罚误伤正确长答案（看 truncation/format rate 与 raw_reward 曲线）。

**辅助看门指标（不参与主判，但出现异常要先解释再下结论）：**
- boxed format rate 两臂应接近且随训练上升；若 DAPO 明显低，说明 overlong 在
  惩罚答案写完。
- truncation rate：DAPO 应 ≤ GRPO（overlong 的预期效果）；反之异常。
- raw_reward 解題率 vs shaped reward：DAPO 用 raw 列判主指标（判分不经惩罚），
  shaped 只用于训练，避免惩罚本身压低指标。
- groups_filtered 与补样倍数：报告 DAPO 实际触发动态采样的比例；若 40 窗内
  filter 率极低（如 <5%），说明动态采样基本没起作用，此时"平/赢"都不能归因于
  动态采样。
- GSM8K 那条曲线的回落提示：单点不足为凭，所以坚持三点趋势 + 配对，不看单点。

**复跑原则**：若 v40 落在 ±3 点的"不确定带"内（最可能的结果），不追加同一
40 窗，而是按预先规则判定为"平"；只有要区分 +2 与 +4 这种贴边情形时，才用更
大评测样本（1000×8）做一次**两臂终点评测**，仍不延长训练。

---

## 5. 执行顺序（给 rl 的 checklist）

1. 确认 `v100/math` 合入，**并把 env T11d（commit `8ec7b61`，window 统计与
   TB 指标）合进运行分支**；v100 上 math_verify 0.9.0 已装；
   `/data00/meshy/models/hendrycks_math_train` 与 `MATH-500` 快照在。
   - **flash-decode（T3h，两臂都开 `MESHY_SM70_FLASH_DECODE=1`，已在
     `v100_run_rl.sh`）**：首次 decode 会用 CUDA 12.4 nvcc JIT 编译
     `meshy/kernels/csrc/flash_decode.cu`，约 1 分钟，第一窗启动慢属正常、
     不是卡死（PATH 必须含 `/usr/local/cuda-12.4/bin`）。第一窗起服后，
     在 scheduler 日志确认出现 `[sglang-sm70] flash-decode plan ...` 且
     `scripts/sm70_flash_decode_e2e.py --only consistency` 式的
     `sentinel_flash_called=true` 验收（kernel 真执行；只 exact_match=true
     而 flash_called=false 是空通过，说明 guard/路径没生效，必须停）。
   - colocate release/resume 已验证 flash 仅多 ~36MB 常驻、两轮 resume 200
     无 OOM；若 resume 失败，看 `torch_memory_saver`/`csrc core.cpp` 行而非
     仅看退出码。
2. 写 `recipe/v100_math_common.py` + 两份薄 recipe（§1 diff 表），受控量
   写死两臂一致；窗数 N 与 ROLLOUT_BATCH 从 env 读但默认 `N=40`、
   `XRL_ROLLOUT_BATCH=64`（recipe 内默认值也设 64，不依赖 shell 记忆）。
3. **先不跑 GPU**，用 CPU 单测断言：两份配置 dump 出来，除 §1.2 五行外
   deep-equal；seed 相同下第 k 窗锚 prompt id 两臂一致（复用 base.Dataset 的
   shuffle(seed=42) 验证，不依赖 T7，因为本次 start_window=0）。
4. T7 完成前，**两臂都只允许冷启动 start_window=0**；recipe 检测到
   `XRL_START_WINDOW!=0` 或存在本 run 的 DCP 时直接报错退出（防止误用热启动
   带重放偏差）。
5. **【第一硬门 · 占卡】先做 §3.0 基座 MATH500 500×4**（15–30 分钟，排
   kern/rl 之后、不自行 hold）。lenient pass@1 ≥15% 才继续；<15% 停在这里，
   把分数与替代方案（MATH level1–3 子集 / 改 GSM8K 对照）报主控裁决，**不进入
   后续步骤**。结论落 awb news。
6. **【第二硬门】基座过线后跑 §3.2 的 DAPO 2 窗计量**（独立 smoke tag，
   RL_STEPS=2，不开启评测），读出 R / filtered_ratio / 每窗生成与训练墙钟，
   按 §3.2 表当场定 N（40 或 20）与是否降 `XRL_DYNAMIC_MAX_PROMPTS`；结论落 awb news。
   冒烟窗必须同时报（max_new=4096 不改，数据给用户定）：
   - **训练侧截断率**（被 stamp `truncated` 的样本占比，区分评测截断）；
   - **截断样本的 raw reward 分布**（截断几乎必为 0：评测测得截断样本 acc
     base .009 / v40 .039）；
   - **截断样本的重复退化比例**（`repetition` flag 命中）；
   - **DAPO 两窗各自的 R**（`prompts_drawn/64`，取高窗对照阈值）；
   - **|corr(raw_reward, shaped−raw)|**：overlong 惩罚与原始正确性的相关，
     确认塑形作用在“长但错”而非“长但对”。
   **这一步未出结果，禁止进入第 7 步。**
7. 按 GPU 队列申请（不自行 hold）；先跑 GRPO 臂还是 DAPO 臂由队列决定，但
   两臂用**同一张卡**（避免跨卡硬件差），run tag 分别 `math-grpo-{N}w` /
   `math-dapo-{N}w`，两者 N 相同。
8. 每臂结束收集：5 个评测点 JSON（lenient/strict/format/trunc/pass@k）、
   逐窗 `rollout_window_stats.jsonl`（prompts_drawn/groups_dropped/
   filtered_ratio）+ 生成/训练分段墙钟/GPU 秒、checkpoint。
9. 用统一脚本出三对齐轴曲线 + §4 判据，结论落一张对照表（含 DAPO 实际补样
   倍率与每 GPU 小时解题率）。

### 5.1 本地盘容量与 DCP 中途清理（3FS 已退役，2026-09-30 用户定）

存储根已从 `/3fs/stage/meshy` 迁到本地 **`/data00/meshy/store`**（`XRL_STORAGE_ROOT`，
v100_run_rl.sh 显式 export，并在启动前 mkdir + 可写检查）。trajectories 走
`XRL_RUNTIME_DIR`（同根 `rollout/`），不再有"只设 storage root、轨迹仍落 3FS"的缝。

- **DCP 实测 7.25 GiB/份**（fp32 master + fp16 + Adam exp_avg/exp_avg_sq + LR +
  train state），不是旧估的 5.3 GB。`XRL_DCP_CKPT_INTERVAL=3`：40 窗落
  step 3,6,…,39 共 13 份加末尾 step40。
- 末尾份 `last_save_model_only=False`（recipe 已设），故 step40 也含优化器/LR/
  train state，可直接 resume；两臂仅末尾份改全量合计多约 8.8 GB。
- **容量**：3FS 旧 engine 数据（data-s1/s2/fdb）已删，/data00 实测可用
  **244 GB**（492 GB 盘，2026-09-30）。两臂里程碑 DCP 全留（interval=3 每臂
  step 3…39 共 13 份加末尾 step40 ≈ 14 份 × 7.25 GiB，两臂 ≈ **190 GB**）现在
  放得下；仍按 `checkpoint_keep=2` 稳态滚动 + 中途清理执行（与 weight-retention
  一致，删前先报分数表），不占满盘。稳态每臂 ~14.5 GB，写下一份瞬时 +7.25 GiB。
- DCP 保留/清理口径（与 weight-retention 一致，删前先报分数表）：
  - 默认 `checkpoint_keep=2` 自动滚动：每臂任一时刻最近 2 份可 resume DCP，
    评测点导出的 HF 落到 `best`/`evalckpt`。
  - 244 GB 下里程碑全留（~190 GB）放得下，允许在评测点额外保留里程碑 DCP；
    是否全留由主控定，不需要为腾盘强制删除。若盘占用逼近巡检阈值，再按
    “跨评测点后删上一里程碑”回收。
  - 巡检用 `deploy/3fs-v100/63_storage_watch.py`（默认根已改本地；
    `XRL_WINDOW_GB=0.3` 只盯异常增长），余量 < 一窗 exit 1。

### 5.2 成败判定：以 ignitor 日志为准，不以 `RL_FAIL`/退出码为准

冒烟实测：训练全部 step 正常完成（DCP/HF 都落盘）后，torchtitan elastic 在
**正常收尾终止服务**时会给 train rank 发 SIGKILL(-9)，`launch.py` 据此报
`recipe FAILED`、`v100_run_rl.sh` 打印 `RL_FAIL rc=1`（其实是成功）。判定一次 run
成败**只看 ignitor 日志** `Run completed cleanly; terminating services` +
`.metadata`/weights 是否落盘；`RL_DONE`/`RL_FAIL` 与退出码不可靠。

已核查：没有任何自动重跑/误触发链路依赖该退出码（crontab 空、无 supervisor 拉起
RL）；消费 `RL_FAIL` 的只是人工一次性 watchdog（`run_watch2/3.sh`、
`clean40_watch_remote.sh`、`ckpt_grab.sh`，不在仓库、只决定监视者退出码，不会
重跑训练）。收尾信号处理（让正常完成退出 0）排到 T11 后修，不阻塞本次。

---

## 6. 前置依赖：T7（热启动题目重放修复）

T7（owner=env，state=wip，awb id 150822）修的是：热启动续跑后数据游标回到
window 0，clean40b 实测 2634 次 prompt 使用只覆盖 896 道题（12%）。

- 本执行单**主动规避**了该依赖：两臂都从 window 0 冷启动（§5 第 4 步硬性
  禁止热启动/DCP resume），所以 T7 未合也能跑出**干净的 40 窗对照**。
- 但要写明：**这套 40 窗结果不能用于支持任何"续跑到 >40 窗 / 多阶段训练"的
  结论**；一旦 T11 后续要拉长或热启动接力，必须先合 T7（并补
  `XRL_START_WINDOW` 的非重叠 CPU 单测），否则数据覆盖偏差会复现。
- rl 启动前向 env 确认 T7 是否在本 run 时间窗内会 merge；若会 merge，本次冷启动
  两臂也不受影响（不同 run tag），无需等待。

---

## 7. 关键文件 / 命令速查

- 数据集/判分：`meshy/dataset/hendrycks_math.py`（HendrycksMATH / math_equiv /
  tuple 防护 / 空 gold 过滤）
- 优势函数（**两臂相同**，不去 std）：`meshy/worker/rollout.py:grpo_advantage`
  （组内 `(r−mean)/std`）。不用 `meshy/advantage.py:_2_6_math_reshaped_advantage`
  （Dr.GRPO 去 std 路线，且自身再做 overlong 会与 reward shaping 双重计数）。
- DAPO overlong：`meshy/reward.py:dapo_overlong_penalty`（软区 3072–4096，
  truncated 硬罚；kwargs `max_response_len`/`cache_len`），由
  `XRL_OVERLONG_SHAPING=1` 在 recipe 挂上 reward_shaping。
- 动态采样：`RolloutServiceConfig.dynamic_sampling` + 硬上限 `dynamic_max_prompts`
  （env `XRL_DYNAMIC_MAX_PROMPTS`，默认 192；`oversample_factor` 在 dynamic 路径
  不生效，预算见 rollout.py 的 `replacement_budget=dynamic_max_prompts-target`）；
  rollout 逻辑在 `meshy/worker/rollout.py`（is_zero_variance_group /
  _run_dynamic_rollouts）；每窗指标（env T11d `8ec7b61`）写
  `<runtime>/rollout_window_stats.jsonl`，字段
  `prompts_drawn/valid_groups/valid_samples/groups_dropped_zero_variance/
  refill_count/filtered_ratio`，并转训练日志 + TensorBoard。
- 单卡骨架：`recipe/grpo_gsm8k_v100.py`
- 评测 hook：`recipe/v100_inloop.py`（`XRL_EVAL_DATASET=math500` 已支持）
- 端点评测：`scripts/eval_math.py --data math500 --samples 4`（8192 备用；
  in-loop 用 4096）

**启动命令（公共前缀，注意 XRL_ROLLOUT_BATCH=64 必须显式带，recipe 文件默认是 8）：**

```bash
export PATH=/usr/local/cuda-12.4/bin:$PATH HF_ENDPOINT=https://hf-mirror.com
export XRL_MODEL=/data00/meshy/models/Qwen3-0.6B
export XRL_ROLLOUT_BATCH=64 XRL_GROUP_SIZE=8   # 512 样本/窗；不靠默认
# flash-decode 两臂都开（v100_run_rl.sh 已 export，命令行直起也显式带上）：
export MESHY_SGLANG_SM70=1 MESHY_SM70_FLASH_DECODE=1
export XRL_DCP_CKPT_INTERVAL=3
```

第 5 步 gating smoke（只 DAPO 配方，2 窗，无评测）。树里只有一份 `recipe/grpo_math_v100.py`，
DAPO 三件套由 `XRL_DYNAMIC_SAMPLING=1 XRL_OVERLONG_SHAPING=1` 开启（GRPO 臂两者留 0）：

```bash
XRL_STEPS=2 XRL_EVAL_EVERY=99 XRL_RUN_TAG=math-dapo-smoke \
  XRL_DYNAMIC_SAMPLING=1 XRL_OVERLONG_SHAPING=1 \
  python scripts/launch.py --recipe recipe.grpo_math_v100
# 然后读 $(XRL_RUNTIME_DIR 或 /data00/meshy/store/rollout/math-dapo-smoke)/rollout_window_stats.jsonl
# 取 kind=dynamic 的 prompts_drawn/filtered_ratio，配合 timer 日志的生成/训练墙钟，按 §3.2 表定 N
```

正式两臂（N 用 gating 结论，40 或 20；正式启动走 v100_run_rl.sh + XRL_RECIPE=grpo_math_v100，
下面直接列 launch.py 仅示意自变量）：

```bash
XRL_STEPS=40 XRL_RUN_TAG=math-grpo-40w \
  python scripts/launch.py --recipe recipe.grpo_math_v100
XRL_STEPS=40 XRL_RUN_TAG=math-dapo-40w \
  XRL_DYNAMIC_SAMPLING=1 XRL_OVERLONG_SHAPING=1 \
  python scripts/launch.py --recipe recipe.grpo_math_v100
# 降 20 窗时四条 XRL_STEPS=40 改 20、run tag 改 -20w，评测 XRL_EVAL_N 与评测点两臂同步改
```
