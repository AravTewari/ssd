# 扩散模型 Draft 实验计划

**开始时间**: 2026-03-31 ~23:00 UTC
**GPU**: 0 和 3 (H200 143GB)
**分支**: `xinyu/diffusion-draft-fixes` on `XinyuJiangCMU/ssd`

---

## 目标

回答：**扩散 draft 能不能在端到端 speculative decoding 吞吐上打赢 AR draft？**

假设：扩散 draft 一次出 K 个 token，双向注意力可能让 acceptance rate 更高。
即使每次调用更慢，如果接受率高很多，总吞吐可能反超。

---

## Phase 1: AR 基线（SSD 原版）

用标准 SSD sync spec decode + AR draft 跑基线。

**Target**: Qwen3-4B（SSD 只支持 Qwen3，不支持 Qwen2.5）
**AR Draft**: Qwen3-0.6B

注意：需要先下载 Qwen3-4B 和 Qwen3-0.6B 完整权重（目前只有 tokenizer）。

### 实验

```bash
# 1a. 纯 AR（无投机）— 吞吐下限
python -O bench.py --qwen --size 4 --b 1 --temp 0 --numseqs 64 --output_len 256 --random

# 1b. Sync spec decode — AR draft, 扫 k
python -O bench.py --qwen --size 4 --spec --draft 0.6 \
  --temp 0 --numseqs 64 --output_len 256 --random \
  --sweep '[{"k":2},{"k":4},{"k":6},{"k":8}]'
```

**记录指标**: 总吞吐(tok/s)、平均接受后缀长度、target/draft step 耗时

---

## Phase 2: 扩散 Draft — Dream-7B

同 target，用 Dream-7B 替代 AR draft。

```bash
python -O bench.py --qwen --size 4 --spec \
  --draft-backend llada_diffusion --draft $DRAFT_DREAM_7B \
  --temp 0 --numseqs 64 --output_len 256 --random \
  --sweep '[{"k":8,"dsteps":32},{"k":8,"dsteps":64},{"k":16,"dsteps":32}]'
```

跟 Phase 1 对比：吞吐、接受率、draft 耗时。

---

## Phase 3: 扩散 Draft — MDLM 0.5B（核心实验）

公平对比：同样 0.5B 参数，AR vs 扩散。

**Draft**: `dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1`

### 先要做的适配工作
- MDLM 用标准 HF attention（不需要 logits shift，不需要 "full" 字符串）
- mask_token_id = 151665（在 tokenizer 里，不在 config 里）
- 但 MDLM 是 Qwen2.5 tokenizer，和 Qwen3 target tokenizer 不一样！
  - 需要找 Qwen3 版本的 MDLM，或者用 Qwen2.5 target（但 SSD 不支持）
  - 或者自己用 dllm 框架把 Qwen3-0.6B 转成 MDLM
- 这是个阻塞项，需要先确认 tokenizer 兼容性

### 实验（适配完成后）

```bash
python -O bench.py --qwen --size 4 --spec \
  --draft-backend llada_diffusion --draft $DRAFT_MDLM \
  --sweep '[{"dsteps":8},{"dsteps":16},{"dsteps":32},{"dsteps":64}]'
```

---

## Phase 4: 接受率深入分析

对最佳配置，测量：
- 逐位置接受率（position 1 vs position K）
- 随 K 增大接受率的衰减速度（AR vs 扩散）
- 扩散在大 K 时是否保持更高接受率

---

## 提交策略

- 每次有显著吞吐提升就 commit + push
- 详细的 commit message，带 benchmark 数据
- 随时更新这个 plan.md 的结果

---

## 进展记录

### 2026-03-31 已完成
- [x] Dream-7B diffusion draft 在 SSD 引擎跑通 smoke test
- [x] 基础 throughput 对比数据（draft-only，不含 target verify）:
  - AR 0.5B: 128 tok/s, 7.8ms/tok
  - MDLM 0.5B: 60 tok/s, 16.5ms/tok（dsteps=32, k=16）
  - Dream 7B: 22 tok/s, 44ms/tok（dsteps=32, k=8）
- [x] 发现 MDLM 0.5B 模型（`dllm-hub/Qwen2.5-Coder-0.5B-MDLM`）
- [x] 代码推到 `XinyuJiangCMU/ssd` xinyu/diffusion-draft-fixes 分支
- [x] 下载 Qwen3-4B + Qwen3-0.6B 完整权重
- [x] Phase 1 + Phase 2 端到端 benchmark 完成

### Phase 1 + 2 端到端结果（Qwen3-4B target, H200 单卡, random prompts）

| 方案 | Draft | k | dsteps | 吞吐(tok/s) | 接受率 | 每步接受tok | draft耗时(ms) |
|------|-------|---|--------|------------|--------|------------|-------------|
| AR only | - | - | - | **239** | - | 1.0 | - |
| AR spec | Qwen3-0.6B | 2 | - | 198 | 71% | 2.42 | ~7.5 |
| AR spec | Qwen3-0.6B | 4 | - | 194 | 57% | 3.29 | ~12.1 |
| AR spec | Qwen3-0.6B | 6 | - | 195 | 47% | 3.79 | ~14.4 |
| AR spec | Qwen3-0.6B | 8 | - | 159 | 39% | 4.12 | ~20.6 |
| Diff spec | Dream-7B | 4 | 32 | **5.4** | 43% | 2.70 | 492 |

**关键发现**:
1. 对 4B target，**纯 AR 最快**（239 tok/s），spec decode 反而更慢
2. AR spec 接受率随 k 下降很快（71% → 39%）
3. Dream-7B 完全不可行 — draft 太大（7B > target 4B），每步 492ms
4. 扩散 draft 的接受率（43%@k=4）和 AR（57%@k=4）**差不多甚至更低**
5. 验证了理论：4B target 太小，verify 很快（4.6ms），draft 开销占主导

**结论**: 需要更大的 target（32B+）或更小的 diffusion draft 才能看到优势

### Qwen3-8B target 结果（H200 单卡）

| 方案 | Draft | k | dsteps | 吞吐(tok/s) | 接受率 | 每步tok | draft耗时(ms) | verify(ms) |
|------|-------|---|--------|------------|--------|--------|-------------|-----------|
| AR only | - | - | - | **173** | - | 1.0 | - | - |
| AR spec | 0.6B | 2 | - | 189 | 80% | 2.60 | ~7.5 | 6.2 |
| AR spec | 0.6B | 4 | - | **197** | 67% | 3.68 | ~12.1 | 6.3 |
| AR spec | 0.6B | 6 | - | 192 | 58% | 4.46 | ~16.4 | 6.2 |
| AR spec | 0.6B | 8 | - | 181 | 50% | 4.99 | ~20.7 | 6.2 |
| Diff spec | Dream-7B | 8 | 32 | **4.6** | **17%** | 2.36 | 488 | 6.3 |

**关键发现（8B target 新增）**:
1. 8B target 上 AR spec **终于比 AR only 快了**（197 vs 173），k=4 是最优
2. Dream-7B 更差了 — 接受率只有 **17%**！（4B 上是 43%）
3. Dream 和 Qwen3 的 tokenizer 虽然兼容但模型能力不匹配，导致接受率极低
4. verify 时间 6.2ms 还是太短，spec decode 优势不明显

**总结**: Dream-7B 作为 diffusion draft 不可行。模型太大、接受率太低。
下一步必须用同 tokenizer 的小 diffusion 模型（MDLM 0.5B），
或者用更大的 target（32B）让 verify 时间成为瓶颈。

### Phase 3 结果：MDLM 0.6B Diffusion Draft

完成了以下适配：
- 给 SSD 加了 Qwen2.5 model support（新文件 ssd/models/qwen2.py）
- 适配 diffusion adapter 支持 MDLM（无 logits shift，标准 HF mask）
- 从 tokenizer 自动检测 mask_token_id
- 下载了 Qwen3 版本的 MDLM（dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1）

| Target | Draft | k | dsteps | 吞吐 | 接受率 | 每步tok |
|--------|-------|---|--------|------|--------|--------|
| Qwen2.5-3B | MDLM-Coder-0.5B | 8 | 32 | 5.2 tok/s | 17% | 2.35 |
| Qwen3-4B | MDLM-Qwen3-0.6B | 8 | 32 | 3.8 tok/s | 17% | 2.35 |
| Qwen3-4B | MDLM-Qwen3-0.6B | 4 | 64 | 1.5 tok/s | 19% | 1.75 |

**核心发现: 接受率恒定 ~17%，和模型匹配度、dsteps 无关**

这意味着问题是**根本性的**：
1. 扩散模型的 bidirectional 预测和 AR target 的 left-to-right 生成天然不一致
2. 扩散模型对每个位置的预测是基于所有其他位置（包括右侧 mask），而 AR 只看左侧
3. 更多 denoising steps 不帮助 — 扩散模型收敛到的分布本身就和 AR 不同
4. 这不是模型大小或质量问题，是范式差异

**可能的后续方向**：
- 自投机解码（self-speculative）：用同一个扩散模型做 draft+verify
- 训练一个专门为投机解码优化的扩散 draft（知识蒸馏从 AR target）
- 用扩散模型的 logits 做 soft matching 而非 greedy matching

---

## 最终对比总结

### Qwen3-4B Target, H200 单卡

| 方案 | Draft 模型 | 参数量 | k | 吞吐(tok/s) | 接受率 | 每步tok | 比 AR-only |
|------|-----------|--------|---|------------|--------|--------|-----------|
| **AR only** | - | - | - | **239** | - | 1.0 | 1.00x |
| AR spec | Qwen3-0.6B | 0.6B | 4 | **197** | 67% | 3.29 | 0.82x |
| Diff spec | Dream-7B | 7.6B | 4 | 5.4 | 43% | 2.70 | 0.02x |
| Diff spec | MDLM-Qwen3-0.6B | 0.6B | 8 | 3.8 | 17% | 2.35 | 0.02x |
| Diff spec | MDLM-Qwen3-0.6B | 0.6B | 4 | 2.9 | 19% | 1.75 | 0.01x |

### Qwen3-8B Target, H200 单卡

| 方案 | Draft 模型 | k | 吞吐(tok/s) | 接受率 | 每步tok | 比 AR-only |
|------|-----------|---|------------|--------|--------|-----------|
| **AR only** | - | - | **173** | - | 1.0 | 1.00x |
| AR spec | Qwen3-0.6B | 4 | **197** | 67% | 3.68 | 1.14x |
| Diff spec | Dream-7B | 8 | 4.6 | 17% | 2.36 | 0.03x |

### 结论

1. **扩散 draft 在当前实现下不可行**：接受率恒定 ~17%，不受模型大小/质量/dsteps 影响
2. **根本原因**：扩散模型（双向）和 AR target（单向）的生成分布天然不同
3. **AR spec decode 在 8B target 上有效**：比纯 AR 快 14%（197 vs 173 tok/s）
4. **对于更大 target（32B+）**，AR spec 的优势会更大，但扩散 draft 的接受率问题不会改善

### 可能的后续方向
- 自投机解码（self-speculative decoding）：避开 AR/扩散分布不匹配
- 从 AR target 知识蒸馏扩散 draft，强制对齐分布
- 修改 verifier 支持 soft/approximate matching
- 在 token-level 之上做 semantic-level 验证

---

## 快速参考

```bash
cd /sgl-workspace/dgm/ssd && source .venv/bin/activate
export SSD_HF_CACHE=/root/.cache/huggingface/hub
export SSD_DATASET_DIR=/tmp/ssd_datasets
export CUDA_VISIBLE_DEVICES=0,3

DRAFT_DREAM_7B=/root/.cache/huggingface/hub/models--Dream-org--Dream-v0-Instruct-7B/snapshots/05334cb9faaf763692dcf9d8737c642be2b2a6ae
DRAFT_MDLM_05B=/root/.cache/huggingface/hub/models--dllm-hub--Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1/snapshots/a284e895a6248256baf2f60502e54aba61b24c1a

git push origin xinyu/diffusion-draft-fixes
```
