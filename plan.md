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

### 待做
- [ ] 下载 Qwen3-4B + Qwen3-0.6B 完整权重
- [ ] 跑 Phase 1 AR 基线
- [ ] 跑 Phase 2 Dream-7B 端到端
- [ ] 解决 MDLM tokenizer 兼容性（Qwen2.5 vs Qwen3）
- [ ] 跑 Phase 3 MDLM 0.5B 端到端
- [ ] 接受率分析

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
