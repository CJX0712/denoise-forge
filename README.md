# Denoise Forge — 图像扩散模型实验室（forge 系列 #15）

<p align="center">
  <a href="https://github.com/CJX0712/denoise-forge/actions/workflows/ci.yml"><img src="https://github.com/CJX0712/denoise-forge/actions/workflows/ci.yml/badge.svg" alt="ci"></a>
  <a href="https://github.com/CJX0712/denoise-forge/releases"><img src="https://img.shields.io/github/v/release/CJX0712/denoise-forge?sort=semver" alt="release"></a>
  <a href="https://github.com/CJX0712/denoise-forge/blob/main/LICENSE"><img src="https://img.shields.io/github/license/CJX0712/denoise-forge" alt="license"></a>
  <img src="https://img.shields.io/badge/author-%E6%99%A8%E6%98%9F-1f6feb" alt="author">
</p>

**零依赖 · 单文件 HTML · 引擎可无头自检的图像 DDPM。**

从纯噪声到形状：浏览器内训练一个**真正的 U-Net 去噪网络**（卷积 + 下/上采样 + skip 连接 + FiLM 时间条件），实时观看 `x_T → x_0` 的完整去噪过程。引擎 `IDF.*` 无任何 DOM 依赖，可在 Node `vm` 中完整验证。打开 `index.html` 即用，无需构建、无需网络。

> 与姊妹项目 `ddpm-forge`（2D 点数据 + 3 层 MLP）不同，**本项目是图像版**：U-Net 卷积架构，生成 16×16 灰度形状。

![license](https://img.shields.io/badge/license-MIT-green) ![deps](https://img.shields.io/badge/dependencies-0-brightgreen) ![selftest](https://img.shields.io/badge/selftest-15%2F15%20%2B%2017%2F17-blue)

## 快速开始

```bash
# 浏览器：直接打开 index.html → 开始训练 → 生成样本 / 看去噪动画

# 无头自检（Node ≥ 18，零依赖）：
node _smoke.js      # 15 项引擎不变量 → _smoke.log
node _uicheck.js    # DOM stub 点遍所有控件 → _uicheck.log
node _probe.js 900  # 训练 900 步并把生成样本打成 ASCII → _probe.txt（需 ~2 分钟）
```

## 架构

```
x_t (1×16×16) ─┬─→ [enc1  conv 1→C @16]  →FiLM→ReLU→ [conv C→C] →FiLM→ReLU ──┬─→ skip1
               │                     ↓ avgpool 2×2                              │
               │   [enc2  conv C→2C @8] →FiLM→ReLU→ [conv 2C→2C] →FiLM→ReLU ─┬┤→ skip2
               │                     ↓ avgpool 2×2                             ││
               │   [bottleneck conv 2C→2C @4] ×2                               ││
               │                     ↑ nearest ×2                              ││
               │   [dec2  up→cat(skip2)→conv 4C→2C→conv 2C→2C]  ←─────────────┘│
               │                     ↑ nearest ×2                               │
               └─→ [dec1  up→cat(skip1)→conv 3C→C→conv C→C] ←──────────────────┘
                              ↓ conv C→1
                        ε̂ = conv_out + ls(t)·x_t      ← 零初始化 t 相关线性 skip
```

- **时间条件**：正弦嵌入 `te(t)` → MLP → `tvec` → 每个卷积后一层 **FiLM**（`h·(1+scale)+shift`）。FiLM 权重零初始化，训练起点即恒等映射。
- **零初始化线性 skip**：`ε̂` 对 `x_t` 近似线性（`ε̂ ≈ (x_t − √ᾱ·x̂₀)/√(1−ᾱ)`），且系数依赖 t，故用 `ls(t) = w·tvec + b`（零初始化）直通，显著加速收敛。
- **采样**：ancestral（DDPM）与 **DDIM**（η=0 时完全确定性，20 步即可出图）。

## 引擎 API（`<script id="engine">`，全局 `IDF`）

| 模块 | API | 说明 |
|---|---|---|
| 调度 | `makeSchedule(T,β₁,β_T)` | 线性 β，`ᾱ_t=Π(1−β_s)` 严格单调递减 |
| 前向 | `qSample / postVar` | 闭式 `q(x_t|x₀)=N(√ᾱ·x₀,(1−ᾱ)I)`；`t=0` 后验方差为 0 |
| 原语 | `convF/B · poolF/B · upF/B · filmF/B · linF/B · reluF/B · catF/B` | 全部手写，逐个经 FD 验证 |
| 网络 | `createNet / forwardNet / backwardNet / zeroGrads` | U-Net，全参数解析反向 |
| 时间 | `tEmbed` | 正弦位置编码，`|te|≤1` |
| 训练 | `lossBatch / makeAdam / adamStep / train` | ε-MSE；Adam 手写 |
| 采样 | `sample( ancestral, 可录轨迹 ) / sampleDDIM` | 后验均值 + 方差；DDIM 支持子采样步数 |
| 数据 | `datasetShapes(n,S,seed,jitter)` | 5 类合成形状（圆/方/环/十字/三角），零均值单位方差 |
| RNG | `mulberry32 / gaussFactory` | 全程种子可复现（逐位确定性） |

## 可验证不变量（`_smoke.js` 15/15）

| # | 不变量 | 实测 |
|---|---|---|
| 1 | `ᾱ_t ∈(0,1)` 严格单调递减 | 0.999900 → 0.363563 ✓ |
| 2 | 前向矩 = 闭式解 `N(√ᾱ·x₀,1−ᾱ)` | 0.5070 vs 0.4974（4σ=0.0351）✓ |
| 3 | 高斯封闭性 `x₀~N(0,1) ⇒ q(x_t)~N(0,1)` | var=0.9967 ✓ |
| 4 | 后验方差 `t=0` 为 0，`t≥1` 为正 | ✓ |
| 5 | 原语反向 FD（conv/pool/up/film） | maxRel=**1.35e-9** ✓ |
| 6 | **全网络梯度检验 vs 有限差分** | maxRel=**1.84e-6**（有效样本 204/218）✓ |
| 7 | 前向确定性 | 逐位一致 ✓ |
| 8 | 训练 loss 下降 | 1.0240 → 0.6764 ✓ |
| 9 | 训练确定性（同 seed 逐位一致） | ✓ |
| 10 | Ancestral 采样有限且方差有界 | E[x²]=0.714 ✓ |
| 11 | 采样轨迹帧数 = T | 100 ✓ |
| 12 | DDIM(η=0) 确定性且有限 | ✓ |
| 13 | DDIM 5 步采样有限 | ✓ |
| 14 | 数据集零均值单位方差 | mean=4.1e-16, var=1.000000 ✓ |
| 15 | 时间嵌入有界 `|te|≤1` | ✓ |

`_uicheck.js` 另有 17 项（含 UI 自检 8/8）全部通过。

## 实测生成质量（`_probe.js`，900 步）

| 指标 | 数值 |
|---|---|
| loss | 1.3665 → **0.0676**（前20均 0.9079 → 末20均 0.0919） |
| 数据 E[x²] vs 生成 E[x²] | 1.000 vs **0.810**（ancestral）/ 0.768（DDIM） |
| 模式覆盖（最近邻归类） | 3/5 类（ancestral 与 DDIM 均 3/5） |
| 相邻像素差均值 | 真实 0.2186 vs 生成 0.2352 —— **同量级，说明学到的是边缘结构而非纯噪声** |

生成样本已能看出清晰的十字、圆、三角轮廓。想更干净可把步数加到 2000+。

## 踩过的坑（都是真金白银）

1. **ReLU 网络的梯度检验，EPS 必须用 1e-6，不是 1e-4。**
   用 1e-4 时误差高达 `2.2e-1`，看上去像反向传播写错了；实际上 EPS 太大会**翻转大量预激活值接近 0 的单元**，FD 量到的是跨折点的割线。
   判别法（扫 EPS）：误差随 EPS 减小而单调下降 → 伪影；基本不变 → 真 bug。实测 1e-3→1.0，1e-4→2.2e-1，1e-5→8.6e-2，**1e-6→3.7e-6**，1e-7→5.7e-5（浮点相消回升）。

2. **梯度过小（<1e-4·max|grad|）的参数拿 FD 测没有意义。** EPS=1e-6 时 FD 的绝对噪声约 1e-10，梯度只有 4e-7 时相对误差必然 ~1e-4。按量级设地板跳过，而不是放宽阈值掩盖问题。

3. **检验脚本的 `1/N` 缩放最容易写错。** 把 4 个配置当"批"归一成 `2e/(4P)`，而 FD 是各自 `2e/P` 求和 → 解析值正好是数值的 1/4，maxRel 整齐等于 **6.0e-1**。看到这种"过于干净的比例"先查缩放，别急着改反向传播。

4. **pool 的反向要把梯度写回全分辨率数组。** 第二层下采样（`s2(S/2) → p2(S/4)`）的 `poolB` 输出必须分配 `(C,S,S)`；分小了越界写入会被 Float64Array 静默丢弃，梯度直接消失。

5. **缓存键约定要前后一致。** 前向用扁平键 `c["fi_"+nm]`，后向写成嵌套 `c.fi_[nm]` 会直接 `undefined` 崩溃。

6. **数据分布复杂度决定能不能学好。** 形状位置/尺度全随机时 300 步只出噪声块（E[x²]=0.43）；改成居中+小抖动后 900 步就出清晰结构（E[x²]=0.81）。

7. **ε 必须配对。** 构造 `x_t` 用的 ε 就是预测目标的 ε，否则 `E[ε|x_t]=0`，网络"正确地"学成全零：loss 卡在 1.0 且梯度检验全绿（反向传播没错，是数据造错），极具迷惑性。

## 文件

```
index.html      单文件交付物（引擎 + UI 全部内联，零外部依赖）
_smoke.js       引擎不变量自检（15 项）
_uicheck.js     DOM stub 冒烟，点遍全部控件（17 项）
_probe.js       训练 + ASCII 出图，人工读数
README.md  LICENSE  .gitignore
```

## License

MIT © 晨星
