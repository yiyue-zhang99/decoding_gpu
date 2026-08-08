# TW 与 Cross-Generalization Matrix 关系的分析思路

## 研究问题

核心问题是：

> Trial-by-trial 的 travelling-wave（TW）强度，是否与同一 trial 的跨时间泛化表征强度有关？这种关系是否能够在群体水平上稳定复现？

Cross-generalization 结果包含两个时间维度：

\[
G(t_{\mathrm{train}}, t_{\mathrm{test}})
\]

TW 则是一个时间序列：

\[
TW(t_{\mathrm{TW}})
\]

因此，完整分析涉及三个时间维度：

\[
t_{\mathrm{TW}} \times t_{\mathrm{train}} \times t_{\mathrm{test}}
\]

---

# 一、首选方案：先做被试内 trial-level 分析，再做群体水平检验

## 1. 所需数据

对于每个被试 \(s\)，需要保留每个 trial 的 TW 指标：

\[
TW_{s,k}(t_{\mathrm{TW}})
\]

其中：

- \(s\)：被试；
- \(k\)：trial；
- \(t_{\mathrm{TW}}\)：TW 的时间点。

同时，需要获得每个 trial 的 cross-generalization decoding evidence：

\[
E_{s,k}(t_{\mathrm{train}}, t_{\mathrm{test}})
\]

这里的 \(E\) 不能只是整个被试的平均 decoding accuracy，而应当是每个 held-out test trial 的连续 decoding evidence，例如：

- 正确类别的 decision value；
- 正确类别相对于其他类别的 evidence；
- circular decoding fidelity；
- 对真实角度的 cosine-weighted evidence；
- single-trial Mahalanobis 或 reconstruction evidence。

最重要的是，每个 trial 的 evidence 必须来自该 trial 没有参与训练的模型，也就是 out-of-fold prediction。

---

## 2. 如何处理 8-fold、重复 800 次的 decoding

在每一次完整的 8-fold cross-validation 中，每个 trial 都会作为测试 trial 得到一次 out-of-fold prediction。

重复 800 次后，同一个真实 trial 会得到约 800 个预测结果。首先应在同一个 trial 内对这些重复结果取平均：

\[
\bar E_{s,k}(t_{\mathrm{train}}, t_{\mathrm{test}})
=
\frac{1}{800}
\sum_{r=1}^{800}
E_{s,k,r}(t_{\mathrm{train}}, t_{\mathrm{test}})
\]

其中 \(r\) 表示 cross-validation 的重复次数。

重复 800 次的作用是提高 trial-level decoding evidence 的稳定性，并不意味着样本量扩大了 800 倍。后续分析中的独立观测仍然是原始 trial，而不是“trial × 800 次重复”。

---

## 3. 每个被试内部的三维相关

对于每一个固定的时间组合：

\[
(t_{\mathrm{TW}}, t_{\mathrm{train}}, t_{\mathrm{test}})
\]

在该被试的 trials 之间计算 TW 与 decoding evidence 的关系：

\[
r_s(t_{\mathrm{TW}}, t_{\mathrm{train}}, t_{\mathrm{test}})
=
\operatorname{corr}_{k}
\left[
TW_{s,k}(t_{\mathrm{TW}}),
\bar E_{s,k}(t_{\mathrm{train}}, t_{\mathrm{test}})
\right]
\]

也就是说，在某个 TW 时间点和某个 train–test decoding 单元格上：

- 每个 trial 有一个 TW 值；
- 同一个 trial 有一个 decoding evidence；
- 在 trials 之间计算二者的相关。

对所有时间组合重复后，每个被试得到一个三维相关矩阵：

\[
r_s
\in
\mathbb{R}^{
N_{\mathrm{TW}}
\times
N_{\mathrm{train}}
\times
N_{\mathrm{test}}
}
\]

该三维矩阵中的一个格子表示：

> 在该被试内部，某个时间的 TW trial-by-trial 波动，是否与某个 train–test 时间组合下的 decoding evidence 共同变化。

---

## 4. 更推荐使用 trial-level 回归

相比简单相关，可以在每个被试内部进行回归：

\[
E_{s,k}(t_{\mathrm{train}}, t_{\mathrm{test}})
=
\beta_{0,s}
+
\beta_{\mathrm{TW},s}
TW_{s,k}(t_{\mathrm{TW}})
+
\beta_{\mathrm{condition},s}
Condition_{s,k}
+
\beta_{\mathrm{RT},s}
RT_{s,k}
+
\cdots
+
\varepsilon_{s,k}
\]

这样可以控制 trial-level 混淆变量，例如：

- 实验条件；
- 刺激类别或角度；
- reaction time；
- trial number 或 block；
- 正确与错误反应；
- EEG power 或整体信号幅度；
- 眼动、肌电或其他噪声指标。

每个被试最终得到：

\[
\beta_{\mathrm{TW},s}
(t_{\mathrm{TW}}, t_{\mathrm{train}}, t_{\mathrm{test}})
\]

它表示：

> 在控制其他 trial-level 因素之后，TW 是否仍然预测该 trial 的 decoding evidence。

---

## 5. 群体水平检验

每个被试最终贡献一张三维效应图，而不是把所有被试的 trials 合并在一起。

如果第一层使用相关系数，应先进行 Fisher \(z\) 转换：

\[
z_s = \operatorname{atanh}(r_s)
\]

然后在被试层面检验：

\[
H_0:
E[z_s(t_{\mathrm{TW}}, t_{\mathrm{train}}, t_{\mathrm{test}})] = 0
\]

如果第一层使用回归，则检验：

\[
H_0:
E[\beta_{\mathrm{TW},s}
(t_{\mathrm{TW}}, t_{\mathrm{train}}, t_{\mathrm{test}})] = 0
\]

由于存在大量时间组合，需要使用三维 cluster-based permutation test 进行多重比较校正。

置换应在被试层面进行。例如，对每个被试的整张三维效应图进行统一的正负号翻转。同一次置换中，同一个被试的整张图必须使用同一个符号，以保留不同时间点之间的相关结构。

---

## 6. 结果如何解释

假设发现一个显著三维 cluster：

- TW 时间：400–600 ms；
- decoding training 时间：200–400 ms；
- decoding testing 时间：800–1100 ms。

可以解释为：

> 在个体被试内部，400–600 ms TW 较强的 trials，倾向于表现出更强的“早期训练表征向晚期测试阶段泛化”的 decoding evidence；而且这种 trial-level 关系在被试群体中稳定存在。

这一结果仍然表示统计关联，不能单独证明 TW 导致了后续表征泛化。

---

# 二、如果每个被试只有一张 cross-generalization matrix

假设每个被试只有：

\[
G_s(t_{\mathrm{train}}, t_{\mathrm{test}})
\]

即所有 trials 汇总后得到的一张 cross-generalization matrix，同时每个被试只有一条平均 TW 时间序列：

\[
TW_s(t_{\mathrm{TW}})
\]

这种情况下不能做 trial-by-trial correlation，因为 trial 身份已经在 decoding matrix 中被平均掉了。

此时只能研究被试间关系。

---

## 方案 A：完整的跨被试三维相关

对于每一个固定时间组合：

\[
(t_{\mathrm{TW}}, t_{\mathrm{train}}, t_{\mathrm{test}})
\]

取所有被试的数据：

\[
TW_s(t_{\mathrm{TW}})
\]

以及：

\[
G_s(t_{\mathrm{train}}, t_{\mathrm{test}})
\]

然后在被试之间计算相关：

\[
r(t_{\mathrm{TW}}, t_{\mathrm{train}}, t_{\mathrm{test}})
=
\operatorname{corr}_{s}
\left[
TW_s(t_{\mathrm{TW}}),
G_s(t_{\mathrm{train}}, t_{\mathrm{test}})
\right]
\]

这样得到一个三维相关矩阵：

\[
N_{\mathrm{TW}}
\times
N_{\mathrm{train}}
\times
N_{\mathrm{test}}
\]

它回答的是：

> 在某个 TW 时间点，TW 较强的被试，是否也在某个 train–test 时间组合上表现出更强的 cross-generalization？

这里的观测单位是被试，而不是 trial。

显著性检验应通过打乱被试对应关系完成。例如，保持 TW 数据不变，随机打乱不同被试的 cross-generalization matrix，再重新计算完整三维相关图，并进行三维 cluster correction。

---

## 方案 B：预先定义 training window，将问题降为二维

完整三维分析计算量较大，也较难解释。更可行的方法是根据独立理论预先定义一个 training window。

例如，关注早期表征形成阶段：

\[
t_{\mathrm{train}} \in T_{\mathrm{early}}
\]

先对 training 维度取平均：

\[
G_s^{\mathrm{early-train}}(t_{\mathrm{test}})
=
\operatorname{mean}_{t_{\mathrm{train}}\in T_{\mathrm{early}}}
G_s(t_{\mathrm{train}}, t_{\mathrm{test}})
\]

然后计算：

\[
r(t_{\mathrm{TW}}, t_{\mathrm{test}})
=
\operatorname{corr}_{s}
\left[
TW_s(t_{\mathrm{TW}}),
G_s^{\mathrm{early-train}}(t_{\mathrm{test}})
\right]
\]

最终得到二维相关矩阵：

\[
TW\ time \times decoding\ test\ time
\]

再使用被试标签置换和二维 cluster test。

这一分析更容易回答：

> 哪个时间段的 TW，与早期形成的神经表征在后续时间的泛化强度有关？

---

## 方案 C：提取预定义的 cross-generalization ROI

也可以预先定义一个 train–test 区域，例如：

\[
t_{\mathrm{train}} = 200\text{–}400\ \mathrm{ms}
\]

\[
t_{\mathrm{test}} = 800\text{–}1200\ \mathrm{ms}
\]

然后为每个被试计算一个平均泛化指标：

\[
G_s^{ROI}
=
\operatorname{mean}
G_s(
t_{\mathrm{train}}\in T_1,
t_{\mathrm{test}}\in T_2
)
\]

再与整条 TW 时间序列进行跨被试相关：

\[
r(t_{\mathrm{TW}})
=
\operatorname{corr}_{s}
\left[
TW_s(t_{\mathrm{TW}}),
G_s^{ROI}
\right]
\]

最终得到一条一维相关时间序列，可进行一维 cluster test。

这种方法统计效率更高、解释最直接，但 ROI 必须根据独立理论、先前研究或独立数据预先定义。不能先在同一批数据中寻找显著区域，再用该区域做相关，否则会产生 circular analysis。

---

# 三、两种分析回答的问题不同

## Trial-level 两层分析

先在每个被试内部分析 trials，再在被试间统计：

\[
TW_{\mathrm{trial}}
\longleftrightarrow
decoding\ evidence_{\mathrm{trial}}
\]

回答的是：

> 对同一个人而言，TW 更强的 trial 是否也具有更强的表征泛化？

这是一个被试内、trial-by-trial coupling 问题。

## 每个被试只有一张 matrix 的跨被试分析

直接在被试之间分析：

\[
TW_{\mathrm{subject}}
\longleftrightarrow
cross\text{-}generalization_{\mathrm{subject}}
\]

回答的是：

> TW 整体较强的被试，是否也表现出更强的跨时间泛化？

这是一个被试间个体差异问题。

两者不能相互替代，也可能得到不同结果。例如，被试内 trial-level 关系可能很稳定，但被试间平均水平未必相关；反过来也可能存在被试间相关，但每个被试内部没有 trial-level coupling。

---

# 四、建议的优先顺序

如果能重新获得每个 held-out trial 的连续 decoding evidence，首选：

\[
\boxed{
\text{每个 trial 的 out-of-fold evidence}
\rightarrow
\text{被试内 trial-level 回归}
\rightarrow
\text{每个被试一张效应图}
\rightarrow
\text{群体 cluster 检验}
}
\]

这最直接对应“TW 强的 trial 是否具有更强 cross-generalization”的假设。

如果现有数据只保留了每个被试的一张总体 cross-generalization matrix，则只能做跨被试相关。考虑到完整三维分析计算量大且较难解释，更推荐预先确定一个 training window 或 train–test ROI，将三维问题降低为二维或一维分析。
