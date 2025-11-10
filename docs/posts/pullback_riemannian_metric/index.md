# Pullback Riemannian Metric

在這篇文章中，我會嘗試用形式化的方式描述 **pullback Riemannian metric**，
整理他的幾何定義以及在 VAE/Generative Models 中的解釋，同時使用典型教課書例子來補充說明。

## 1. 直觀概念

給定一個映射 \(f: \mathcal{Z} \to \mathcal{X}\) （例如 VAE 中的 decoder），
在資料存在的於輸出空間 \(\mathcal{X}\) 中，我們通常可以使用標準歐幾里得內積來度量不同筆資料之間的「差異」。

Pullback 利用 \(f\) 將輸出空間 \(\mathcal{X}\) 的度量規則搬回潛在空間 \(\mathcal{Z}\)，
使得在 \(\mathcal{Z}\) 中的每個點 \(z\) 的小位移（切向量）也能依「對輸出造成的實際變化」來度量。

若輸出空間 \(\mathcal{X}\) 使用標準內積，則在潛在空間 \(\mathcal{Z}\) 定義的拉回度量可以寫成：

\[G(z) = J_f(z)^T J_f(z)\]

其中 \(J_f(z)\) 是 \(f\) 在點 \(z\) 的 Jacobian。

所以，潛在空間 \(\mathcal{Z}\) 裡的一小步 \(dz\) 的長度由該點在輸出空間 \(\mathcal{X}\) 的影響 \(J_f(z) dz\) 的歐氏距離所描述：

\[\|dz\|_{G(z)} = \|J_f(z) dz\|_2\]

**語義改變大**的方向根據 pullback metric 量測的距離就**較長**。

## 2. 正式定義

設 \(f : (M, -) \to (N, g_N)\) 為光滑映射；\(g_N\) 是目標流形 \(N\) 上的 Riemann metric。
Pullback metric（記為 \(f^* g_N\)）是在 \(M\) 上的新度量，對任意 \(p\in M\) 及切向量 \(u,v \in T_p M\) 定義：

\[(f^* g_N)_p(u,v) = g_N\big(f(p)\big)\big( (Df)_p(u), (Df)_p(v) \big)\]

即：先透過 Jacobian 將 \((u, v)\) 推送到 \(N\) 的切空間，再用既有度量量測。這種「將結構從目標空間帶回來源空間」的普遍操作稱為 pullback（對偶操作是 pushforward 將向量送往目標）。

若 \(N = \mathbb{R}^m\) 且採用標準內積 \(\langle a,b \rangle = a^T b\)，則：

\[(f^* g_{\text{Euc}})_p(u,v) = \langle J_f(p) u, J_f(p) v \rangle = u^T (J_f(p)^T J_f(p)) v\]

## 3. 在 VAE 以及其他生成模型中的語意空間

對一個 VAE 來說，decoder \(f\) 將潛在向量 \(z\) 映射為圖片（資料） \(x = f(z)\)。
透過 pullback，於每個 \(z\) 得到對稱正半定矩陣：

\[G(z) = J_f(z)^T J_f(z)\]

其誘導 (induced) 的內積與長度：

\[\|u\|_{G(z)} = \sqrt{u^T G(z) u} = \|J_f(z) u\|_2\]

對**輸出影像 \(x\)** 造成大改變的**潛在方向 \(dz\)** 被賦予較大長度，
當我們選擇沿著 geodesic （測地線，最短曲線）行走時， geodesic 傾向避開語義跳躍或是不連貫的方向，
因而產生更平滑、語義連續的插值。

## 4. 典型例子

### 4.1 曲線參數化

對於一個單位圓曲線的參數 \(t\)，我們可以把它看作為一個 embedding 。
對於這個潛空間，我們可以定義 projection ：

\[f : [0,2\pi) \to \mathbb{R}^2\]

\[f(t) = (\cos t, \sin t)\]

Projection \(f\) 將一維參數繞成平面單位圓。

他的 Jacobian 可以寫成一個 2x1 的矩陣：

\[J_f(t) = \begin{bmatrix}-\sin t \\ \cos t \end{bmatrix}\]

因此可以算出他的 pullback metric 為：

\[G(t) = J_f(t)^T J_f(t)\]

\[= \begin{bmatrix}-\sin t & \cos t\end{bmatrix} \begin{bmatrix}-\sin t \\ \cos t \end{bmatrix}\]

\[= \sin^2 t + \cos^2 t = 1\]

因為參數空間是一維，因此 metric 是一個純量。

**幾何意義：**

1. 參數 \(t\) 已經是弧長參數（arc-length parameterization），故單位的 \(dt\) 直接對應圓上長度 \(ds=dt\)。
2. 路徑長度 \(L = \int_a^b \sqrt{G(t)} dt = b-a\) 。
3. 沒有尺度扭曲，曲線上任何地方的參數步長都代表同樣的實際弧長。

**討論：**

- 這個參數化本身已經均勻等速的遍歷單位圓，所以 \(G(t)\) 使一個不依賴 \(t\) 的函數。
- 若改成 \(f(t) = (\cos t^2, \sin t^2)\) 則 \(G(t)=4 t^2\) ， pullback 會反映非均勻速度。
- 一維情況下 pullback metric 就是曲線速度平方，速度是一個常數代表 metric 是一個常數。

### 4.2 球面參數化

對於一個標準單位球面 \(S^2\) 的參數化：

\[f(\theta,\phi) = (\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)\]

\[\theta\in[0,\pi], \quad \phi\in[0,2\pi)\]

\(\theta\) 為極角（colatitude），\(\phi\) 為方位角（longitude）。

他的 Jacobian 可以寫成一個 3×2 的矩陣，此時他作用的是一個二維向量 \(\begin{bmatrix}\theta & \phi\end{bmatrix}\)

\[
J_f(\theta,\phi)=\begin{bmatrix}
\cos\theta\cos\phi & -\sin\theta\sin\phi \\
\cos\theta\sin\phi & \sin\theta\cos\phi \\
-\sin\theta & 0
\end{bmatrix}
\]

計算他的 pullback metric \(G = J_f^T J_f\)，可以得到經典球面第一基本形式：

\[G(\theta,\phi)=\begin{bmatrix}1 & 0 \\ 0 & \sin^2\theta\end{bmatrix}\]

其中：
\(\langle \partial_\theta f, \partial_\theta f\rangle=1\)，
\(\langle \partial_\phi f, \partial_\phi f\rangle=\sin^2\theta\)，
\(\langle \partial_\theta f, \partial_\phi f\rangle=0\)。

**幾何意義：**

1. 微線元素 \(ds^2 = d\theta^2 + \sin^2\theta d\phi^2\)。
2. 緯度圈半徑為 \(\sin\theta\)：越靠近兩極（\(\theta \to 0,\pi\)）沿 \(\phi\) 方向的實際距離越小。
3. 參數化是正交的（無交叉項），故微面元素 \(dA = \sin\theta\, d\theta d\phi\)。

**討論：**

- 此參數系統本身是正交座標，所以沒有交叉項。
- 可用 \(\sin^2\phi\) 嗎？不行，因為縮放來自緯度圈半徑，僅依賴 \(\theta\)。
- 兩極「奇異」是坐標系統問題，並非球面幾何本身退化。

**與 4.1 比較：**

| 面向 | 曲線 (4.1) | 球面 (4.2) |
|------|-----------|------------|
| 維度 | 1 → 2 | 2 → 3 |
| Jacobian 形狀 | 2×1 | 3×2 |
| 度量形式 | 常數 1 | \(\mathrm{diag}(1,\sin^2\theta)\) |
| 均勻性 | 參數=弧長 | 經度的縮放隨著 \(\theta\) 改變 |
| 幾何現象 | 等速繞圓 | 緯度圈收縮、兩極奇異 |

這例子展示 pullback 如何自動反映局部**有效尺度**，Jacobian 的 intrinsics 本身定義了局部尺度。

### 4.3 VAE 圖像語義

在某點 \(z\) 上，假設方向 \(du\) 僅改變微弱背景，代表 \(\|du\|_G\) 很小。
假設另一個方向 \(dv\) 改變物件類別，代表 \(\|dv\|_G\) 很大，語意在這個方向上會變化很大。
因此，當我們選擇沿著潛空間 \(\mathcal{Z}\) 中的一個 geodesic 行走，映射到圖像（資料）空間中則會維持語義平順。

## 5. 為何需要拉回

如果我們直接在潛空間中用歐氏距離來計算，會忽略**輸出語義變化速率**的非均勻性。但使用 pullback metric 則可以提供：

1. 語義敏感的距離函數。
2. 更合理的插值，避免穿越輸出大幅扭曲區域或是語意未定義的區域。
3. 可導出 geodesic 最短路徑問題來得到一個合理的 transition trajectory：

\[\mathrm{Length}(\gamma) = \int_0^1 \sqrt{ \dot{\gamma}(t)^T G(\gamma(t)) \dot{\gamma}(t)} dt\]

## 6. Pullback 與 Pushforward（操作對偶）

- Pushforward: 將向量 \(z\) 映射到目標空間中的資料點 \(x\)。

\[(Df)_p : T_p \mathcal{Z} \to T_{f(p)} \mathcal{X}\]

- Pullback Metric: 使用 pushforward 把來源切向量映到目標後，用 \(g_N\) 評估，再把結果視為來源上的內積。

對偶的操作為：向量 \(z\) 被 pushforward 成 \(x\)，metric 被 pullback 回潛空間 \(\mathcal{Z}\)。

## 7. 退化與正則（Degeneracy & Regularization）

若潛在空間 \(\mathcal{Z}\) 的維度 \(d\) 小於目標空間 \(\mathcal{X}\) 的維度 \(m\)，
\(G(z)\) 為 \(d \times d\) Gram 矩陣，若 \(J_f\) 秩不足 \(\mathrm{rank(J_f(z))} < d\) 則 \(G\) 退化為半正定矩陣，也就是會出現一或多個「零特徵值」，表示資訊摺疊。
出現零特徵值意味著在那些方向上，潛在空間的小變動 \(dz\) 幾乎不會在輸出空間造成變化。
換句話說，這些方向對應的是「無資訊方向」或「冗餘方向」。
在 geodesic 最佳化時，這樣的零特徵值會造成：

1. 連續 geodesic ODE 可能病態。
2. 跑最短路時會出現「零成本」方向（不穩定），優化時可能傾向在這些方向「亂跑」，因為在這些方向推動不會被懲罰（距離為零）。

解決方案：

1. 加 Regularization ：\(G_{\varepsilon}(z) = J_f(z)^T J_f(z) + \varepsilon I\)。
2. 特徵值截斷：只保留 \(\lambda_i > \tau\) 主子空間作距離。
3. 以圖近似取代連續優化，隱式避開數值解 geodesic ODE。
