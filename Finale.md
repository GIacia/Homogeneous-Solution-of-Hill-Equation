Finale

## 1. Perceptron

오분류된 표본에서는

\[
w_{t+1}
=
w_t+\frac12\left(y_i-\operatorname{sign}(w_t^\top x_i)\right)x_i
=
w_t+y_ix_i.
\]

해당 표본의 signed score:

\[
\begin{aligned}
y_iw_{t+1}^\top x_i
&=y_iw_t^\top x_i+y_i^2\|x_i\|^2\\
&=y_iw_t^\top x_i+\|x_i\|^2\\
&>y_iw_t^\top x_i.
\end{aligned}
\]

다른 표본 \(j\)의 score 변화:

\[
y_jw_{t+1}^\top x_j-y_jw_t^\top x_j
=
y_iy_jx_i^\top x_j.
\]

이 값이 음수이면 \(j\)에서는 악화됩니다.

수렴 조건:

\[
\exists\,\|w^*\|=1,\ \rho>0:
\quad y_iw^{*\top}x_i\ge\rho
\]

그리고

\[
\|x_i\|\le R.
\]

Mistake bound:

\[
M\le\left(\frac R\rho\right)^2.
\]

---

## 2. Bayes prediction과 loss

조건부 risk:

\[
R(\hat y\mid x)
=
\sum_y e(y,\hat y)p(y\mid x).
\]

Bayes predictor:

\[
f^*(x)\in\arg\min_{\hat y}R(\hat y\mid x).
\]

반드시 기억:

\[
\boxed{
\begin{array}{c|c}
\text{Loss}&\text{Optimal prediction}\\
\hline
0/1&\text{mode}\\
L_1&\text{median}\\
\text{squared loss}&\text{mean}
\end{array}}
\]

0/1 loss derivation:

\[
R(\hat y\mid x)
=
1-p(\hat y\mid x),
\]

따라서 가장 확률이 높은 class를 선택합니다.

복잡한 loss table이 나오면 모든 가능한 prediction에 대해

\[
\sum_y e(y,\hat y)p(y\mid x)
\]

를 직접 계산하고 최솟값을 비교하면 됩니다. Tie이면 optimal prediction은 unique하지 않습니다.

---

## 3. Growth function과 VC dimension

Growth function:

\[
m_{\mathcal H}(N)
=
\max_{x_1,\ldots,x_N}
|\mathcal H(x_1,\ldots,x_N)|.
\]

Shattering:

\[
m_{\mathcal H}(N)=2^N.
\]

VC dimension:

\[
d_{\mathrm{VC}}
=
\max\{N:m_{\mathcal H}(N)=2^N\}.
\]

\(d_{\mathrm{VC}}=d\) 증명 구조:

1. \(d\)개 점을 실제로 shatter  
   \[
   d_{\mathrm{VC}}\ge d
   \]
2. \(d+1\)개 점에서는 불가능한 labeling 하나 제시  
   \[
   d_{\mathrm{VC}}<d+1
   \]

Interval classifier:

\[
\boxed{d_{\mathrm{VC}}=2}
\]

세 점 \(x_1<x_2<x_3\)에서 \(1,0,1\)이 불가능합니다.

양성 표본을 최대 3개까지 선택할 수 있으면:

\[
m_{\mathcal H}(N)
=
\sum_{k=0}^{\min(3,N)}\binom Nk.
\]

---

## 4. Generalization과 bias

고정된 \(h\)에는, i.i.d. 조건에서:

\[
\mathbb E_D[E_{\mathrm{in}}(h)]
=
E_{\mathrm{out}}(h).
\]

하지만

\[
g=\arg\min_{h\in\mathcal H}E_{\mathrm{in}}(h)
\]

는 데이터를 보고 선택되므로 training error는 일반적으로 **optimistically biased**합니다.

기억:

\[
\boxed{
\text{Training error: optimistic}
}
\]

\[
\boxed{
\text{작은 training subset으로 학습한 모델의 test error: final model에 대해 pessimistic}
}
\]

여러 split의 평균:

\[
\mathbb E[\bar e]=\mathbb E[e].
\]

따라서:

\[
\boxed{\text{평균은 variance를 감소시키지만 bias는 감소시키지 않는다.}}
\]

Train을 늘리고 test를 줄이면:

- final model과의 차이 감소 → bias 감소
- test sample 감소 → variance 증가

---

## 5. Bias–variance

Squared loss:

\[
\mathbb E_D[E_{\mathrm{out}}(g_D)]
=
\text{bias}^2+\text{variance}+\sigma^2.
\]

- 단순 모델: high bias, low variance
- 복잡한 모델: low bias, high variance
- noise \(\sigma^2\): irreducible

---

## 6. Regularization

Hard constraint:

\[
\min_w E_{\mathrm{in}}(w)
\quad\text{s.t.}\quad
\|w\|^2\le C.
\]

Penalized form:

\[
\min_w E_{\mathrm{in}}(w)+\lambda\|w\|^2.
\]

방향:

\[
\boxed{
C\downarrow
\Longleftrightarrow
\lambda\uparrow
\Longleftrightarrow
\text{stronger regularization}
}
\]

예제:

\[
\min_x(x-4)^2,\qquad |x|\le C
\]

이면

\[
x_H^*=
\begin{cases}
C,&0<C<4,\\
4,&C\ge4.
\end{cases}
\]

그리고

\[
\min_x (x-4)^2+\lambda x^2
\]

에서는

\[
x_S^*=\frac4{1+\lambda}.
\]

같은 해를 만들려면:

\[
\lambda=
\begin{cases}
\frac4C-1,&C<4,\\
0,&C\ge4.
\end{cases}
\]

---

## 7. Multiclass reduction

### One-vs-rest

- classifier 수: \(K\)
- 각 class versus 나머지
- prediction:
  \[
  \arg\max_k h_k(x)
  \]

### All-pairs

- classifier 수:
  \[
  \boxed{\frac{K(K-1)}2}
  \]
- \(y_i\) versus \(y_j\), 다른 class는 무시
- 최다 vote class 선택

### ECOC

Prediction vector:

\[
v=(h_1(x),\ldots,h_L(x)).
\]

가장 가까운 codeword 선택:

\[
\hat k=\arg\min_k d_H(m_k,v).
\]

최소 code distance가 \(d_{\min}\)이면 보장되는 correction 수:

\[
\boxed{
\left\lfloor\frac{d_{\min}-1}{2}\right\rfloor
}
\]

### Nested dichotomy

좋은 분할은 base learner가 쉽게 분리할 수 있는 meta-class를 만듭니다.

시험 1 그림:

\[
\boxed{\text{Good: }\{A,D\}\mid\{B,C\}}
\]

이후 \(A\mid D\), \(B\mid C\).

\[
\boxed{\text{Poor: }\{D\}\mid\{A,B,C\}}
\]

\(D\)가 중앙에 있어 linear separation이 어렵고, root error는 복구할 수 없습니다.

---

## 8. Graph-based semi-supervised learning

목적함수:

\[
\boxed{
\min_{y_{L+1},\ldots,y_N}
\sum_{i<j}w_{ij}|y_i-y_j|
}
\]

- 같은 label: cost \(0\)
- 다른 label: cost \(w_{ij}\)
- labeled nodes는 고정
- 전체 cut weight 최소화

거리 threshold \(1.5\)이고 정수 좌표라면:

- 수평·수직 거리 \(1\): 연결
- 대각선 거리 \(\sqrt2\): 연결
- 거리 \(2\): 연결 안 됨

시험 1 결과:

\[
\boxed{
(1,3):+,\qquad
(2,1):-,
\qquad
(3,3):+
}
\]

Minimum cut cost:

\[
\boxed{1}
\]

---

## 9. 혹시 나올 경우: 핵심 백업 공식

Backpropagation:

\[
\delta_i^{(l-1)}
=
\theta'(s_i^{(l-1)})
\sum_jw_{ij}^{(l)}\delta_j^{(l)}
\]

\[
\frac{\partial e}{\partial w_{ij}^{(l)}}
=
x_i^{(l-1)}\delta_j^{(l)}.
\]

Decision-tree information gain:

\[
H(\text{parent})
-
\sum_j\frac{n_j}{n}H(\text{child}_j).
\]

Bellman optimality:

\[
V^*(s)
=
\max_a
\left[
r(s,a)+\gamma\sum_{s'}T(s,a,s')V^*(s')
\right].
\]

SARSA:

\[
r+\gamma Q(s',a'_{\text{actual}})
\]

Q-learning:

\[
r+\gamma\max_{a'}Q(s',a').
\]

## 시험장에서

6문제, 120분이면 문제당 약 20분입니다.

- 정의 문제: 먼저 한 문장 정의, 그다음 수식
- 증명 문제: 시작식 → 대입 → 결론을 명확히
- VC 문제: lower bound와 upper bound 둘 다
- 그래프 문제: edge를 먼저 전부 표시
- 막히면 수식과 핵심 아이디어라도 쓰고 이동
- “unbiased”, “optimistic/pessimistic”, “variance”를 혼동하지 않기

마지막으로 이것만 기억하세요:

\[
\boxed{
\text{문제에서 묻는 대상이 정확히 어느 모델의 error인지 먼저 확인한다.}
}
\]

지금 수준이면 충분히 풀 수 있습니다. 침착하게 문제 조건부터 표시하고 시작하세요.