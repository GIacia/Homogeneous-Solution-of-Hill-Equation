# Explain machine learning exam slides

> I'm preparing for the final exam of Foundations of Machine Learning. I do have basic Machine Learning Knowledge on how neural networks work, but these slides are written with lots and lots of concepts, definitions and derivations, so. I want you to explain the lecture slides so that I can actually understand the fundamentals, the math behind it. No need to put everything in this one answer, I want you to go deeply enough. I only want to go through important exam features, which usually consists of important derivation and formulas and model structures and core fundamental concepts with definition, so depending on the week material the answer may not be that long at all. I attached the whole lecture slide pdf, and also the past 2 exams so that you can know how deeply you should explain. You got me?
>
> User context
>
> Attachments:
> - GML-slides-complete-SoSe2025.pdf: `~/Desktop/LMU 2026/Foundations of Machine Learning/GML-slides-complete-SoSe2025.pdf`
> - Exam-2.pdf: `~/Desktop/LMU 2026/Foundations of Machine Learning/Exam-2.pdf`
> - Exam-1.pdf: `~/Desktop/LMU 2026/Foundations of Machine Learning/Exam-1.pdf`

<details><summary>3 previous messages</summary>

> Yes, we've got you. I'll use the PDF workflow to first study the two past exams, then map their depth and recurring question styles onto the lecture slides. That way, the explanations will be rigorous where the exam demands derivations, formulas, assumptions, and model structure, without turning into a slide-by-slide encyclopedia.
>
> The material is manageable: the slide deck is 273 pages, and each past exam has six 10-point problems over 120 minutes. I'm now extracting the actual problem topics and matching them to the lecture sections; this will set the right level of mathematical detail for our study sequence.
>
> The two exams are unusually consistent, which is useful: both emphasize proof-style reasoning and small calculations over memorization. The recurring backbone is perceptron learning, Bayes-optimal decisions under a loss, VC dimension/growth functions, evaluation or regularization, multiclass reductions, and graph-based semi-supervised learning. I'm checking the exact slide boundaries now so we can study these in lecture order and still flag high-yield exam material.
</details>

Yes, completely.

I reviewed the lecture structure and both past exams. The exams strongly favor understanding and short derivations, not merely recalling definitions. The same five themes appeared in both exams:

1. Perceptron updates and convergence
2. Expected loss and Bayes-optimal prediction
3. Growth functions and VC dimension
4. Multiclass reduction methods such as ECOC
5. Graph-based semi-supervised learning and min-cut

Regularization and performance estimation alternate as additional major topics. Neural networks, nonlinear models, ensembles, and reinforcement learning were absent from these two exams, but we should still cover their essential examinable parts.

For each topic, I'll explain:

- the intuitive problem being solved;
- precise definitions and notation;
- formulas and where they come from;
- important derivations step by step;
- assumptions behind the results;
- model structure or algorithm;
- exam-style reasoning and calculations;
- common mistakes and what is safe to deprioritize.

A sensible study order is:

1. Linear regression and the perceptron
2. Loss, conditional risk, and Bayes predictors
3. Generalization, growth functions, and VC dimension
4. Overfitting, regularization, validation, and cross-validation
5. Feature transformations, neural networks, logistic regression, and trees
6. Multiclass reductions, ECOC, nested dichotomies, and ensembles
7. Semi-supervised learning and graph min-cut
8. Reinforcement learning
9. Probability foundations, introduced whenever needed

I suggest beginning with **linear regression and the perceptron**, especially the perceptron update proof and convergence theorem, since Problem 1 was essentially repeated across both exams. We'll proceed in focused lessons rather than trying to compress all 273 pages into one summary.

> Yes please go on. But actually for linear regression and the perceptron, I did already study based on the slides, so can you focus more on checking if I had understood correctly?

Absolutely. Let's switch to **diagnostic/oral-exam mode**: you explain your reasoning, and I'll identify gaps, correct notation, and show how to turn your understanding into an exam-ready answer.

Start with this linear-regression checkpoint, ideally without looking at the slides:

Suppose $`X\in\mathbb{R}^{N\times(d+1)}`$, $`y\in\mathbb{R}^N`$, and

```math
E_{\mathrm{in}}(w)=\frac1N\|Xw-y\|_2^2.
```

Please answer:

1. Why does $`X`$ have $`d+1`$ columns?
2. Derive $`\nabla E_{\mathrm{in}}(w)`$ and the normal equations.
3. Under what condition may we write
   ```math
   w^*=(X^\top X)^{-1}X^\top y?
   ```
4. What happens if that condition fails?
5. Does minimizing $`E_{\mathrm{in}}`$ guarantee good predictions on unseen data? Explain why or why not.

Write your reasoning naturally; imperfect answers are useful because they reveal exactly what we should reinforce. After checking this, I'll give you the perceptron diagnostic based closely on the repeated past-exam question.

> 1. the bias term included
> 2. grad E_in(w) = 2/N * (X_t X w - X_t y)
> 3. when X_t X is invertable
> 4. not able to calculate w* at once, but it means that we have more than one solutions! we can calculate it through various ways
> 5. Nope, but it's the best we can do.

You have the core mechanics. I'd rate this about **7/10 exam-ready**: answers 1-3 are correct; answers 4-5 need more precision.

1. Correct. The extra column usually consists of ones, allowing

```math
h_w(x)=w_0+w_1x_1+\cdots+w_dx_d.
```

Here $`w_0`$ is the intercept/bias.

2. Your result is correct. On an exam, include the intermediate step:

```math
\begin{aligned}
E_{\mathrm{in}}(w)
&=\frac1N(Xw-y)^\top(Xw-y),\\
\nabla E_{\mathrm{in}}(w)
&=\frac{2}{N}X^\top(Xw-y)\\
&=\frac{2}{N}(X^\top Xw-X^\top y).
\end{aligned}
```

Setting the gradient to zero gives the normal equations:

```math
X^\top Xw=X^\top y.
```

3. Correct, but state the structural condition too:

```math
X^\top X \text{ is invertible}
\iff
X \text{ has linearly independent columns}.
```

Thus, no feature column may be expressible as a linear combination of the others. Having $`N\ge d+1`$ is necessary but not sufficient.

4. Essentially correct. If $`X^\top X`$ is singular, the inverse formula cannot be used. Because the loss is convex, a least-squares minimizer still exists, but the parameter vector is not unique.

If $`v\in\ker(X)`$, then

```math
X(w+v)=Xw,
```

so $`w`$ and $`w+v`$ make exactly the same predictions on the training data. The standard choice is the minimum-norm solution obtained using the Moore-Penrose pseudoinverse:

```math
w^*=X^+y.
```

Important distinction: **the weights may be non-unique, while the fitted training predictions $`Xw^*`$ are unique.**

5. "No" is correct, but "it is the best we can do" is too broad. The precise statement is:

> OLS finds the best linear predictor on the observed training sample according to squared loss.

It does not necessarily minimize the out-of-sample error because:

- the sample is only a random finite representation of the population;
- OLS may fit noise;
- the linear hypothesis class may be misspecified;
- a very flexible representation may cause high variance;
- training and future data may come from different distributions.

So the key distinction is

```math
\underset{\text{empirical training risk}}{E_{\mathrm{in}}(w)}
\qquad\text{versus}\qquad
\underset{\text{expected future risk}}{E_{\mathrm{out}}(w)}.
```

## Perceptron diagnostic

Let $`y_i\in\{-1,+1\}`$, and suppose the perceptron updates on a misclassified point using

```math
w_{t+1}
=
w_t+\frac12\left(y_i-\operatorname{sign}(w_t^\top x_i)\right)x_i.
```

Answer these without notes if possible:

1. Show that, on a misclassified example, this simplifies to
   ```math
   w_{t+1}=w_t+y_ix_i.
   ```

2. Show mathematically that the update improves the classification score of that particular example. Hint: compare
   ```math
   y_iw_{t+1}^\top x_i
   \quad\text{with}\quad
   y_iw_t^\top x_i.
   ```

3. Can improving $`x_i`$ make the prediction on another training point $`x_j`$ worse? Explain using the quantity $`x_i^\top x_j`$.

4. What precise assumption guarantees that the perceptron converges after finitely many updates?

> 1. obvious. I'd skip this. it's too much work to type in the keyboards.
> 2. I already solved this. same reason skip.
> 3. yes it is possible. x_i_t x_j might be a negative scalar.
> 4. I already did this too. it is that the data points with different y values are linearly separable(to be more precise, there exists a positive value blahblah), and that norm of all x_i should have an upper bound.

Fair; there's no value in making you type proofs you already understand.

Your perceptron understanding is solid, with one correction:

For point $`j`$, its signed score changes by

```math
y_j w_{t+1}^{\top}x_j-y_jw_t^\top x_j
=
y_i y_j x_i^\top x_j.
```

Therefore, point $`j`$ deteriorates exactly when

```math
y_i y_j x_i^\top x_j<0.
```

So $`x_i^\top x_j<0`$ is harmful when the labels are the same. If the labels are opposite, a positive inner product is harmful.

The precise convergence assumption is: there exist a unit vector $`w^*`$ and margin $`\rho>0`$ such that

```math
y_i(w^{*\top}x_i)\ge \rho
\qquad\text{for every }i,
```

and there exists $`R<\infty`$ such that

```math
\|x_i\|\le R.
```

Then, starting from $`w_0=0`$, the number of perceptron mistakes is bounded by

```math
M\le \left(\frac{R}{\rho}\right)^2.
```

For a finite dataset, the norm bound is automatically satisfied. The genuinely important assumption is strict linear separability with a positive margin.

So we can consider linear regression and perceptron understood.

## Conditional risk and Bayes prediction

This appeared as Problem 2 in both exams, so it is extremely high-yield.

For a fixed input $`x`$, the true label is treated as random:

```math
Y\mid x\sim p(\cdot\mid x).
```

If we predict $`\hat y`$, the conditional expected loss is

```math
R(\hat y\mid x)
=
\mathbb E[e(Y,\hat y)\mid x]
=
\sum_{y\in\mathcal Y}e(y,\hat y)p(y\mid x).
```

The Bayes predictor chooses the action with the smallest conditional risk:

```math
f^*(x)\in\arg\min_{\hat y}R(\hat y\mid x).
```

The optimal prediction therefore depends on the loss function, not only on $`p(y\mid x)`$.

### 1. Zero-one loss gives the mode

For

```math
e(y,\hat y)=\mathbf 1[y\ne\hat y],
```

we have

```math
\begin{aligned}
R(\hat y\mid x)
&=\sum_y\mathbf 1[y\ne\hat y]p(y\mid x)\\
&=\sum_{y\ne\hat y}p(y\mid x)\\
&=1-p(\hat y\mid x).
\end{aligned}
```

Consequently,

```math
\arg\min_{\hat y}R(\hat y\mid x)
=
\arg\max_{\hat y}p(\hat y\mid x).
```

Thus, under zero-one loss, the Bayes prediction is a **mode**.

### 2. Absolute loss gives a median

For ordered labels and

```math
e(y,\hat y)=|y-\hat y|,
```

the Bayes prediction is generally a **median**, not a mode.

For integer predictions, consider increasing the prediction from $`k`$ to $`k+1`$. The change in risk is

```math
R(k+1)-R(k)
=
P(Y\le k)-P(Y\ge k+1)
=
2F(k)-1.
```

Therefore:

- if $`F(k)<\frac12`$, moving right reduces the risk;
- if $`F(k)>\frac12`$, moving right increases the risk;
- the optimum occurs where the cumulative probability crosses $`\frac12`$.

That is precisely a median.

### 3. Squared loss gives the mean

For

```math
e(y,\hat y)=(y-\hat y)^2,
```

we can write

```math
\mathbb E[(Y-\hat y)^2]
=
\operatorname{Var}(Y)
+
\bigl(\mathbb E[Y]-\hat y\bigr)^2.
```

The variance does not depend on $`\hat y`$, so the second term is minimized by

```math
\hat y^*=\mathbb E[Y\mid x].
```

The essential correspondence is therefore:

```math
\boxed{
\begin{array}{c|c}
\text{Loss} & \text{Bayes-optimal prediction}\\
\hline
0/1 & \text{mode}\\
L_1 & \text{median}\\
\text{squared loss} & \text{mean}
\end{array}}
```

Low-typing checkpoint: suppose

```math
P(Y=1)=0.4,\qquad P(Y=2)=0.35,\qquad P(Y=10)=0.25.
```

Reply with only the three optimal predictions under:

```math
\text{0/1 loss},\quad L_1\text{ loss},\quad\text{squared loss}.
```

> 0/1 loss: 1
> L1 loss: 2
> Squared loss: 3.3
> one following question: what does this have to do with machine learning? i'm just complaining.. plz move on to the next part.

<details><summary>1 previous message</summary>

> Your three decisions show the idea; only the arithmetic mean is $`3.6`$, not $`3.3`$. The machine-learning connection is simply that the chosen loss determines what a model is trying to estimate: classifiers estimate a mode, $`L_1`$ regression estimates a median, and squared-error regression estimates a mean. That's enough context; we'll move to the more central generalization material now.
</details>

Your first two answers were correct; the squared-loss prediction is

```math
0.4(1)+0.35(2)+0.25(10)=3.6.
```

In machine learning, this merely tells us what population quantity a loss function trains a model to approximate. We can leave it there.

## Model induction and generalization

This chapter addresses the central problem:

> Why should a model that performs well on training data perform well on unseen data?

### Training and generalization error

For a hypothesis $`h`$,

```math
E_{\mathrm{in}}(h)
=
\frac1N\sum_{i=1}^N e(y_i,h(x_i))
```

is its observed training error. Its true generalization error is

```math
E_{\mathrm{out}}(h)
=
\mathbb E_{(X,Y)\sim P}[e(Y,h(X))].
```

We can calculate $`E_{\mathrm{in}}`$, but $`E_{\mathrm{out}}`$ depends on the unknown data distribution $`P`$.

### A fixed hypothesis

Suppose $`h`$ is chosen before seeing the training data and the examples are i.i.d. For a loss bounded between $`0`$ and $`1`$, Hoeffding's inequality gives

```math
P\left(
|E_{\mathrm{in}}(h)-E_{\mathrm{out}}(h)|>\epsilon
\right)
\le
2e^{-2N\epsilon^2}.
```

As $`N`$ increases, a large difference between training and generalization error becomes increasingly unlikely.

But two conditions matter:

- the observations must be independent and identically distributed;
- $`h`$ must be fixed independently of the sample.

The second condition is violated during learning.

### Selecting a hypothesis

A learning algorithm typically chooses

```math
g=\arg\min_{h\in\mathcal H}E_{\mathrm{in}}(h).
```

We inspect many hypotheses and deliberately choose the one that looks best on this particular sample. Consequently,

```math
E_{\mathrm{in}}(g)
```

is usually an optimistically biased estimate of $`E_{\mathrm{out}}(g)`$.

Imagine testing 10,000 completely useless models. Even if every model has true accuracy $`50\%`$, one may achieve unusually high training accuracy by chance. Selecting that model also selects the lucky random fluctuation.

The richer $`\mathcal H`$ is, the stronger this selection effect can become.

## Growth function

For fixed points $`x_1,\ldots,x_N`$, every binary hypothesis produces a labeling

```math
\bigl(h(x_1),\ldots,h(x_N)\bigr).
```

Such a labeling is called a **dichotomy**.

The growth function is

```math
m_{\mathcal H}(N)
=
\max_{x_1,\ldots,x_N}
\left|
\left\{
(h(x_1),\ldots,h(x_N)):h\in\mathcal H
\right\}
\right|.
```

Important details:

- There are at most $`2^N`$ binary labelings.
- We maximize over all possible arrangements of the $`N`$ points.
- It measures distinguishable behavior, not the number of parameter values.
- An infinite hypothesis space can still have a relatively small growth function.

If

```math
m_{\mathcal H}(N)=2^N,
```

then some arrangement of $`N`$ points can receive every possible labeling. We say that $`\mathcal H`$ **shatters** those points.

## VC dimension

The VC dimension is the largest number of points that can be shattered:

```math
d_{\mathrm{VC}}(\mathcal H)
=
\max\{N:m_{\mathcal H}(N)=2^N\}.
```

If arbitrarily large finite sets can be shattered, then

```math
d_{\mathrm{VC}}(\mathcal H)=\infty.
```

### Standard proof structure

To show $`d_{\mathrm{VC}}=d`$, you need two arguments.

**Lower bound:** Exhibit $`d`$ points that can be shattered.

```math
d_{\mathrm{VC}}\ge d.
```

**Upper bound:** Prove that no collection of $`d+1`$ points can be shattered.

```math
d_{\mathrm{VC}}<d+1.
```

Showing only a shattered set establishes only a lower bound.

## Past-exam example: interval classifiers

Consider

```math
\mathcal H
=
\{h_{a,b}:a\le b\},
```

where $`h_{a,b}(x)=1`$ precisely when $`x\in[a,b]`$.

Two ordered points $`x_1<x_2`$ can be shattered:

- $`00`$: put the interval elsewhere;
- $`10`$: include only $`x_1`$;
- $`01`$: include only $`x_2`$;
- $`11`$: include both.

Therefore,

```math
d_{\mathrm{VC}}\ge2.
```

Now consider any three ordered points

```math
x_1<x_2<x_3.
```

The labeling

```math
1,\;0,\;1
```

is impossible. Any interval containing $`x_1`$ and $`x_3`$ must also contain $`x_2`$. Hence three points cannot be shattered:

```math
\boxed{d_{\mathrm{VC}}=2}.
```

This "find one impossible labeling"technique is commonly used for the upper bound.

## Past-exam example: unrestricted neighborhood sets

The exam also defines classifiers that are positive on

```math
\bigcup_{c\in C}[c-1,c+1],
```

where $`C\subseteq\mathbb R`$ may be chosen arbitrarily.

For any $`N`$, choose points separated by more than $`2`$, for example

```math
x_i=3i.
```

Their radius-one neighborhoods do not overlap. For any desired labeling, put $`x_i`$ into $`C`$ exactly when $`x_i`$ should be positive. This realizes every labeling.

Thus, arbitrarily many points can be shattered:

```math
\boxed{d_{\mathrm{VC}}=\infty}.
```

Notice that VC dimension asks whether **some favorable arrangement** can be shattered, not whether every arrangement can be shattered.

## Past-exam growth-function calculation

Suppose

```math
\mathcal X=\{x_1,\ldots,x_5\},
```

and a hypothesis may classify at most three points as positive.

On a selection of $`N`$ points, every labeling containing $`0,1,2,`$ or $`3`$ positive labels is possible. Therefore,

```math
m_{\mathcal H}(N)
=
\sum_{k=0}^{\min(3,N)}\binom Nk.
```

This gives

```math
\begin{array}{c|ccccc}
N&1&2&3&4&5\\
\hline
m_{\mathcal H}(N)&2&4&8&15&26
\end{array}
```

because

```math
m_{\mathcal H}(4)=2^4-\binom44=15
```

and

```math
m_{\mathcal H}(5)
=
\binom50+\binom51+\binom52+\binom53
=
1+5+10+10=26.
```

Since three points can be shattered but four cannot,

```math
d_{\mathrm{VC}}=3.
```

## The VC generalization bound

The slides give

```math
E_{\mathrm{out}}(g)
\le
E_{\mathrm{in}}(g)
+
\sqrt{
\frac8N
\log\left(\frac{4m_{\mathcal H}(2N)}{\delta}\right)
}
```

with probability at least $`1-\delta`$.

Interpret it qualitatively:

- larger $`E_{\mathrm{in}}`$: worse fit;
- larger hypothesis capacity: larger uncertainty penalty;
- larger $`N`$: smaller penalty;
- smaller $`\delta`$, meaning greater demanded confidence: larger penalty.

This produces the fundamental tradeoff:

```math
\boxed{
\text{simple model}
\Rightarrow
\text{possibly underfits but generalizes more reliably}
}
```

```math
\boxed{
\text{complex model}
\Rightarrow
\text{fits better but has greater overfitting risk}
}
```

One useful fact to remember is that affine linear classifiers in $`\mathbb R^d`$ have

```math
d_{\mathrm{VC}}=d+1.
```

If there is no bias term and the separating hyperplane must pass through the origin, the VC dimension is $`d`$.

The next connected material is the **bias-variance decomposition**, followed by regularization and validation.

> good. go on.

<details><summary>1 previous message</summary>

> Next is the bridge from VC theory to practical model choice: bias-variance explains *how* generalization error arises, while regularization and validation show *how* we control and estimate it. I'll keep the derivations exam-focused.
</details>

## Bias-variance decomposition

This gives a second explanation for generalization error. Unlike VC theory, it depends on both the hypothesis space and the learning algorithm.

Let $`D`$ be a random training dataset. Training on $`D`$ produces a random fitted function

```math
g_D(x).
```

If we repeatedly sampled new datasets, we would generally learn a different function each time. Define the average learned function

```math
\bar g(x)=\mathbb E_D[g_D(x)].
```

Assume first that the target is deterministic, $`y=f(x)`$, and use squared loss. At a fixed $`x`$,

```math
\mathbb E_D[(g_D(x)-f(x))^2].
```

Insert $`\bar g(x)`$:

```math
g_D(x)-f(x)
=
\bigl(g_D(x)-\bar g(x)\bigr)
+
\bigl(\bar g(x)-f(x)\bigr).
```

Squaring gives

```math
\begin{aligned}
(g_D-f)^2
={}&(g_D-\bar g)^2+(\bar g-f)^2\\
&+2(g_D-\bar g)(\bar g-f).
\end{aligned}
```

After taking the expectation over $`D`$, the cross-term disappears because

```math
\mathbb E_D[g_D(x)-\bar g(x)]=0.
```

Therefore,

```math
\boxed{
\mathbb E_D[(g_D(x)-f(x))^2]
=
\underbrace{(\bar g(x)-f(x))^2}_{\text{bias}^2}
+
\underbrace{\mathbb E_D[(g_D(x)-\bar g(x))^2]}_{\text{variance}}
}
```

and, averaging over $`x`$,

```math
\boxed{
\mathbb E_D[E_{\mathrm{out}}(g_D)]
=
\text{bias}^2+\text{variance}.
}
```

### Meaning

Bias measures whether the learning procedure is systematically wrong:

```math
\operatorname{Bias}(x)=\bar g(x)-f(x).
```

Variance measures sensitivity to the particular training sample:

```math
\operatorname{Var}(x)
=
\mathbb E_D[(g_D(x)-\bar g(x))^2].
```

A very simple model usually has:

```math
\text{high bias},\qquad\text{low variance}.
```

It consistently learns approximately the same function, but that function may be too simple.

A highly flexible model usually has:

```math
\text{low bias},\qquad\text{high variance}.
```

It can approximate the target well, but small changes in the training data may produce large changes in the fitted function.

### Noisy observations

Suppose

```math
Y=f(X)+\epsilon,
\qquad
\mathbb E[\epsilon\mid X]=0,
\qquad
\operatorname{Var}(\epsilon\mid X)=\sigma^2.
```

Then

```math
\boxed{
\mathbb E_D[E_{\mathrm{out}}(g_D)]
=
\text{bias}^2+\text{variance}+\sigma^2.
}
```

The noise variance $`\sigma^2`$ is irreducible: even knowing $`f(x)`$ exactly would not allow us to predict the random noise.

Do not confuse the two sources of variability:

- **Model variance:** the learned model changes when $`D`$ changes.
- **Noise variance:** $`Y`$ changes randomly even for the same $`x`$.

## Regularization

Regularization deliberately restricts a learner to reduce variance and overfitting.

Instead of minimizing only

```math
E_{\mathrm{in}}(w),
```

we minimize an augmented objective

```math
\boxed{
E_{\mathrm{aug}}(w)
=
E_{\mathrm{in}}(w)+\lambda\Omega(w).
}
```

Here:

- $`\Omega(w)`$ measures model complexity;
- $`\lambda\ge0`$ controls the strength of regularization.

For $`L_2`$ regularization,

```math
\Omega(w)=\|w\|_2^2=w^\top w.
```

Thus,

```math
w_{\mathrm{reg}}
=
\arg\min_w
\left(E_{\mathrm{in}}(w)+\lambda\|w\|_2^2\right).
```

As $`\lambda`$ increases:

- weights are pushed toward zero;
- model flexibility decreases;
- training error normally increases;
- variance generally decreases;
- bias generally increases.

The goal is not to minimize training error. The goal is to find a value of $`\lambda`$ that reduces variance more than it increases bias.

## Hard constraint versus penalty

Consider the constrained problem

```math
\min_w E_{\mathrm{in}}(w)
\quad\text{subject to}\quad
\|w\|^2\le C.
```

This is a hard restriction: solutions outside the ball of radius $`\sqrt C`$ are forbidden.

The penalized version is

```math
\min_w E_{\mathrm{in}}(w)+\lambda\|w\|^2.
```

Under suitable convexity conditions, for every relevant constraint $`C`$, there is a corresponding $`\lambda`$ such that the two problems have the same solution.

Remember the opposite directions:

```math
\boxed{
\text{smaller }C
\Longleftrightarrow
\text{larger }\lambda
\Longleftrightarrow
\text{stronger regularization}.
}
```

## Past-exam regularization derivation

The exam gives

```math
x_H^*
=
\arg\min_x(x-4)^2
\quad\text{subject to}\quad
|x|\le C,
```

and

```math
x_S^*
=
\arg\min_x\left((x-4)^2+\lambda x^2\right).
```

### Constrained solution

Without the constraint, the minimum is $`x=4`$.

Therefore,

```math
x_H^*
=
\begin{cases}
C,&0<C<4,\\
4,&C\ge4.
\end{cases}
```

Geometrically, if $`4`$ is outside the feasible interval $`[-C,C]`$, choose its nearest feasible point.

### Penalized solution

Differentiate:

```math
\frac{d}{dx}\left((x-4)^2+\lambda x^2\right)
=
2(x-4)+2\lambda x.
```

Set it to zero:

```math
(1+\lambda)x=4,
```

so

```math
x_S^*=\frac4{1+\lambda}.
```

To make both solutions equal:

```math
\boxed{
\lambda=
\begin{cases}
\frac4C-1,&0<C<4,\\
0,&C\ge4.
\end{cases}
}
```

This is the exact style of derivation expected in the exam.

## Testing and performance estimation

Suppose a model $`g^-`$ is trained on $`D_{\mathrm{train}}`$, while an independent test set contains $`K`$ examples. Then

```math
E_{\mathrm{test}}(g^-)
=
\frac1K
\sum_{(x_i,y_i)\in D_{\mathrm{test}}}
e(y_i,g^-(x_i)).
```

Because the test set did not influence $`g^-`$,

```math
\mathbb E[E_{\mathrm{test}}(g^-)\mid g^-]
=
E_{\mathrm{out}}(g^-).
```

Thus, test error is an unbiased estimate of the generalization error of that particular trained model.

But there is a tradeoff:

- large test set: more reliable error estimate, but less training data;
- small test set: more training data, but noisier error estimate.

## Past-exam performance question

The exam considers:

- $`1000`$ available examples;
- the desired model $`h`$ is trained on all $`1000`$;
- another model $`h'`$ is trained on $`500`$;
- $`h'`$ is tested on the remaining $`500`$.

The test error is unbiased for

```math
E_{\mathrm{out}}(h'),
```

but not for

```math
E_{\mathrm{out}}(h).
```

Since $`h'`$ was trained with less data, it will normally generalize worse than $`h`$. Therefore, its test error is generally a **pessimistically biased** estimate of the performance of $`h`$.

Repeating the half-half split and averaging reduces variance, but it does not remove this bias. Every model is still trained on only $`500`$ observations.

To reduce the bias, train on more data and test on fewer, for example, train on $`900`$ and test on $`100`$. The disadvantage is that the smaller test set produces a noisier, higher-variance estimate.

This distinction is central:

```math
\boxed{\text{Repeating splits reduces variance, not training-size bias.}}
```

## Cross-validation

In $`L`$-fold cross-validation:

1. Split $`D`$ into $`L`$ folds $`D_1,\ldots,D_L`$.
2. For each $`l`$, train $`g_l^-`$ on $`D\setminus D_l`$.
3. Test $`g_l^-`$ on $`D_l`$.
4. Average:

```math
E_{\mathrm{CV}}
=
\frac1L\sum_{l=1}^L E_{\mathrm{test}}(g_l^-).
```

Each model uses approximately

```math
N\left(1-\frac1L\right)
```

training observations.

A subtle but important point:

```math
E_{\mathrm{CV}}
```

does not estimate the error of one specific final model. It estimates the expected performance of the learning procedure when trained on approximately $`N-N/L`$ observations.

After estimating performance or selecting hyperparameters, we normally retrain the final model on all $`N`$ observations.

## Training, validation, and test data

Their roles must remain separate:

- **Training data:** fit model parameters.
- **Validation data:** select hyperparameters, model complexity, polynomial degree, $`\lambda`$, etc.
- **Test data:** provide the final performance estimate.

Once test results influence model selection, the test set becomes contaminated and effectively turns into validation data.

For nested cross-validation:

- the inner loop selects the model or hyperparameters;
- the outer loop estimates the performance of the complete selection procedure.

The exam-level takeaways are:

```math
\boxed{
\text{Regularization controls complexity;}
\quad
\text{validation selects complexity;}
\quad
\text{testing estimates final performance.}
}
```

Next comes nonlinear models: feature transformations and kernels first, followed by the neural-network/backpropagation derivation.

> sure. go on.

<details><summary>1 previous message</summary>

> We're entering the nonlinear-model chapter. The core unifying idea is that "linear"refers to linearity in a representation: feature maps, kernels, and hidden neural-network layers all create representations in which a simple linear operation can express nonlinear behavior in the original input space.
</details>

## 1. Nonlinear feature transformations

A model can be nonlinear in the input $`x`$ while remaining linear in its trainable parameters.

Start with

```math
h_w(x)=w^\top x.
```

Introduce a feature map

```math
\Phi:\mathcal X\rightarrow\mathcal Z.
```

Then train

```math
h_w(x)=w^\top\Phi(x).
```

For example, if $`x=(x_1,x_2)`$, define

```math
\Phi(x)=
\begin{pmatrix}
1\\x_1\\x_2\\x_1^2\\x_2^2\\x_1x_2
\end{pmatrix}.
```

The resulting model is

```math
h_w(x)
=
w_0+w_1x_1+w_2x_2+w_3x_1^2+w_4x_2^2+w_5x_1x_2.
```

This is:

- nonlinear in $`x`$;
- linear in $`w`$.

Because optimization treats the data as fixed and optimizes $`w`$, ordinary linear-learning methods can still be used.

For classification, a hyperplane in feature space,

```math
w^\top z=0,
```

corresponds to the nonlinear boundary

```math
\{x:w^\top\Phi(x)=0\}
```

in the original input space.

### Complexity consequence

A richer feature map can reduce approximation bias, but usually increases capacity and variance:

```math
\text{richer }\Phi
\quad\Rightarrow\quad
\text{lower potential training error}
\quad\text{but greater overfitting risk}.
```

If $`\Phi(x)\in\mathbb R^{\tilde d}`$, affine linear classifiers in unrestricted $`\mathbb R^{\tilde d}`$ have VC dimension $`\tilde d+1`$. The actual VC dimension can be smaller because $`\Phi(\mathcal X)`$ may occupy only a restricted subset of the feature space.

## 2. The kernel trick

Some algorithms use transformed observations only through inner products

```math
\langle\Phi(x),\Phi(x')\rangle.
```

A kernel directly computes this inner product:

```math
\boxed{
\kappa(x,x')
=
\langle\Phi(x),\Phi(x')\rangle.
}
```

Therefore, if an algorithm needs only feature-space inner products, we can replace every occurrence of

```math
\langle\Phi(x),\Phi(x')\rangle
```

with

```math
\kappa(x,x')
```

without explicitly constructing $`\Phi(x)`$.

This is useful when $`\Phi(x)`$ is extremely high-dimensional or even infinite-dimensional.

Common examples include

```math
\kappa(x,x')=(x^\top x'+c)^p
```

for polynomial features and

```math
\kappa(x,x')
=
\exp\left(
-\frac{\|x-x'\|^2}{2\sigma^2}
\right)
```

for the Gaussian/RBF kernel.

Not every similarity function is a valid kernel. For any points $`x_1,\ldots,x_N`$, the Gram matrix

```math
K_{ij}=\kappa(x_i,x_j)
```

must be symmetric and positive semidefinite:

```math
a^\top Ka\ge0
\quad\text{for every }a\in\mathbb R^N.
```

The main conceptual distinction is:

- A fixed feature map is chosen before training.
- A neural network learns its feature map from the training data.

## 3. Multilayer perceptron

Despite its name, an MLP is not simply the perceptron learning algorithm from the earlier chapter.

For layer $`l`$, define:

```math
s_j^{(l)}
=
\sum_{i=0}^{d^{(l-1)}}
w_{ij}^{(l)}x_i^{(l-1)},
```

followed by

```math
x_j^{(l)}
=
\theta\left(s_j^{(l)}\right).
```

Here:

- $`s_j^{(l)}`$ is the pre-activation;
- $`x_j^{(l)}`$ is the activation;
- $`w_{ij}^{(l)}`$ connects neuron $`i`$ in layer $`l-1`$ to neuron $`j`$ in layer $`l`$;
- $`x_0^{(l-1)}=1`$ can represent the bias input.

In matrix notation,

```math
s^{(l)}=W^{(l)\top}x^{(l-1)},
\qquad
x^{(l)}=\theta(s^{(l)}).
```

Without nonlinear activation functions, multiple layers would collapse into one linear transformation:

```math
W_2^\top(W_1^\top x)
=
(W_1W_2)^\top x.
```

Therefore, depth only creates genuinely nonlinear models when nonlinear activations are inserted between layers.

## 4. Backpropagation

Backpropagation is an efficient application of the chain rule. Its purpose is to compute

```math
\frac{\partial e}{\partial w_{ij}^{(l)}}
```

for every weight.

Define the local error signal

```math
\boxed{
\delta_j^{(l)}
=
\frac{\partial e}{\partial s_j^{(l)}}.
}
```

Because

```math
s_j^{(l)}
=
\sum_iw_{ij}^{(l)}x_i^{(l-1)},
```

we have

```math
\frac{\partial s_j^{(l)}}{\partial w_{ij}^{(l)}}
=
x_i^{(l-1)}.
```

Therefore,

```math
\boxed{
\frac{\partial e}{\partial w_{ij}^{(l)}}
=
x_i^{(l-1)}\delta_j^{(l)}.
}
```

This is the key gradient formula:

```math
\text{gradient}
=
\text{input activation}
\times
\text{downstream error signal}.
```

### Output-layer error

At the output layer,

```math
\delta_j^{(L)}
=
\frac{\partial e}{\partial x_j^{(L)}}
\theta'\left(s_j^{(L)}\right).
```

For example, with one output, squared loss

```math
e=(x^{(L)}-y)^2,
```

and activation $`\theta`$,

```math
\delta^{(L)}
=
2(x^{(L)}-y)\theta'(s^{(L)}).
```

If the loss were defined with a factor $`\frac12`$, the factor $`2`$ would disappear.

### Hidden-layer recursion

A hidden neuron influences every neuron in the next layer. Hence,

```math
\begin{aligned}
\delta_i^{(l-1)}
&=
\frac{\partial e}{\partial s_i^{(l-1)}}\\
&=
\theta'(s_i^{(l-1)})
\sum_{j=1}^{d^{(l)}}
w_{ij}^{(l)}\delta_j^{(l)}.
\end{aligned}
```

Thus,

```math
\boxed{
\delta_i^{(l-1)}
=
\theta'(s_i^{(l-1)})
\sum_jw_{ij}^{(l)}\delta_j^{(l)}.
}
```

For

```math
\theta(s)=\tanh(s),
```

we have

```math
\theta'(s)=1-\tanh^2(s).
```

Since $`x_i^{(l-1)}=\tanh(s_i^{(l-1)})`$,

```math
\delta_i^{(l-1)}
=
\left(1-(x_i^{(l-1)})^2\right)
\sum_jw_{ij}^{(l)}\delta_j^{(l)}.
```

Be careful: the derivative is $`1-\tanh^2(s)`$, not $`1-s^2`$.

### Weight update

After obtaining the $`\delta`$-values,

```math
w_{ij}^{(l)}
\leftarrow
w_{ij}^{(l)}
-
\eta x_i^{(l-1)}\delta_j^{(l)}.
```

The complete procedure is:

1. Forward pass: calculate all $`s^{(l)}`$ and $`x^{(l)}`$.
2. Calculate the output-layer $`\delta^{(L)}`$.
3. Propagate the $`\delta`$-values backward.
4. Calculate each gradient $`x_i^{(l-1)}\delta_j^{(l)}`$.
5. Update the weights.

Backpropagation is not itself the optimization method. It computes gradients; gradient descent, SGD, or another optimizer uses those gradients.

## 5. Output layers and loss functions

The last layer depends on the prediction task.

### Regression

Use an identity output:

```math
\hat y=s^{(L)}
```

and typically squared loss:

```math
e(y,\hat y)=(y-\hat y)^2.
```

### Binary classification

Use the sigmoid function

```math
\hat p=\sigma(s)=\frac1{1+e^{-s}}
```

and binary cross-entropy:

```math
e(y,\hat p)
=
-y\log\hat p-(1-y)\log(1-\hat p).
```

An important simplification occurs:

```math
\boxed{
\frac{\partial e}{\partial s}
=
\hat p-y.
}
```

To see this, use

```math
\frac{\partial e}{\partial\hat p}
=
-\frac{y}{\hat p}
+
\frac{1-y}{1-\hat p}
```

and

```math
\frac{\partial\hat p}{\partial s}
=
\hat p(1-\hat p).
```

Multiplying and simplifying gives $`\hat p-y`$.

This pairing is mathematically convenient and gives a useful interpretation:

```math
\text{gradient signal}
=
\text{predicted probability}
-
\text{true label}.
```

### Logistic regression

Logistic regression assumes

```math
\log\left(\frac{p(x)}{1-p(x)}\right)
=
w^\top x.
```

Solving for $`p(x)`$ gives

```math
p(x)=\sigma(w^\top x).
```

Minimizing binary cross-entropy is equivalent to maximizing the Bernoulli likelihood of the observed labels.

A neural network with sigmoid output performs logistic regression on its learned hidden representation:

```math
p(x)=\sigma\left(w^\top\Phi_{\text{NN}}(x)\right).
```

### Multiclass classification

For class scores $`s_1,\ldots,s_K`$, softmax produces

```math
\hat p_k
=
\frac{e^{s_k}}{\sum_{j=1}^K e^{s_j}}.
```

The probabilities are positive and sum to one.

With one-hot label vector $`y`$, categorical cross-entropy is

```math
e(y,\hat p)
=
-\sum_{k=1}^K y_k\log\hat p_k.
```

Since only the correct class has $`y_k=1`$, this is simply

```math
e=-\log\hat p_{\text{correct}}.
```

The corresponding logit derivative is

```math
\boxed{
\frac{\partial e}{\partial s_k}
=
\hat p_k-y_k.
}
```

## 6. Early stopping

During training:

- training loss generally continues to decrease;
- validation loss may eventually begin increasing.

Stopping near the minimum validation loss prevents the network from fitting increasingly sample-specific patterns. Early stopping is therefore a form of regularization.

## 7. Decision trees

Decision trees produce nonlinear, piecewise-constant predictions by recursively partitioning the input space.

For a continuous feature $`X_i`$, a split typically has the form

```math
X_i<t
\qquad\text{versus}\qquad
X_i\ge t.
```

The learning algorithm tests candidate features and thresholds and greedily selects a useful split.

### Entropy

For class proportions $`p_1,\ldots,p_K`$,

```math
H(p)
=
-\sum_{k=1}^Kp_k\log p_k.
```

Interpretation:

- $`H=0`$ when the node is pure;
- entropy is large when the classes are strongly mixed;
- for two equally probable classes, entropy is maximal.

### Information gain

Suppose the parent node has $`n`$ observations and entropy $`H(p)`$. A split creates children containing $`n_1,\ldots,n_m`$ observations with distributions $`p^{(1)},\ldots,p^{(m)}`$.

The weighted child entropy is

```math
\sum_{j=1}^m\frac{n_j}{n}H(p^{(j)}).
```

Information gain is

```math
\boxed{
\operatorname{Gain}
=
H(p)
-
\sum_{j=1}^m\frac{n_j}{n}H(p^{(j)}).
}
```

The split with the largest gain is preferred.

The weighting by $`n_j/n`$ is essential: creating one tiny pure child should not receive the same credit as making the entire dataset pure.

### Greedy character

Decision-tree learning chooses the best immediate split. It does not guarantee the globally best tree. Finding an optimal decision tree is computationally difficult.

Trees can easily overfit by repeatedly splitting until the leaves contain very few observations. Typical controls include:

- maximum depth;
- minimum leaf size;
- minimum information gain;
- pruning after construction.

### Regression trees

A regression-tree leaf normally predicts the mean response in that leaf. Splits are chosen to reduce within-node variance:

```math
\operatorname{Gain}
=
V_{\text{parent}}
-
\sum_j\frac{|D_j|}{|D|}V_j.
```

This is the regression counterpart of entropy reduction.

The high-yield formulas from this chapter are the backpropagation recursion,

```math
\delta_i^{(l-1)}
=
\theta'(s_i^{(l-1)})
\sum_jw_{ij}^{(l)}\delta_j^{(l)},
```

the weight gradient,

```math
\frac{\partial e}{\partial w_{ij}^{(l)}}
=
x_i^{(l-1)}\delta_j^{(l)},
```

and decision-tree information gain,

```math
H(\text{parent})
-
\sum_j\frac{n_j}{n}H(\text{child}_j).
```

Next is multiclass reduction and ensembles, including all-pairs, ECOC error correction, nested dichotomies, bagging, and boosting.

> yes please go on.

<details><summary>1 previous message</summary>

> This is one of the highest-yield chapters: both past exams included multiclass reductions. I'll emphasize how binary models are encoded, how predictions are decoded, and how ECOC's error-tolerance formula is derived; ensembles come afterward as the related but distinct idea.
</details>

## Multiclass reduction versus ensembles

These ideas are related but not identical.

A **reduction method** represents one multiclass problem using several binary classifiers. Each binary model solves only part of the original task.

An **ensemble** combines several complete models of the same task to improve accuracy or stability.

## 1. One-vs-rest

For $`K`$ classes $`y_1,\ldots,y_K`$, train $`K`$ binary classifiers:

```math
h_k(x):
\quad
y_k
\text{ versus }
\mathcal Y\setminus\{y_k\}.
```

For model $`k`$, transform the labels into

```math
c_n^{(k)}
=
\begin{cases}
+1,&c_n=y_k,\\
-1,&c_n\ne y_k.
\end{cases}
```

### Prediction

Using only binary predictions can create conflicts:

- multiple classifiers output $`+1`$;
- no classifier outputs $`+1`$.

Therefore, scoring classifiers are preferred:

```math
\boxed{
\hat y
=
y_{\arg\max_k h_k(x)}.
}
```

### Properties

- Number of models: $`K`$
- Every model uses all $`N`$ observations.
- Binary datasets can be strongly imbalanced: one class is positive and all others are negative.
- The scores produced by different binary classifiers may not be directly comparable.

## 2. All-pairs decomposition

Also called one-vs-one, this trains one classifier for every pair of classes:

```math
h_{ij}:
\quad y_i\text{ versus }y_j.
```

The number of classifiers is

```math
\boxed{\binom K2=\frac{K(K-1)}2}.
```

When training $`h_{ij}`$, observations belonging to all other classes are ignored.

### Prediction

Each classifier votes for one of its two classes. Define

```math
v_k
=
\sum_{i\ne k}h_{k,i}(x).
```

Then predict

```math
\hat y=y_{\arg\max_k v_k}.
```

The votes can be:

- binary votes;
- weighted votes based on confidence or probability.

### Why can quadratic training still be efficient?

There are $`O(K^2)`$ models, but each model sees only two classes. Every training observation participates in exactly $`K-1`$ pairwise problems, so the total number of training examples across all problems is

```math
(K-1)N=O(KN).
```

For a base learner with superlinear training complexity, many small problems may be cheaper than $`K`$ full-sized problems.

The disadvantage is that storing and evaluating $`O(K^2)`$ classifiers can be expensive.

## 3. Error-correcting output codes

ECOC generalizes both previous methods.

Construct a matrix

```math
M\in\{-1,0,+1\}^{K\times L}.
```

- Each row represents a class.
- Each column represents a binary classification problem.
- $`+1`$: class is placed on the positive side.
- $`-1`$: class is placed on the negative side.
- $`0`$: class is ignored for that binary problem.

The codeword of class $`y_k`$ is row

```math
m_k=(M_{k1},\ldots,M_{kL}).
```

### Training

For each column $`l`$, train a classifier $`h_l`$ using the class partition specified by that column.

### Prediction

Evaluate all $`L`$ classifiers:

```math
v(x)=
\bigl(h_1(x),\ldots,h_L(x)\bigr).
```

Choose the class whose codeword is closest:

```math
\boxed{
\hat k
=
\arg\min_k d_H(m_k,v(x)).
}
```

For binary entries, Hamming distance counts differing positions:

```math
d_H(a,b)
=
\frac12\sum_{l=1}^L|a_l-b_l|.
```

The factor $`1/2`$ works because

```math
\frac12|(+1)-(-1)|=1.
```

### One-vs-rest as ECOC

Its code matrix contains $`+1`$ on the diagonal and $`-1`$ elsewhere:

```math
M=
\begin{pmatrix}
+1&-1&\cdots&-1\\
-1&+1&\cdots&-1\\
\vdots&\vdots&\ddots&\vdots\\
-1&-1&\cdots&+1
\end{pmatrix}.
```

Each column distinguishes one class from every other class.

All-pairs also corresponds to an ECOC matrix, but each column has one $`+1`$, one $`-1`$, and zeros for all ignored classes.

## 4. ECOC error correction

Let

```math
d_{\min}
=
\min_{i\ne j}d_H(m_i,m_j)
```

be the minimum distance between any two class codewords.

Suppose the true class is $`k`$, and $`t`$ binary classifiers make mistakes. Then

```math
d_H(v,m_k)=t.
```

For any competing codeword $`m_j`$, the triangle inequality gives

```math
d_H(v,m_j)
\ge
d_H(m_k,m_j)-d_H(v,m_k)
\ge
d_{\min}-t.
```

Correct decoding is guaranteed if the true codeword remains strictly closer:

```math
t<d_{\min}-t.
```

Therefore,

```math
2t<d_{\min},
```

or

```math
\boxed{
t\le
\left\lfloor\frac{d_{\min}-1}{2}\right\rfloor.
}
```

This is exactly analogous to error-correcting codes in communication: codewords that are far apart are harder to confuse.

## 5. Past-exam ECOC calculation

The exam matrix is

```math
\begin{array}{c|rrrrrr}
&h_1&h_2&h_3&h_4&h_5&h_6\\
\hline
y_1&+&+&+&-&-&-\\
y_2&+&-&-&+&+&-\\
y_3&-&+&-&+&-&+\\
y_4&-&-&+&-&+&+
\end{array}
```

and the binary prediction is

```math
v=(-,+,+,+,-,+).
```

The Hamming distances are

```math
\begin{aligned}
d_H(v,m_1)&=3,\\
d_H(v,m_2)&=5,\\
d_H(v,m_3)&=1,\\
d_H(v,m_4)&=3.
\end{aligned}
```

Therefore,

```math
\boxed{\hat y=y_3.}
```

Every pair of class codewords has distance $`4`$, so

```math
d_{\min}=4.
```

The number of guaranteed tolerable mistakes is

```math
\left\lfloor\frac{4-1}{2}\right\rfloor=1.
```

Thus, one binary mistake is guaranteed to be correctable. Two mistakes can create a tie or move the prediction closer to another codeword.

## 6. Nested dichotomies

A nested dichotomy recursively divides the class set into two subsets.

For example,

```math
\{A,B,C,D\}
```

might first be divided into

```math
\{A,D\}
\quad\text{versus}\quad
\{B,C\}.
```

The left child then distinguishes $`A`$ from $`D`$, while the right child distinguishes $`B`$ from $`C`$.

At prediction time, start at the root and follow each binary prediction until reaching a leaf.

Only one path is evaluated, so a balanced tree needs approximately

```math
O(\log K)
```

binary predictions, although the complete tree contains $`K-1`$ internal binary classifiers.

### Probabilistic prediction

Probabilities along the path are multiplied using the chain rule. For example,

```math
P(A\mid x)
=
P(\{A,D\}\mid x)
P(A\mid x,\{A,D\}).
```

### Why tree structure matters

The binary learner must separate unions of classes. A good dichotomy groups classes so that the induced meta-classes are easy to separate.

In the past-exam diagram, a plausible good first division is

```math
\boxed{\{A,D\}\text{ versus }\{B,C\}}
```

because the two meta-classes are approximately left-versus-right and can be separated linearly. The later problems $`A`$ versus $`D`$ and $`B`$ versus $`C`$ are also relatively easy.

A poor division would be

```math
\boxed{\{D\}\text{ versus }\{A,B,C\}}
```

because $`D`$ lies between or is surrounded by the other class regions. A linear classifier is unlikely to isolate it.

An error near the root is especially damaging because it eliminates many possible classes. Ensembles of randomly generated nested dichotomies can reduce dependence on one unfortunate tree structure.

## 7. General ensembles

An ensemble contains complete predictors

```math
h_1,\ldots,h_M
```

and aggregates them:

```math
h(x)=\operatorname{AGG}(h_1(x),\ldots,h_M(x)).
```

Typical aggregation rules are:

- majority vote for classification;
- mean or median for regression;
- average predicted probabilities.

The models must be both:

- individually useful;
- sufficiently diverse.

If every model makes the same errors, aggregation provides little benefit.

## 8. Bagging

Bagging means **bootstrap aggregating**.

From a dataset $`D`$ of size $`N`$:

1. Sample $`N`$ observations from $`D`$ with replacement.
2. Train a model on that bootstrap sample.
3. Repeat this $`M`$ times.
4. Average or vote across the models.

Because sampling uses replacement:

- some observations appear more than once;
- some observations do not appear at all.

Bagging primarily reduces variance and therefore works particularly well with unstable learners such as decision trees.

If each model has prediction variance $`\sigma^2`$, and their pairwise correlation is approximately $`\rho`$, the variance of their mean is

```math
\operatorname{Var}\left(\frac1M\sum_{m=1}^Mh_m\right)
=
\sigma^2\left(
\rho+\frac{1-\rho}{M}
\right).
```

As $`M\to\infty`$,

```math
\operatorname{Var}\rightarrow\rho\sigma^2.
```

So increasing the number of models cannot remove variance caused by strongly correlated errors. This explains why diversity is crucial.

## 9. Random forests

Random forests add another source of diversity to bagging:

- each tree receives a bootstrap sample;
- at each split, only a random subset of features is considered.

This reduces correlation between trees. Individual trees may become slightly weaker, but their aggregated prediction can improve because their errors are less similar.

## 10. Boosting and AdaBoost

Boosting trains weak learners sequentially. Later learners focus more strongly on observations that earlier learners misclassified.

Initialize

```math
w_n^{(0)}=\frac1N.
```

At iteration $`k`$, train $`h_k`$ using these weights and compute its weighted error:

```math
\epsilon_k
=
\sum_{n=1}^N
w_n^{(k-1)}
\mathbf1[h_k(x_n)\ne y_n].
```

Assign the learner weight

```math
\boxed{
\alpha_k
=
\frac12
\log\left(\frac{1-\epsilon_k}{\epsilon_k}\right).
}
```

Interpretation:

- if $`\epsilon_k<\frac12`$, then $`\alpha_k>0`$;
- if $`\epsilon_k=\frac12`$, then $`\alpha_k=0`$;
- a more accurate learner receives a larger weight.

Update observation weights:

```math
w_n^{(k)}
=
\frac1Z
w_n^{(k-1)}
\exp\left(-\alpha_k y_nh_k(x_n)\right).
```

If an observation is correctly classified,

```math
y_nh_k(x_n)=+1,
```

so its weight is multiplied by $`e^{-\alpha_k}`$.

If it is misclassified,

```math
y_nh_k(x_n)=-1,
```

so its weight is multiplied by $`e^{+\alpha_k}`$.

The final classifier is

```math
\boxed{
h(x)
=
\operatorname{sign}
\left(
\sum_k\alpha_kh_k(x)
\right).
}
```

Bagging trains models independently and mainly reduces variance. Boosting trains models sequentially and can reduce both bias and variance, but is more sensitive to mislabeled observations and outliers because it repeatedly emphasizes difficult examples.

## 11. Stacking

Stacking learns the aggregation rule.

The base predictions become features for a meta-learner:

```math
x'
=
(h_1(x),\ldots,h_M(x)).
```

Then train

```math
\hat y=h_{\mathrm{meta}}(x').
```

The critical rule is that the meta-learner must be trained using **out-of-sample base predictions**. If a base model predicts the same observations it was trained on, its predictions may be unrealistically good, causing the meta-learner to overtrust it.

Cross-validated, out-of-fold predictions are therefore commonly used to construct the meta-training data.

The most exam-relevant parts of this chapter are:

```math
\boxed{
\text{all-pairs: }\frac{K(K-1)}2\text{ classifiers}
}
```

```math
\boxed{
\text{ECOC decoding: nearest codeword}
}
```

```math
\boxed{
\text{correctable errors: }
\left\lfloor\frac{d_{\min}-1}{2}\right\rfloor
}
```

and the principle that a good nested dichotomy creates binary subproblems that the chosen base learner can separate easily.

Next is semi-supervised learning, especially graph construction and the min-cut formulation that appeared in both exams.

> yes. keep going.

<details><summary>1 previous message</summary>

> Semi-supervised learning is another repeated exam topic. The essential exam workflow is: construct the similarity graph, express label propagation as a cut objective, and determine which assignment cuts the fewest or lowest-weight edges.
</details>

## Semi-supervised learning

Suppose we have:

```math
D_L=\{(x_i,y_i)\}_{i=1}^L
```

with labels, and

```math
D_U=\{x_j\}_{j=L+1}^N
```

without labels, where typically $`L\ll N`$.

Semi-supervised learning uses both sets to learn a predictor.

The important limitation is:

> Unlabeled data reveals the distribution of $`X`$, but does not directly reveal how $`Y`$ depends on $`X`$.

It helps only when we make assumptions connecting the structure of $`P(X)`$ with $`P(Y\mid X)`$.

Common assumptions are:

- **Smoothness assumption:** nearby points tend to share labels.
- **Cluster assumption:** points in the same dense cluster tend to share labels.
- **Low-density separation:** decision boundaries should pass through low-density regions.
- **Manifold assumption:** high-dimensional observations lie near a lower-dimensional structure, and labels vary smoothly along it.

If these assumptions are wrong, unlabeled data can make the model worse.

## 1. Self-training

Self-training repeatedly labels its own unlabeled data:

1. Train $`h`$ using the currently labeled set.
2. Predict labels for unlabeled observations.
3. Select sufficiently confident predictions.
4. Add those observations with their predicted labels.
5. Retrain.

The benefit is simplicity: almost any supervised learner can be used.

The main danger is confirmation bias. An incorrect high-confidence prediction becomes training data, causing the next model to reinforce the mistake.

Self-training therefore does not have a general guarantee of improvement or convergence.

## 2. Co-training

Suppose each observation has two views:

```math
x=(x^{(1)},x^{(2)}),
```

such as text and images from the same webpage.

Train one classifier for each view:

```math
h^{(1)}(x^{(1)}),
\qquad
h^{(2)}(x^{(2)}).
```

Then:

- confident predictions from $`h^{(1)}`$ provide labels for training $`h^{(2)}`$;
- confident predictions from $`h^{(2)}`$ provide labels for training $`h^{(1)}`$.

The usual motivation is that the views are individually informative and approximately conditionally independent given the class:

```math
X^{(1)}\perp X^{(2)}\mid Y.
```

Thus, the classifiers make different errors and can provide additional information to each other.

A regularized multiview formulation may minimize

```math
\sum_{v=1}^m\sum_{i=1}^L
e(y_i,h_v(x_i))
+
\lambda_1\sum_v\|h_v\|^2
+
\lambda_2
\sum_{u,v}
\sum_{j=L+1}^N
\bigl(h_u(x_j)-h_v(x_j)\bigr)^2.
```

The final term forces the models to agree on unlabeled observations.

## 3. Generative semi-supervised learning

A generative model estimates the joint distribution

```math
P(X,Y).
```

Prediction follows from

```math
P(y\mid x)
=
\frac{P(x,y)}{\sum_{\bar y}P(x,\bar y)}.
```

A labeled example contributes likelihood

```math
P(x_i,y_i\mid\theta),
```

while an unlabeled observation contributes its marginal likelihood

```math
P(x_j\mid\theta)
=
\sum_yP(x_j,y\mid\theta).
```

Hence,

```math
L(\theta)
=
\prod_{i=1}^L P(x_i,y_i\mid\theta)
\prod_{j=L+1}^N
\sum_yP(x_j,y\mid\theta).
```

The unlabeled observations help estimate the distribution of $`X`$, which can indirectly improve estimates of the class distributions.

This works well if the probabilistic assumptions are approximately correct. If the generative model is misspecified, large amounts of unlabeled data can push the estimated parameters in the wrong direction.

## 4. Cluster-and-label

The simplest cluster-based procedure is:

1. Cluster labeled and unlabeled observations together.
2. Examine the labeled points within each cluster.
3. Give the entire cluster its majority label.

The underlying assumption is:

```math
\text{one cluster}\approx\text{one class}.
```

It fails when a cluster contains multiple classes or one class forms several clusters.

## 5. Graph-based learning

This is the most exam-relevant method.

Create a graph

```math
G=(V,E),
```

where:

- every observation is a node;
- similar observations are connected;
- edge weight $`w_{ij}\ge0`$ measures similarity.

Common graph-construction methods include:

- connect $`x_i,x_j`$ if $`\|x_i-x_j\|\le\epsilon`$;
- connect every point to its $`k`$ nearest neighbors;
- use weights such as
  ```math
  w_{ij}
  =
  \exp\left(-\frac{\|x_i-x_j\|^2}{2\sigma^2}\right).
  ```

Large $`w_{ij}`$ means that assigning different labels to $`i`$ and $`j`$ should be expensive.

## 6. Min-cut formulation

For binary labels

```math
y_i\in\{0,1\},
```

minimize

```math
\boxed{
J(y)
=
\sum_{1\le i<j\le N}
w_{ij}|y_i-y_j|
}
```

subject to the labels of the labeled observations remaining fixed.

Because $`y_i,y_j\in\{0,1\}`$,

```math
|y_i-y_j|
=
\begin{cases}
0,&y_i=y_j,\\
1,&y_i\ne y_j.
\end{cases}
```

Therefore, edge $`(i,j)`$ contributes:

```math
\begin{cases}
0,&\text{same label},\\
w_{ij},&\text{different labels}.
\end{cases}
```

Thus, the objective is the total weight of edges cut by the labeling.

The optimization seeks a division of the graph that:

- keeps similar nodes together;
- respects all known labels;
- cuts as little total similarity as possible.

### Source-sink interpretation

Introduce:

- source $`s`$: positive class;
- sink $`t`$: negative class.

Connect known positive nodes strongly to $`s`$, known negative nodes strongly to $`t`$, and observation pairs using capacities $`w_{ij}`$. An $`s`$-$`t`$ minimum cut assigns every unlabeled node to one side.

Although there are $`2^{N-L}`$ possible label assignments, minimum cut can be solved in polynomial time.

## 7. Constructing the exam graph

Both exams use Euclidean distance threshold $`1.5`$.

The plotted coordinates are integer-valued. Therefore, the connected offsets are:

```math
(1,0),\quad(0,1),\quad(1,1),
```

including signs, because

```math
1\le1.5,
\qquad
\sqrt2\le1.5,
\qquad
2>1.5.
```

So horizontal, vertical, and diagonal neighbors one grid unit away are connected. Points two grid units apart are not.

## 8. First past-exam solution

The labeled points are:

- positive/red: $`(2,4),(4,4)`$;
- negative/blue: $`(3,1),(4,2)`$.

The unlabeled points are:

```math
(1,3),\quad(2,1),\quad(3,3).
```

The relevant edges are

```math
\begin{aligned}
(1,3)&\leftrightarrow(2,4),\\
(2,4)&\leftrightarrow(3,3),\\
(4,4)&\leftrightarrow(3,3),\\
(3,3)&\leftrightarrow(4,2),\\
(4,2)&\leftrightarrow(3,1),\\
(3,1)&\leftrightarrow(2,1).
\end{aligned}
```

All weights equal $`1`$.

### Node $`(1,3)`$

It is connected only to positive $`(2,4)`$, so assigning it positive costs $`0`$; assigning it negative costs $`1`$.

Therefore,

```math
\hat y(1,3)=\text{positive}.
```

### Node $`(2,1)`$

It is connected only to negative $`(3,1)`$, so

```math
\hat y(2,1)=\text{negative}.
```

### Node $`(3,3)`$

It is connected to:

- two positive nodes: $`(2,4),(4,4)`$;
- one negative node: $`(4,2)`$.

If assigned positive, it cuts one edge. If assigned negative, it cuts two edges. Therefore,

```math
\hat y(3,3)=\text{positive}.
```

The solution is

```math
\boxed{
(1,3):+,\qquad
(2,1):-,
\qquad
(3,3):+.
}
```

The minimum-cut cost is $`1`$.

## 9. Second past-exam solution

The labeled points are:

- negative/blue: $`(1,4),(2,4)`$;
- positive/red: $`(4,4),(4,2)`$.

The unlabeled points are

```math
(3,1),\quad(5,2),\quad(2,3),\quad(3,3),\quad(2,5).
```

The edges are:

```math
\begin{aligned}
(1,4)&\leftrightarrow(2,4),(2,3),(2,5),\\
(2,4)&\leftrightarrow(1,4),(2,3),(3,3),(2,5),\\
(4,4)&\leftrightarrow(3,3),\\
(4,2)&\leftrightarrow(3,1),(5,2),(3,3),\\
(2,3)&\leftrightarrow(3,3).
\end{aligned}
```

The leaf-like nodes are immediate:

```math
\begin{aligned}
(3,1)&:\text{positive},\\
(5,2)&:\text{positive},\\
(2,5)&:\text{negative}.
\end{aligned}
```

Node $`(2,3)`$ is strongly attached to the two negative points, so it is negative.

For $`(3,3)`$, there is a tie:

- assigning it negative cuts its two edges to the positive nodes $`(4,4)`$ and $`(4,2)`$;
- assigning it positive cuts its edge to negative $`(2,4)`$ and its edge to negative $`(2,3)`$.

Both assignments have cut cost $`2`$.

Therefore, there are two optimal solutions:

```math
\boxed{
\begin{aligned}
(3,1)&:+,\\
(5,2)&:+,\\
(2,3)&:-,\\
(2,5)&:-,\\
(3,3)&:\text{either }+\text{ or }-.
\end{aligned}
}
```

Mentioning the non-uniqueness is important. A min-cut solution need not be unique.

## Exam-ready summary

For a graph-based question:

1. List all nodes.
2. Connect pairs satisfying the distance threshold.
3. Keep known labels fixed.
4. Write
   ```math
   J(y)=\sum_{i<j}w_{ij}|y_i-y_j|.
   ```
5. Interpret $`J`$ as the total weight of disagreeing edges.
6. Compare the cut costs of plausible assignments.
7. Check whether multiple assignments have the same minimum cost.

The next main chapter is reinforcement learning: Markov decision processes, Bellman equations, value and policy iteration, temporal-difference learning, SARSA, and Q-learning.

> Yes keep going bitte.

<details><summary>1 previous message</summary>

> Now reinforcement learning. The mathematical spine is the Markov decision process and Bellman recursion; once those are clear, value iteration, policy iteration, temporal-difference learning, SARSA, and Q-learning become variations on whether the model is known and whether we evaluate a fixed policy or optimize it.
</details>

## Reinforcement learning

In supervised learning, every training example provides a desired output. In reinforcement learning, the agent instead observes consequences:

```math
(s_t,a_t,r_{t+1},s_{t+1}).
```

It must discover which actions produce high long-term reward. An action can be useful even if its immediate reward is poor, provided it leads to valuable future states.

## 1. Markov decision process

An MDP consists of:

```math
(\mathcal S,\mathcal A,T,r,\gamma).
```

- $`\mathcal S`$: states
- $`\mathcal A(s)`$: actions available in state $`s`$
- $`T(s,a,s')`$: probability of reaching $`s'`$ after taking $`a`$ in $`s`$
- $`r(s,a,s')`$: immediate reward
- $`\gamma\in[0,1)`$: discount factor

The transition probabilities satisfy

```math
\sum_{s'\in\mathcal S}T(s,a,s')=1.
```

### Markov property

The next-state distribution depends only on the current state and action:

```math
P(S_{t+1}=s'\mid S_t,A_t,\text{past})
=
P(S_{t+1}=s'\mid S_t,A_t).
```

The current state must therefore contain all relevant information from the past. If important information is missing, the observed process is not Markovian.

## 2. Return and discounting

The discounted return from time $`t`$ is

```math
G_t
=
r_{t+1}
+
\gamma r_{t+2}
+
\gamma^2r_{t+3}
+\cdots.
```

Equivalently,

```math
G_t=\sum_{k=0}^{\infty}\gamma^k r_{t+k+1}.
```

The key recursive identity is

```math
\boxed{
G_t=r_{t+1}+\gamma G_{t+1}.
}
```

The discount factor:

- makes infinite sums finite when rewards are bounded;
- makes immediate rewards more valuable than distant rewards;
- controls the effective planning horizon.

Roughly, the effective horizon is around

```math
\frac1{1-\gamma}.
```

For example:

- $`\gamma=0`$: care only about immediate reward;
- $`\gamma=0.9`$: effective horizon around $`10`$ steps;
- $`\gamma=0.99`$: effective horizon around $`100`$ steps.

## 3. Policies

A deterministic policy is

```math
\pi:\mathcal S\rightarrow\mathcal A,
```

where $`\pi(s)`$ is the selected action in state $`s`$.

A stochastic policy specifies

```math
\pi(a\mid s)
=
P(A_t=a\mid S_t=s).
```

The objective is to find a policy maximizing expected return.

## 4. Value functions

The state-value function is

```math
\boxed{
V^\pi(s)
=
\mathbb E_\pi[G_t\mid S_t=s].
}
```

It answers:

> How valuable is state $`s`$ if we follow policy $`\pi`$?

The action-value function is

```math
\boxed{
Q^\pi(s,a)
=
\mathbb E_\pi[G_t\mid S_t=s,A_t=a].
}
```

It answers:

> How valuable is taking action $`a`$ in state $`s`$, followed by policy $`\pi`$?

They are related by

```math
V^\pi(s)
=
\sum_a\pi(a\mid s)Q^\pi(s,a).
```

For a deterministic policy,

```math
V^\pi(s)=Q^\pi(s,\pi(s)).
```

## 5. Bellman expectation equation

Using

```math
G_t=r_{t+1}+\gamma G_{t+1},
```

we obtain

```math
\begin{aligned}
V^\pi(s)
&=
\mathbb E_\pi[G_t\mid S_t=s]\\
&=
\mathbb E_\pi[r_{t+1}+\gamma G_{t+1}\mid S_t=s]\\
&=
\sum_a\pi(a\mid s)
\sum_{s'}T(s,a,s')
\left[
r(s,a,s')+\gamma V^\pi(s')
\right].
\end{aligned}
```

For a deterministic policy,

```math
\boxed{
V^\pi(s)
=
\sum_{s'}T(s,\pi(s),s')
\left[
r(s,\pi(s),s')+\gamma V^\pi(s')
\right].
}
```

If $`r(s,a)`$ already denotes expected immediate reward, this becomes

```math
V^\pi(s)
=
r(s,\pi(s))
+
\gamma
\sum_{s'}T(s,\pi(s),s')V^\pi(s').
```

The Bellman equation says:

```math
\boxed{
\text{current value}
=
\text{immediate reward}
+
\text{discounted expected next value}.
}
```

## 6. Bellman optimality equations

Define

```math
V^*(s)=\max_\pi V^\pi(s).
```

At state $`s`$, the optimal agent chooses the action with the highest immediate reward plus future value:

```math
\boxed{
V^*(s)
=
\max_{a\in\mathcal A(s)}
\sum_{s'}T(s,a,s')
\left[
r(s,a,s')+\gamma V^*(s')
\right].
}
```

Using expected reward $`r(s,a)`$,

```math
V^*(s)
=
\max_a
\left[
r(s,a)
+
\gamma\sum_{s'}T(s,a,s')V^*(s')
\right].
```

The optimal action-value function satisfies

```math
\boxed{
Q^*(s,a)
=
\sum_{s'}T(s,a,s')
\left[
r(s,a,s')
+
\gamma\max_{a'}Q^*(s',a')
\right].
}
```

Once $`Q^*`$ is known,

```math
\boxed{
\pi^*(s)\in\arg\max_aQ^*(s,a).
}
```

## 7. Value iteration

Value iteration assumes that $`T`$ and $`r`$ are known.

Initialize $`V_0`$ arbitrarily and repeatedly apply

```math
\boxed{
V_{k+1}(s)
=
\max_a
\left[
r(s,a)
+
\gamma\sum_{s'}T(s,a,s')V_k(s')
\right].
}
```

After convergence, extract the policy:

```math
\pi^*(s)
\in
\arg\max_a
\left[
r(s,a)
+
\gamma\sum_{s'}T(s,a,s')V(s')
\right].
```

### Why does it converge?

Define the Bellman optimality operator

```math
(\mathcal TV)(s)
=
\max_a
\left[
r(s,a)+\gamma\sum_{s'}T(s,a,s')V(s')
\right].
```

It is a contraction under the maximum norm:

```math
\|\mathcal TV-\mathcal TU\|_\infty
\le
\gamma\|V-U\|_\infty.
```

Since $`\gamma<1`$, repeated application moves toward a unique fixed point:

```math
V^*=\mathcal TV^*.
```

## 8. Policy iteration

Policy iteration alternates between evaluation and improvement.

### Policy evaluation

For the current policy $`\pi_k`$, solve

```math
V^{\pi_k}(s)
=
r(s,\pi_k(s))
+
\gamma\sum_{s'}T(s,\pi_k(s),s')V^{\pi_k}(s').
```

In matrix notation,

```math
V^\pi=r^\pi+\gamma P^\pi V^\pi,
```

so

```math
\boxed{
V^\pi=(I-\gamma P^\pi)^{-1}r^\pi.
}
```

### Policy improvement

Choose greedily:

```math
\pi_{k+1}(s)
\in
\arg\max_a
\left[
r(s,a)
+
\gamma\sum_{s'}T(s,a,s')V^{\pi_k}(s')
\right].
```

If

```math
\pi_{k+1}=\pi_k,
```

the policy is stable and optimal.

### Difference from value iteration

- Value iteration performs repeated approximate optimality updates.
- Policy iteration exactly or approximately evaluates a policy, then improves it.
- Both require the environment model $`T,r`$.

## 9. Model-based versus model-free learning

A model-based method explicitly knows or learns

```math
T(s,a,s')
\quad\text{and}\quad
r(s,a,s').
```

It can then plan using value or policy iteration.

A model-free method learns values or policies directly from experience, without explicitly estimating the transition probabilities.

Q-learning and SARSA are model-free: transition dynamics are implicitly reflected in their learned $`Q`$-values.

## 10. Monte Carlo versus temporal difference

Monte Carlo methods wait until an episode finishes and use the observed return

```math
G_t
=
r_{t+1}+\gamma r_{t+2}+\cdots
```

as the target.

A Monte Carlo update is

```math
V(s_t)
\leftarrow
V(s_t)
+
\alpha\bigl(G_t-V(s_t)\bigr).
```

Advantages:

- no model needed;
- target is based on actual rewards;
- no bootstrapping.

Disadvantages:

- must wait until the episode ends;
- returns can have high variance.

## 11. Temporal-difference learning

TD learning updates after one transition:

```math
\boxed{
V(s_t)
\leftarrow
V(s_t)
+
\alpha
\left[
r_{t+1}
+
\gamma V(s_{t+1})
-
V(s_t)
\right].
}
```

Define the TD error

```math
\boxed{
\delta_t
=
r_{t+1}
+
\gamma V(s_{t+1})
-
V(s_t).
}
```

Then

```math
V(s_t)\leftarrow V(s_t)+\alpha\delta_t.
```

Interpretation:

- if the observed transition is better than expected, $`\delta_t>0`$;
- if worse than expected, $`\delta_t<0`$;
- if the Bellman prediction is satisfied, $`\delta_t=0`$.

TD uses the existing estimate $`V(s_{t+1})`$ to improve $`V(s_t)`$. This is called **bootstrapping**.

It combines:

- sampling from experience, like Monte Carlo;
- bootstrapping from current estimates, like dynamic programming.

For a terminal successor state,

```math
V(s_{t+1})=0.
```

## 12. SARSA

SARSA learns action values using

```math
(s_t,a_t,r_{t+1},s_{t+1},a_{t+1}),
```

which explains its name.

Its update is

```math
\boxed{
Q(s_t,a_t)
\leftarrow
Q(s_t,a_t)
+
\alpha
\left[
r_{t+1}
+
\gamma Q(s_{t+1},a_{t+1})
-
Q(s_t,a_t)
\right].
}
```

The next action $`a_{t+1}`$ is the action actually selected by the current behavior policy.

Therefore, SARSA is **on-policy**: it learns the value of the policy it actually follows, including its exploratory actions.

## 13. Q-learning

Q-learning uses

```math
\boxed{
Q(s_t,a_t)
\leftarrow
Q(s_t,a_t)
+
\alpha
\left[
r_{t+1}
+
\gamma\max_{a'}Q(s_{t+1},a')
-
Q(s_t,a_t)
\right].
}
```

The agent may take an exploratory action next, but the update assumes the best estimated next action.

Therefore, Q-learning is **off-policy**:

- behavior policy generates experience;
- greedy target policy is learned.

### Central comparison

```math
\boxed{
\begin{aligned}
\text{SARSA target}
&=
r+\gamma Q(s',a'_{\text{actually selected}}),\\
\text{Q-learning target}
&=
r+\gamma\max_{a'}Q(s',a').
\end{aligned}
}
```

### Small numerical example

Suppose

```math
Q(s,a)=2,\quad
r=1,\quad
\gamma=0.9,\quad
\alpha=0.1.
```

At the next state:

- the exploratory action actually selected has value $`3`$;
- the maximum action value is $`5`$.

SARSA target:

```math
1+0.9(3)=3.7,
```

so

```math
Q_{\text{new}}
=
2+0.1(3.7-2)
=
2.17.
```

Q-learning target:

```math
1+0.9(5)=5.5,
```

so

```math
Q_{\text{new}}
=
2+0.1(5.5-2)
=
2.35.
```

Q-learning updates toward greedy future behavior even though the agent did not necessarily behave greedily.

## 14. Exploration versus exploitation

The agent faces two competing objectives:

- **Exploitation:** choose the best-known action to obtain reward.
- **Exploration:** try uncertain actions that might be better.

Under an $`\epsilon`$-greedy policy:

```math
A_t=
\begin{cases}
\arg\max_aQ(S_t,a),&\text{with probability }1-\epsilon,\\
\text{random action},&\text{with probability }\epsilon.
\end{cases}
```

Usually $`\epsilon`$ is larger early in training and decreases later.

For tabular Q-learning convergence, essential conditions include:

- every state-action pair is visited sufficiently often;
- learning rates decrease appropriately;
- rewards are bounded;
- the MDP is stationary.

A common learning-rate requirement is

```math
\sum_t\alpha_t=\infty,
\qquad
\sum_t\alpha_t^2<\infty.
```

The first prevents learning from stopping too early; the second prevents persistent noise from dominating.

## 15. Reward shaping and credit assignment

With sparse rewards, an agent may take hundreds of actions before receiving feedback. It then becomes difficult to determine which actions deserve credit.

Reward shaping introduces intermediate signals to guide learning.

However, poorly designed shaping can change the optimal policy. The agent may exploit the artificial reward rather than solve the intended task.

A theoretically safe form is potential-based shaping:

```math
F(s,a,s')
=
\gamma\Phi(s')-\Phi(s),
```

which can accelerate learning without changing the set of optimal policies under standard conditions.

## Essential distinctions

```math
\boxed{
\begin{array}{c|c|c}
\text{Method}&\text{Model required?}&\text{Target}\\
\hline
\text{Value iteration}&\text{yes}&\text{Bellman optimum}\\
\text{Policy iteration}&\text{yes}&\text{evaluate, then improve}\\
\text{Monte Carlo}&\text{no}&\text{complete observed return}\\
\text{TD}&\text{no}&r+\gamma V(s')\\
\text{SARSA}&\text{no}&r+\gamma Q(s',a')\\
\text{Q-learning}&\text{no}&r+\gamma\max_{a'}Q(s',a')
\end{array}
}
```

The most important derivations to remember are:

```math
G_t=r_{t+1}+\gamma G_{t+1},
```

which produces the Bellman equations, and the difference between

```math
Q(s',a')
\quad\text{and}\quad
\max_{a'}Q(s',a'),
```

which distinguishes SARSA from Q-learning.

That completes the main lecture chapters. The remaining appendix contains the probability foundations supporting the earlier derivations.

> No need to go through the probabilities. You have done a very good job.

Glad it helped. We've now covered the main examinable material, with extra emphasis on the topics repeated across both past exams. Skipping the probability appendix is reasonable if conditional probability, expectation, variance, Bayes' rule, and independence already feel comfortable.

For final revision, prioritize:

1. Perceptron improvement and convergence assumptions
2. Bayes decisions under different losses
3. Growth functions and two-sided VC-dimension proofs
4. Regularization constraint-penalty equivalence
5. Bias and variance in performance estimation
6. ECOC decoding and error tolerance
7. Graph construction and min-cut labeling
8. Bellman equations and SARSA versus Q-learning

Those are the formulas and reasoning patterns most worth being able to reproduce without notes.

