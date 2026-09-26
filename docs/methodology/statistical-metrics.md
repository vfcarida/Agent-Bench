# Statistical Metrics & Reliability Engineering

Autonomous agents exhibit non-deterministic stochastic behavior across repetitions. Evaluating an agent based on a single trial ($N=1$) provides insufficient statistical power and hides critical failure modes.

**Agent-Bench** implements mathematically rigorous, unbiased statistical estimators to evaluate agent reliability and consistency.

---

## 1. Unpooled $\text{Pass}@k$ Estimator

The standard accuracy metric measures the empirical mean across $N$ independent trials:

$$\text{Accuracy} = \frac{c}{n}$$

where $n$ is the total number of evaluation trials and $c$ is the number of successful trials for a given task.

### Mathematical Formulation

To estimate the probability that **at least one** generated solution succeeds when $k$ independent attempts are sampled ($k \le n$), Agent-Bench uses the unbiased combinatorial estimator (Chen et al., HumanEval):

$$\text{Pass}@k = \mathbb{E}\left[ 1 - \frac{\binom{n - c}{k}}{\binom{n}{k}} \right] = 1 - \frac{\binom{n - c}{k}}{\binom{n}{k}}$$

### Boundary Properties
- If $c = n$ (all trials pass): $\text{Pass}@k = 1.0$.
- If $c = 0$ (no trials pass): $\text{Pass}@k = 0.0$.
- If $n < k$: $\text{Pass}@k = 1.0$ if $c > 0$, else $0.0$.
- If $k \le 0$: safely returns $0.0$.

### Why Anti-Pooling is Critical
In naive benchmark implementations, samples across different tasks are often pooled into a single global array. This distorts results because an agent that passes 10/10 trials on an easy task and 0/10 on a hard task would receive an inflated pooled score. In Agent-Bench, $\text{Pass}@k$ is computed **strictly per task**, and then averaged across tasks.

---

## 2. $\text{Pass}^k$ Reliability for High-Risk Domains

In high-risk autonomous domains (e.g. banking transfers, cybersecurity responses), an agent that succeeds 1 out of 5 times ($\text{Pass}@5 \approx 100\%$) is completely unusable in production. For transactional operations, we require **consistency across all sampled attempts**.

### Mathematical Formulation

Inspired by Sierra's $\tau$-bench ($\tau^2$-bench, 2024), Agent-Bench implements $\text{Pass}^k$: the probability that **all $k$ sampled trials succeed simultaneously**:

$$\text{Pass}^k = \frac{\binom{c}{k}}{\binom{n}{k}}$$

### Monotonicity Invariants
For any valid $k \le n$:
$$\text{Pass}^k \le \text{Pass}@1 \le \text{Pass}@k$$

And across increasing sample size $k$:
- $k_1 < k_2 \implies \text{Pass}@k_1 \le \text{Pass}@k_2$ (monotonically non-decreasing)
- $k_1 < k_2 \implies \text{Pass}^{k_1} \ge \text{Pass}^{k_2}$ (monotonically non-increasing)

---

## 3. Stratified Bootstrap 95% Confidence Intervals

To quantify statistical uncertainty without assuming a normal distribution, Agent-Bench computes non-parametric bootstrap confidence intervals (1,000 to 10,000 resamples):

1. For each domain with $M$ tasks, sample $M$ tasks with replacement.
2. For each task, compute the headline metric ($\text{Pass}@1$, $\text{Pass}^3$, $\text{Pass}@5$).
3. Aggregate the resampled domain score.
4. Compute the 2.5th and 97.5th percentiles across bootstrap replications:

$$\text{CI}_{95\%} = \left[ q_{0.025}, q_{0.975} \right]$$

This ensures differences between evaluated models (e.g., Model A vs Model B) are statistically meaningful and not noise artifacts.

---

## 4. Latency Percentiles & Financial Cost Accounting

### Latency Distribution
Agent-Bench tracks exact per-step and cumulative end-to-end execution latency in milliseconds, computing:
- $p_{50}$ (Median response time)
- $p_{90}$ (Tail latency)
- $p_{99}$ (Worst-case latency)

### Financial Accounting (USD)
Token consumption (`tokens_in`, `tokens_out`) is multiplied by the provider's exact token pricing declared in `configs/models/`:

$$\text{Cost}_{\text{task}} = \left(\frac{\text{tokens}_{\text{in}}}{1,000} \times P_{\text{in}}\right) + \left(\frac{\text{tokens}_{\text{out}}}{1,000} \times P_{\text{out}}\right)$$

We also report **Cost-per-Successful-Task**:

$$\text{Cost}_{\text{success}} = \frac{\sum \text{Cost}}{\max(1, \text{Passed Tasks})}$$
