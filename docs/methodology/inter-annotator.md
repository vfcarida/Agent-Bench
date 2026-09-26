# Inter-Annotator Agreement & Judge Calibration

In benchmark evaluation—especially when evaluating subjective dimensions or calibrating LLM judges against human consensus—measuring **rater agreement** is essential for scientific reliability and defensibility.

Agent-Bench provides native statistical implementations for inter-rater reliability in [`agent_bench.metrics.inter_annotator`](file:///c:/Users/vinicius/Documents/GeminiCodes/Agent-Bench/src/agent_bench/metrics/inter_annotator.py):
1. **Cohen's Kappa ($\kappa$)**: Pairwise agreement for nominal categories between two annotators.
2. **Krippendorff's Alpha ($\alpha$)**: Generalized agreement for two or more annotators supporting missing values.
3. **Landis & Koch Benchmark Scale**: Standardized interpretation thresholds.

---

## 1. Cohen's Kappa ($\kappa$)

Cohen's Kappa measures pairwise inter-annotator agreement above that expected purely by chance.

### Formulation
$$\kappa = \frac{P_o - P_e}{1 - P_e}$$

Where:
- $P_o$ is the observed proportional agreement:
  $$P_o = \frac{1}{N} \sum_{i=1}^N \mathbf{1}(y_{1,i} = y_{2,i})$$
- $P_e$ is the expected agreement under statistical independence:
  $$P_e = \sum_{k \in \mathcal{K}} P(y_1 = k) \cdot P(y_2 = k)$$

### Usage Example
```python
from agent_bench.metrics.inter_annotator import compute_cohens_kappa

# Ratings from two independent evaluators
rater_human = ["pass", "pass", "fail", "pass", "refusal"]
rater_model = ["pass", "pass", "fail", "fail", "refusal"]

kappa = compute_cohens_kappa(rater_human, rater_model)
print(f"Cohen's Kappa: {kappa:.4f}")
```

---

## 2. Krippendorff's Alpha ($\alpha$)

Krippendorff's Alpha is a non-parametric agreement coefficient that generalizes across any number of observers and naturally tolerates missing data (unrated items).

### Formulation (Nominal Scale)
$$\alpha = 1 - \frac{D_o}{D_e}$$

Where:
- $D_o$ is the observed disagreement among units:
  $$D_o = \frac{1}{N \cdot \bar{m} (\bar{m} - 1)} \sum_{u} \sum_{c} \sum_{k} n_{u,c} \cdot n_{u,k} \cdot \delta(c, k)$$
- $D_e$ is the disagreement expected by chance across all recorded ratings:
  $$D_e = \frac{1}{N \cdot \bar{m} (N \cdot \bar{m} - 1)} \sum_{c} \sum_{k} n_c \cdot n_k \cdot \delta(c, k)$$

### Usage Example
```python
from agent_bench.metrics.inter_annotator import compute_krippendorffs_alpha_nominal

# Matrix of ratings: items x raters (None represents unrated items)
ratings_matrix = [
    ["pass", "pass", "pass"],
    ["fail", "fail", None],
    ["pass", "fail", "pass"],
    ["refusal", "refusal", "refusal"],
]

alpha = compute_krippendorffs_alpha_nominal(ratings_matrix)
print(f"Krippendorff's Alpha: {alpha:.4f}")
```

---

## 3. Landis & Koch Agreement Interpretation

Agent-Bench classifies agreement coefficients using the established Landis & Koch (1977) scale:

| Coefficient Range ($\kappa, \alpha$) | Agreement Strength | Benchmark Implication |
| :--- | :--- | :--- |
| $< 0.00$ | Poor | Systematic disagreement; reject rubric |
| $0.00 - 0.20$ | Slight | Inadequate alignment; re-train raters |
| $0.21 - 0.40$ | Fair | Ambiguous task instructions |
| $0.41 - 0.60$ | Moderate | Minimum threshold for exploratory pilots |
| $0.61 - 0.80$ | Substantial | Validated threshold for LLM judge deployment |
| $0.81 - 1.00$ | Almost Perfect | Gold standard reference quality |

```python
from agent_bench.metrics.inter_annotator import interpret_agreement

strength = interpret_agreement(0.78)
# Returns: "Substantial agreement"
```

---

## 4. Calibration Datasets

Agent-Bench stores inter-annotator calibration data in `datasets/gold/calibration/annotator_agreement.yaml`. 

To validate agreement across a full calibration dataset manifest:
```python
from agent_bench.metrics.inter_annotator import evaluate_annotation_dataset

report = evaluate_annotation_dataset("datasets/gold/calibration/annotator_agreement.yaml")
print(f"Evaluated items: {report['total_items']}")
print(f"Overall Alpha: {report['krippendorff_alpha']}")
```
