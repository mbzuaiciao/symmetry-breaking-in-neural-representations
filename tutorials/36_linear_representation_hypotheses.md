# Tutorial 36: The Linear Representation Hypothesis

## 1. What Is the Linear Representation Hypothesis?

The **Linear Representation Hypothesis (LRH)** posits that high-level concepts and features in neural networks correspond to linear 1D directions or low-dimensional subspaces in activation space:
$$
\text{Property}(x) = \langle h(x), v_{\text{concept}} \rangle
$$
Classical examples:
- Word2Vec vector arithmetic: $v_{\text{king}} - v_{\text{man}} + v_{\text{woman}} \approx v_{\text{queen}}$.
- Truthfulness / honesty steering vectors in LLMs (Marks & Tegmark, 2023).

---

## 2. Empirical Reality vs Strong Theoretical Claims

- **Empirical Observation**: Linear probes achieve high accuracy on simple binary attributes (e.g. sentiment, grammatical tense).
- **Caveat**: Finding a linear separating hyperplane does *not* prove the network computes the concept linearly; non-linear manifolds in high dimensions are almost always linearly separable (Cover's Theorem on Separability).
