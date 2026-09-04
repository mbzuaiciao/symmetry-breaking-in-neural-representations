# Tutorial 32: Representational Similarity Analysis (RSA) in Neuroscience

## 1. Bridging Artificial and Biological Neural Representations

How can we compare the representations in human fMRI / macaque electrophysiology with activations in an artificial neural network when the recording modalities and neuron identities do not match?

---

## 2. Representational Dissimilarity Matrices (RDMs)

**RSA** (Kriegeskorte et al., 2008) compares representations via their **relational geometry**:
1. Present the same set of $N$ experimental stimuli (e.g. $N = 92$ images) to both the biological subject and artificial network.
2. Construct the $N \times N$ **Representational Dissimilarity Matrix (RDM)**:
   $$
   \text{RDM}_{ij} = 1 - r_{\text{Pearson}}(h(x_i), h(x_j)) \quad \text{or} \quad \|h(x_i) - h(x_j)\|_2
   $$
3. Compare the biological $\text{RDM}_{\text{bio}}$ and artificial $\text{RDM}_{\text{model}}$ via Spearman rank correlation between their upper triangular entries.

---

## 3. Invariance Properties of RSA

Because RDMs are constructed purely from pairwise stimulus relationships, RSA is completely invariant to:
- Permutations of neurons.
- Arbitrary orthogonal coordinate rotations ($R \in O(d)$).
- Non-linear monotonic rescalings of distances (when using Spearman rank correlation).
