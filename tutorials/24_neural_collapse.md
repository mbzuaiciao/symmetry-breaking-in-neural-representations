# Tutorial 24: Neural Collapse: Emergent Simplex ETF Geometry

## 1. The Terminal Phase of Training (TPT)

When training an overparameterized deep classifier with cross-entropy loss beyond the point of zero training error ($t > T_{\text{zero\_error}}$), the representations and classifier weights undergo a remarkable geometric phase transition termed **Neural Collapse (NC)** (Papyan, Han, & Donoho, 2020).

---

## 2. The Four Properties of Neural Collapse

Let $H_{c, i}$ be the feature representation of sample $i$ in class $c \in \{1, \dots, C\}$.
- **Class Means**: $\mu_c = \frac{1}{N_c} \sum_{i=1}^{N_c} H_{c, i}$
- **Global Mean**: $\mu_G = \frac{1}{C} \sum_{c=1}^C \mu_c$
- **Within-Class Covariance**: $\Sigma_W = \frac{1}{C} \sum_{c=1}^C \frac{1}{N_c} \sum_{i=1}^{N_c} (H_{c, i} - \mu_c)(H_{c, i} - \mu_c)^T$
- **Between-Class Covariance**: $\Sigma_B = \frac{1}{C} \sum_{c=1}^C (\mu_c - \mu_G)(\mu_c - \mu_G)^T$

### NC1: Within-Class Variability Collapse
The representations of all samples in class $c$ collapse to their class mean $\mu_c$:
$$
\text{NC1} = \frac{1}{C} \text{Tr}(\Sigma_W \Sigma_B^{\dagger}) \to 0
$$

### NC2: Simplex Equiangular Tight Frame (ETF) Geometry
The normalized centered class means $\tilde{\mu}_c = \frac{\mu_c - \mu_G}{\|\mu_c - \mu_G\|}$ form a **Simplex ETF**:
- Equal lengths: $\|\tilde{\mu}_c\| = 1$
- Equiangular pairwise angles:
  $$
  \langle \tilde{\mu}_c, \tilde{\mu}_{c'} \rangle = -\frac{1}{C - 1}, \quad \forall c \ne c'
  $$

### NC3: Classifier-to-Feature Duality
The normalized classifier weight vectors $w_c / \|w_c\|$ align perfectly with normalized class means $\tilde{\mu}_c$:
$$
\text{NC3} = \frac{1}{C} \sum_{c=1}^C \left(1 - \frac{\langle w_c, \mu_c - \mu_G \rangle}{\|w_c\| \|\mu_c - \mu_G\|}\right) \to 0
$$

### NC4: Nearest Class-Center Decision Rule
The classifier simplifies to nearest Euclidean distance to class means $\hat{y}(x) = \arg\min_c \|h(x) - \mu_c\|$.
