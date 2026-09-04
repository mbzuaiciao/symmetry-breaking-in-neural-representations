# Tutorial 35: Geometry of Word and Token Embeddings

## 1. Representation Geometry in Language Models

In modern Transformer language models, token embedding matrices $E \in \mathbb{R}^{V \times d}$ map vocabulary tokens into continuous vector spaces.

Key geometric properties include:
1. **Representation Anisotropy**: Embeddings frequently collapse into a narrow cone in embedding space, where random pairs of tokens have high positive cosine similarity (Ethayarajh, 2019).
2. **Frequency and Singular Value Decay**: The dominant principal components often encode token frequency rather than semantics.

---

## 2. Removing Anisotropy: Mean-Centering and All-but-the-Top

To recover semantic angles, embedding matrices are centered and high-variance isotropic noise directions are projected out:
$$
\tilde{e}_w = (e_w - \bar{e}) - \sum_{k=1}^K \langle e_w - \bar{e}, v_k \rangle v_k
$$
where $v_k$ are the top $K$ principal components of the embedding covariance matrix.
