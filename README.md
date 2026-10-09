# Predicting Student Health Risk — Single FT-Transformer with PLR Embeddings

**Kaggle Rank 89 / 3,355 (top 2.7%) — LB Balanced Accuracy 0.95059, achieved with a single model (no ensembling, no seed averaging)**

Solution for [Kaggle Playground Series S6E7](https://www.kaggle.com/competitions/playground-series-s6e7): a 3-class student health risk classification task, evaluated by **balanced accuracy** on imbalanced classes.

## Approach

A single **FT-Transformer** (Feature Tokenizer + Transformer, Gorishniy et al. 2021), implemented in **pure PyTorch**, upgraded with **PLR numerical embeddings** (Periodic-Linear-ReLU, Gorishniy et al. 2022, *On Embeddings for Numerical Features*).

### The key upgrade: PLR numerical embeddings

The vanilla FT-Transformer tokenizes each numeric feature with a simple linear map (`x*W+b`). Here that tokenizer is replaced by:

$$x \;\rightarrow\; \text{ReLU}\Big(\text{Linear}\big[\sin(2\pi v x),\ \cos(2\pi v x)\big]\Big)$$

with **learnable frequencies `v` per feature** (`k=24` frequencies, `σ=0.1` init). This gives the transformer a multi-scale periodic representation of every numeric input — and it measurably beats the vanilla FT-Transformer on this dataset.

### Leak-free pipeline (all preprocessing refit per fold)

| Stage | Detail |
|---|---|
| Target encoding | `TargetEncoder(cv=5)` by exact value over 13 columns → 39 features, **refit on training folds only** |
| Numeric tokens | 7 raw numerics + 39 TE features (PLR-embedded) |
| Categorical tokens | 6 categorical columns as learned embeddings |
| Scaling | `QuantileTransformer(normal)` fit per fold |
| Loss | Class-weighted cross-entropy (anti-imbalance lever) |
| Early stopping | On **balanced accuracy**, not loss |
| Training | AMP mixed precision, TF32, data on GPU — ~30 min on one GPU |
| CV | `StratifiedKFold(5, seed=42)`, single seed `[42]` |

### Model configuration

| Hyperparameter | Value |
|---|---|
| Token dimension | 128 |
| Blocks | 2 (8 heads) |
| Attention / FFN dropout | 0.2 / 0.1 |
| PLR frequencies `k` | 24 (init σ = 0.1) |
| Batch size | 4096 |
| LR / weight decay | 1e-3 / 1e-5 (10% warmup) |
| Max epochs | 16 (patience 4) |

The notebook also includes a **bias monitor** (per-class recall + row-normalized confusion matrix on OOF predictions) and a **decision-weight optimizer** — deliberately unused at submission, since with a class-weighted loss the plain argmax decision rule is already optimal.

## Results

- **LB balanced accuracy: 0.95059** — rank **89 / 3,355 (top 2.7%)** with a single FT-Transformer
- OOF balanced accuracy and per-class recall reported in the final cells

## Repository Structure

```
├── notebook/
│   └── ft-transformer-plr-embeddings-lb-0.95059.ipynb   # Full solution (pure PyTorch)
├── requirements.txt
└── README.md
```

## Reproduction

1. Join [Playground Series S6E7](https://www.kaggle.com/competitions/playground-series-s6e7) and download the data
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Update `DATA_DIR` in cell 1 to your local data path (defaults to the Kaggle input directory)
4. Run the notebook end-to-end — a single modern GPU finishes in ~30 minutes

## References

- Gorishniy et al., 2021 — *Revisiting Deep Learning Models for Tabular Data* (FT-Transformer)
- Gorishniy et al., 2022 — *On Embeddings for Numerical Features in Tabular Data* (PLR embeddings)

## Author

**Zhenrui Weng (翁振瑞)** — [Kaggle](https://www.kaggle.com/zhenruiweng) · [GitHub](https://github.com/ZhenruiWeng)
