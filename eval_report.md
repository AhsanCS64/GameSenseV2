# GameSense evaluation (real Steam Video Games / Tamber data)

- Raw rows: 200,000 | liked (>= 2.0h) after k-core filter: 32,464 pairs, 1,772 users, 1,007 games
- Games with embeddings: 993 | train/val/test users: 1417/177/178
- Protocol: held-out users, half of each user's games shown as seeds, the other half must be recovered
- Best model: embedding dim 128, weight decay 1e-06, early-stopped at epoch 14

## Test metrics (K=10)

| Model | Hit@10 | Recall@10 | NDCG@10 |
|---|---|---|---|
| Popularity | 0.7360 | 0.1898 | 0.2128 |
| Item-item cosine | 0.8708 | 0.2920 | 0.3398 |
| History-MF (ours) | 0.8146 | 0.2827 | 0.3324 |
