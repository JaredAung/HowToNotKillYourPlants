# Generated data

Shared outputs used across pipelines:

| File | Produced by | Consumed by |
|------|-------------|-------------|
| `synthetic_users.json` | `synthetic_user/generate_users.py` | Feast flow, training, eval |
| `two_tower_training/synthetic_interactions.json` | `synthetic_user/generate_interactions.py` | Training script, retrain, eval (DVC-tracked) |
