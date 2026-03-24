# Resources

Data pipelines, schemas, and training assets for the plant recommendation system.

---

## Quick start (pipeline order)

Run these steps in order. Each step produces outputs used by the next.

| Step | Command | Produces |
|------|---------|----------|
| **1. Plant data** | `python -m resources.data_creating.data_creating_pipeline gather-permapeople` | `data_creating/permapeople_plants_sample.json` |
| | `python -m resources.data_creating.data_creating_pipeline map-permapeople` | `data_creating/permapeople_plants_mapped*.json` |
| **2. Embed & upload plants** | `python -m resources.ETL.embed_and_upload` | MongoDB `NewPlantCollection` |
| **3. Synthetic users** | `python -m resources.synthetic_user.generate_users` | `data/synthetic_users.json` |
| **4. Synthetic interactions** | `python -m resources.synthetic_user.generate_interactions` | `data/synthetic_interactions.json` |
| **5. Feast features** | `python -m resources.ETL.flow` | Feast feature store |
| **6. Train model** | `python -m resources.two_tower_training.training_script` | `two_tower_training/output/two_tower.pt` |

---

## Folder layout

```
resources/
├── data/                     # Shared generated outputs (see data/README.md)
│   ├── synthetic_users.json
│   └── synthetic_interactions.json
│
├── data_creating/            # Plant data pipeline (Permapeople API → JSON)
│   ├── data_creating_pipeline.py   # gather + map + upload to MongoDB
│   ├── permapeople_plant.py        # Permapeople API client
│   ├── plant_data_clean.py         # Clean + embed plant descriptions
│   ├── upload.py                   # Merge profiles + embeddings → MongoDB
│   ├── interactions.py             # Legacy interaction generator
│   └── permapeople_plants_*.json
│
├── ETL/                      # Feast pipeline (MongoDB → feature store)
│   ├── flow.py               # Main Prefect flow (plants + users → Feast)
│   ├── feast_store.py        # Push parquet, materialize to Feast
│   ├── feature_engineer.py   # Categorical embeddings; water_freq_to_days (dry/moist/wet → days)
│   ├── mapping.py            # Plant field mappings
│   ├── embed_and_upload.py   # Load mapped JSON → embed → upload to MongoDB
│   ├── score.py              # Plant scoring utilities
│   └── vocabs.json, region_climate_map.json
│
├── synthetic_user/           # Synthetic user & interaction generation
│   ├── generate_users.py     # Create synthetic user profiles
│   ├── generate_interactions.py  # User–plant interactions with oracle labels
│   └── personas.json, vocabs.json
│
├── schema/                   # JSON schemas and field mappings
│   ├── plant_profiles_schema.json
│   ├── user_profile_schema.json
│   └── permapeople_plants_sample.schema.json
│
└── two_tower_training/       # Two-tower model training
    ├── training_script.py    # Train on Feast features, push embeddings to MongoDB
    ├── two_tower_model.py    # Model definition
    ├── two_tower_training.py # Legacy retrain pipeline
    └── output/               # Checkpoints, metrics, plant_embeddings.json
```

---

## Data flow

```
Permapeople API
      │
      ▼
data_creating/  ──► permapeople_plants_*.json
      │
      ▼
ETL/embed_and_upload  ──► MongoDB (NewPlantCollection)
      │
      ├──────────────────────────────────────┐
      │                                      │
      ▼                                      ▼
synthetic_user/generate_users  ──► data/synthetic_users.json
      │                                      │
      ▼                                      │
synthetic_user/generate_interactions  ──► data/synthetic_interactions.json
      │                                      │
      │                                      │
      ▼                                      ▼
ETL/flow  (plants from MongoDB + users from data/)  ──► Feast
      │
      ▼
two_tower_training/training_script  ──► two_tower.pt, plant_embeddings.json
```

---

## Naming notes

- **`data_creating/`** – Plant data pipeline (gather from Permapeople, map, upload).
- **`ETL/embed_and_upload.py`** – Loads mapped plant JSON, embeds descriptions, uploads to MongoDB. Distinct from `data_creating/` which gathers raw data.
