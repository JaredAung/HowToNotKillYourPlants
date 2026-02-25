# Vector Search Index Setup

The recommend pipeline uses MongoDB Atlas `$vectorSearch` for dot-product similarity.

## Create the index in Atlas

1. Atlas UI → Database → Browse Collections → **PlantCollection**
2. **Search Indexes** tab → **Create Search Index**
3. Choose **JSON Editor**, paste from `vector_index_definition.json`
4. Name the index **plant_tower_index** (or set `VECTOR_SEARCH_INDEX` in `.env`)

The index uses `dotProduct` to match the two-tower model's L2-normalized embeddings.
