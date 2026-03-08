# 🌱 How To Keep Your Plants Alive

**HowToKeepYourPlantsAlive** is an intelligent plant recommendation system designed to **match plants to a user's real environment and learn from plant failures**.

Unlike traditional plant apps that recommend plants using static filters or popularity, this system uses **machine learning, semantic reranking, and failure-aware learning** to continuously improve plant recommendations.

This project is the V2 of the project developed for **SFHacks 2026** 

V1 : [HowToNotKillYourIndoorPlants](https://github.com/JaredAung/HowToNotKillYourIndoorPlants).

---

# 🌿 Project TODO (for AI / development)

- [ ] **README:** demo video link, example recommendation output
- [ ] **Evaluation:** Precision@K, MAP, Coverage, Diversity; comparison table; % improvement vs baseline
- [ ] **Dashboard:** Streamlit — model metrics, system metrics, dataset stats, coverage, diversity
- [ ] **Architecture doc:** pipeline diagram (User Input → Feature Encoding → Vector Retrieval → Two-Tower → Reranker → LLM Explanation → Final Rec)
- [ ] **Dataset doc:** plant count, features, example record; user interactions (garden, deaths, search)
- [ ] **Death learning:** explain positives (garden adds), negatives (deaths), retrain usage
- [ ] **Demo video:** 2–3 min — preferences, results, LLM explanation, dashboard
- [ ] **GitHub:** architecture diagram, metrics table, dashboard screenshots, demo video
- [ ] **Future:** online learning, sequential history, cold-start, RL, disease detection

---

# 🚀 Key Innovations

### 🧠 ML Recommendation Engine

Uses a **Two-Tower neural network** to learn compatibility between **user environments** and **plant care requirements**.

### ⚡ Semantic Reranking

Improves recommendation quality using **Cohere semantic reranking**.

### 💀 Failure-Aware Recommendations

The system **learns from plants that died** and penalizes similar plants in future recommendations.

### 🤖 Conversational Plant Assistant

An **LLM-powered assistant** built with **LangGraph** allows users to explore, compare, and add plants using natural language.

### 🌿 Environment-Aware Profiles

Recommendations are based on **real user conditions**, including:

* Light availability
* Humidity
* Temperature
* Watering habits
* Care difficulty tolerance

---

# 🏗 System Architecture

```mermaid
flowchart TD

User[User Profile]
PlantDB[(Plant Database)]

User --> UserEmbedding
PlantDB --> PlantEmbedding

UserEmbedding[User Tower]
PlantEmbedding[Plant Tower]

UserEmbedding --> VectorSearch
PlantEmbedding --> VectorSearch

VectorSearch[MongoDB Vector Search]

VectorSearch --> Reranker

Reranker[Cohere Reranker]

Reranker --> DeathPenalty

DeathPenalty[Death Similarity Penalty]

DeathPenalty --> Results[Final Recommendations]
```

---

# 🧠 Two-Tower Model Architecture

```mermaid
flowchart LR

subgraph UserTower[User Tower]
    U_Light[Light]
    U_Hum[Humidity]
    U_Temp[Temp]
    U_Water[Watering]
    U_Care[Care Level]
    U_Light --> U_Embed[64-d Embedding]
    U_Hum --> U_Embed
    U_Temp --> U_Embed
    U_Water --> U_Embed
    U_Care --> U_Embed
end

subgraph PlantTower[Plant Tower]
    P_Light[Light Req]
    P_Hum[Humidity Req]
    P_Temp[Temp Req]
    P_Water[Water Req]
    P_Care[Care Level]
    P_Light --> P_Embed[64-d Embedding]
    P_Hum --> P_Embed
    P_Temp --> P_Embed
    P_Water --> P_Embed
    P_Care --> P_Embed
end

U_Embed --> DotProduct[Dot Product]
P_Embed --> DotProduct
DotProduct --> Score[Similarity Score]
```

---

# 🔄 Death-Learning Feedback Loop

```mermaid
flowchart TD

User[User adds plant to garden]
User --> Garden[(Garden)]
Garden --> PlantDies[Plant dies]
PlantDies --> DeathReport[Death Report Form]
DeathReport --> DeathDB[(PlantDeathCollection)]
DeathDB --> Penalty[Death Penalty in Recommendations]
Penalty --> AvoidSimilar[Avoid similar plants]
AvoidSimilar --> BetterRecs[Better recommendations]
BetterRecs --> User
```

---

# 🧠 Recommendation Pipeline

The recommendation system operates in **three stages**.

---

## 1️⃣ Two-Tower Model (Candidate Retrieval)

A **two-tower deep learning model** embeds both **users** and **plants** into the same vector space.

### User Features

* Light conditions
* Humidity
* Temperature
* Watering preference
* Care difficulty tolerance

### Plant Features

* Light requirements
* Water requirements
* Humidity tolerance
* Temperature tolerance
* Care level

Both towers output **64-dimensional embeddings**.

MongoDB **vector search** retrieves candidate plants using **dot-product similarity**.

---

## 2️⃣ Semantic Reranking

The candidate list is reranked using the **Cohere Reranker**, which evaluates semantic relevance between:

* User environment description
* Plant descriptions

This improves ranking quality beyond structured matching.

---

## 3️⃣ Failure-Aware Learning (Death Penalty)

If a plant dies, the system **learns from that failure**.

Plants similar to previously dead plants receive a **score penalty**.

```
final_score = base_score − λ × similarity_to_dead_plants
```

This prevents recommending plants that **historically failed for the user**.

---

# 📊 Evaluation Metrics

**Current metrics:**

- [ ] Recall@K
- [ ] NDCG@K
- [ ] Hit Rate
- [ ] Latency (mean, p95)

**Add additional metrics:**

- [ ] Precision@K
- [ ] MAP (Mean Average Precision)
- [ ] Coverage
- [ ] Diversity

**Comparison table:**

| Model             | Recall@5 | Recall@10 | Recall@20 | NDCG@5 | NDCG@10 | NDCG@20 | Hit@5 | Hit@10 | Hit@20 | Latency (mean) | Latency (p95) |
| ----------------- | -------- | --------- | --------- | ------ | ------- | ------- | ----- | ------ | ------ | -------------- | ------------- |
| Semantic Baseline | 0.019    | 0.060     | 0.140     | 0.162  | 0.175   | 0.192   | 0.29  | 0.48   | 0.74   | 135 ms         | 197 ms        |
| Rec Pipeline      | 0.292    | 0.504     | 0.785     | 0.642  | 0.693   | 0.751   | 0.90  | 0.97   | 1.00   | 3000 ms        | 4060 ms      |

**% improvement vs baseline (Rec Pipeline):**

| Metric     | @5    | @10   | @20   |
| ---------- | ----- | ----- | ----- |
| Recall     | +1413% | +738% | +461% |
| NDCG       | +296% | +296% | +291% |
| Hit Rate   | +211% | +100% | +35%  |

---


# 🤖 LLM Chat Assistant

The application includes a **conversational assistant** built with **LangGraph**.

The assistant routes user intents to specialized actions.

### Supported Actions

| Action      | Description                              |
| ----------- | ---------------------------------------- |
| **EXPAND**  | Learn detailed information about a plant |
| **COMPARE** | Compare multiple plants                  |
| **PICK**    | Add a plant to your garden               |

---


# 🌿 Application Features

### Personalized Recommendations

Machine learning pipeline generates **environment-aware plant suggestions**.

### Home Feed

Displays **top plant recommendations** with optional AI explanations.

### Garden Tracking

Users can add plants to their personal garden and track them.

### Plant Care Profiles

Each plant includes detailed care information:

* Light
* Water
* Humidity
* Temperature
* Care difficulty

### Natural Language Search

Users can search using descriptions like:

> "Small plant that survives low light and doesn't need frequent watering."

The system extracts environmental constraints and returns matching plants.

### Death Reporting System

Users can report plant deaths with contextual data.

Fields include:

* What happened
* Watering frequency
* Plant location
* Humidity
* Room temperature

Death reports expire after **30 days using MongoDB TTL indexes**.

---

# 🧩 Technology Stack

| Layer               | Technologies                      |
| ------------------- | --------------------------------- |
| **Frontend**        | Next.js 16, React 19, TailwindCSS |
| **Backend**         | FastAPI                           |
| **Database**        | MongoDB Atlas                     |
| **Authentication**  | JWT                               |
| **ML Model**        | PyTorch Two-Tower Network         |
| **Embeddings**      | Voyage AI                         |
| **Reranking**       | Cohere                            |
| **LLM Framework**   | LangChain + LangGraph             |
| **LLM Runtime**     | Ollama                            |
| **External Search** | Tavily                            |

---

# 📂 Project Structure

```
HowToKeepYourPlantsAlive
│
├── backend
│   ├── auth
│   ├── chat
│   ├── garden
│   ├── plant
│   ├── profile
│   ├── recommend
│   ├── search
│   ├── database
│   └── schemas
│
├── frontend
│   └── app
│       ├── auth
│       ├── garden
│       ├── plant
│       ├── profile
│       ├── onboarding
│       ├── agent
│       ├── search
│       └── chat
│
├── resources
│   ├── two_tower_training
│   ├── data_creating
│   └── schema
│
└── .env
```

---

# ⚙️ Setup

## Environment Variables

Create `.env` in the project root:

```env
# Required
MONGO_URI=mongodb+srv://...
MONGO_DATABASE=HowNotToKillYourPlants
JWT_SECRET=your-secret

# Collections (optional)
MONGO_USER_PROFILES_COLLECTION=UserCollection
MONGO_USER_GARDEN_COLLECTION=User_Garden_Collection
PLANT_DEATH_COLLECTION=PlantDeathCollection
PLANT_MONGO_COLLECTION=PlantCollection

# ML & APIs
VOYAGE_API_KEY=...
COHERE_API_KEY=...
TAVILY_API_KEY=...
VECTOR_SEARCH_INDEX=vector_index

# Optional
USE_RERANK=true
USE_DEATH_PENALTY=true
DEATH_PENALTY_LAMBDA=0.5
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## MongoDB Vector Index

Create a vector search index on `PlantCollection`:

1. Atlas → Database → PlantCollection → Search Indexes
2. Create index (JSON editor) from `resources/vector_index_definition.json`
3. Index name must match `VECTOR_SEARCH_INDEX` (default: `vector_index`)

See `resources/VECTOR_INDEX_SETUP.md` for details.

---

## Backend

```bash
cd backend
pip install -r requirements.txt
```

---

## Train the Model

```bash
python resources/two_tower_training/two_tower_training.py
```

Outputs:

* `two_tower.pt`
* `plant_embeddings.json`

---

## Upload Plant Data

```bash
python resources/data_creating/plant_data_clean.py
python resources/data_creating/upload.py
```

---

## Run Backend

```bash
cd backend
uvicorn main:app --reload
```

Backend runs at: **http://localhost:8000**

---

## Run Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at: **http://localhost:3000**

---

