# 🚀 Deployment Guide — Step by Step

Deploy **HowToKeepYourPlantsAlive** to production. Recommended stack:

| Component | Platform |
|-----------|----------|
| Frontend | Vercel |
| Backend | Railway |
| Database | MongoDB Atlas (already cloud) |

---

## Prerequisites

- [ ] GitHub repo pushed (code ready)
- [ ] MongoDB Atlas cluster with data (plants, vector index)
- [ ] Model file: `resources/two_tower_training/output/two_tower.pt` (run `dvc pull` if needed)
- [ ] API keys: GEMINI_API_KEY, COHERE_API_KEY, VOYAGE_API_KEY

---

## Step 1: Deploy Backend (Railway)

### 1.1 Create Railway project

1. Go to [railway.app](https://railway.app) and sign in with GitHub
2. **New Project** → **Deploy from GitHub repo**
3. Select your `HowToKeepYourPlantsAlive` repo
4. Railway will detect the project

### 1.2 Configure the service

1. In the project, click the service
2. **Settings** → **Root Directory**: leave empty (project root)
3. The project includes a `Dockerfile` — Railway will use it automatically
4. **No custom Start Command** needed (Dockerfile CMD handles it)

### 1.3 Model file before deploy

The model `two_tower.pt` is DVC-tracked (not in git). Options:

**Option A — Include in image (simplest):**
1. Locally: `dvc pull` to fetch the model
2. Temporarily copy the model into the repo (or use `dvc export`) so it's included in the Docker build
3. Or add a build step in Dockerfile: `RUN pip install "dvc[gdrive]" && dvc pull` (requires DVC remote config + credentials as build secrets)

**Option B — Railway volume (if supported):**
Upload the model to a persistent volume and mount it at `resources/two_tower_training/output/`.

For a first deploy, Option A with a manual copy is often easiest.

### 1.4 Set environment variables

In Railway → **Variables**, add:

| Variable | Value |
|----------|-------|
| `MONGO_URI` | Your MongoDB Atlas connection string |
| `MONGO_DATABASE` | `HowNotToKillYourPlants` |
| `JWT_SECRET` | New random secret (e.g. `openssl rand -hex 32`) |
| `GEMINI_API_KEY` | Your Gemini API key |
| `COHERE_API_KEY` | Your Cohere API key |
| `VOYAGE_API_KEY` | Your Voyage API key |
| `TAVILY_API_KEY` | Your Tavily API key |
| `VECTOR_SEARCH_INDEX` | `vector_index` |
| `CORS_ORIGINS` | `https://your-app.vercel.app` (update after Step 2) |

### 1.5 Deploy and get URL

1. Click **Deploy**
2. After build, go to **Settings** → **Networking** → **Generate Domain**
3. Copy the URL, e.g. `https://howtokeepyourplantsalive-production.up.railway.app`

---

## Step 2: Deploy Frontend (Vercel)

### 2.1 Create Vercel project

1. Go to [vercel.com](https://vercel.com) and sign in with GitHub
2. **Add New** → **Project**
3. Import your `HowToKeepYourPlantsAlive` repo

### 2.2 Configure build

1. **Root Directory**: `frontend`
2. **Framework Preset**: Next.js (auto-detected)
3. **Build Command**: `npm run build` (default)
4. **Output Directory**: `.next` (default)

### 2.3 Set environment variables

In Vercel → **Settings** → **Environment Variables**:

| Variable | Value |
|----------|-------|
| `NEXT_PUBLIC_API_URL` | Your Railway backend URL (e.g. `https://xxx.up.railway.app`) |

### 2.4 Deploy

1. Click **Deploy**
2. After build, copy your frontend URL, e.g. `https://howtokeepyourplantsalive.vercel.app`

### 2.5 Update backend CORS

Go back to Railway → **Variables** and set:

| Variable | Value |
|----------|-------|
| `CORS_ORIGINS` | `https://howtokeepyourplantsalive.vercel.app` (your actual Vercel URL) |

Redeploy the backend so the new CORS value is applied.

---

## Step 3: Model file

The backend needs `resources/two_tower_training/output/two_tower.pt`.

### Option A: Include in repo (not recommended for large files)

If the model is small and tracked in git, it will deploy with the code.

### Option B: DVC + build step

1. Locally: `dvc pull` to fetch the model
2. Ensure the model is present before building the Docker image
3. Or add a Railway build step that runs `dvc pull` (requires DVC config and credentials)

### Option C: Bake into image

1. Run `dvc pull` locally
2. Commit the model (or use a private artifact store)
3. Build the Docker image; the model will be in the image

For a first deploy, Option C (run `dvc pull` locally, then build) is usually simplest.

---

## Step 4: Verify deployment

### 4.1 Backend

```bash
curl https://your-backend.railway.app/health
# Expected: {"status":"healthy"}
```

### 4.2 Frontend

1. Open your Vercel URL
2. Sign up / log in
3. Check recommendations, search, and chat

### 4.3 Common issues

| Issue | Fix |
|-------|-----|
| CORS error in browser | Set `CORS_ORIGINS` to your exact Vercel URL (no trailing slash) |
| 401 on API calls | Check JWT_SECRET is set and consistent |
| No recommendations | Ensure MongoDB has plants with `plant_tower_embedding`, vector index exists |
| Model not found | Run `dvc pull` and rebuild, or add model to image |

---

## Step 5: Optional — custom domains

### Backend (Railway)

1. **Settings** → **Networking** → **Custom Domain**
2. Add e.g. `api.yourdomain.com`
3. Update DNS as instructed

### Frontend (Vercel)

1. **Settings** → **Domains** → **Add**
2. Add e.g. `app.yourdomain.com`
3. Update DNS as instructed

Remember to update `NEXT_PUBLIC_API_URL` and `CORS_ORIGINS` if you change domains.

---

## Checklist before going live

- [ ] Backend health check returns 200
- [ ] Frontend loads and can sign up / log in
- [ ] Recommendations load
- [ ] Chat works (Gemini)
- [ ] Search works
- [ ] Garden add / death report work
- [ ] No secrets in git
- [ ] `CORS_ORIGINS` matches frontend URL exactly
