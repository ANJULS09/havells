# Customer Voice Intelligence Agent (CVIA)

An enterprise-grade, multi-agent AI system designed for Havells product managers to ingest and analyze customer reviews. It features automated language translation, dynamic unsupervised theme clustering, aspect-based sentiment mapping (ABSA), and a Retrieval-Augmented Generation (RAG) question-answering workflow with strict anti-hallucination evidence auditing.

---

## Folder Structure

The project has been structured according to standard enterprise layouts:

```
customer-voice-intelligence/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                # FastAPI Application entry point
│   │   ├── config.py              # Settings configuration (Pydantic Settings)
│   │   ├── database.py            # SQLite/PostgreSQL connection layer
│   │   ├── models.py              # SQLAlchemy database entities
│   │   ├── schemas.py             # Pydantic schemas (validations)
│   │   ├── agents/                # Agentic workflows
│   │   │   ├── __init__.py
│   │   │   ├── base.py            # Base Agent
│   │   │   ├── ingestion.py       # Review Ingestion Agent (CSV/JSON stream parser)
│   │   │   ├── cleaning.py        # Text cleaning, language detection & translator
│   │   │   ├── theme_discovery.py # Topic Discovery (KMeans clustering + LLM labeler)
│   │   │   ├── sentiment.py       # Aspect Sentiment Agent (Aspect-Based Sentiment ABSA)
│   │   │   ├── trend.py           # Trend calculation & period aggregation (Pandas)
│   │   │   ├── retriever.py       # Hybrid semantic vector search retriever
│   │   │   ├── verification.py    # Grounding & citation auditor agent
│   │   │   └── answer_gen.py      # Answer Generation coordinator agent
│   │   └── services/
│   │       ├── __init__.py
│   │       ├── llm.py             # LLM API connection (Gemini, OpenAI, Ollama, Mock)
│   │       └── vector_store.py    # Vector database store (NumPy local, Qdrant client)
│   ├── evaluation/
│   │   └── evaluate.py            # Benchmark script for ABSA, RAG, and Grounding
│   ├── tests/
│   │   ├── __init__.py
│   │   └── test_agents.py         # Pytest unit tests for cleaning & vector indexing
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   └── app/
│   │       ├── page.tsx           # React Dashboard interface (with Recharts, Chat, Upload)
│   │       ├── layout.tsx
│   │       └── globals.css
│   ├── Dockerfile
│   └── package.json
├── docker-compose.yml             # Orchestration compose configurations
├── sample_data.json               # Seed reviews file for instant boots
├── .env                           # Local environment config variables
└── README.md
```

---

## Core System Architecture

### Agent Processing Workflow
```
[Review Upload]
      │
      ▼
[Ingestion Agent] ───────► (De-duplication Check)
      │
      ▼
[Cleaning Agent] ────────► (HTML strip, Emojis filter, langdetect)
      │
      ▼
[Translation Agent] ─────► (Translate non-English/Hinglish to English)
      │
      ▼
[Embedding Agent] ───────► (Generate text embeddings)
      │
      ▼
[Knowledge Store] ───────► (SQLite/PostgreSQL relational DB + NumPy/Qdrant Vector DB)
```

### Retrieval & Answer Generation Workflow
```
             [User Query]
                  │
                  ▼
          [Retriever Agent]
                  │
                  ▼
          [Semantic Search]
                  │
                  ▼
         [Retrieved Reviews]
                  │
                  ▼
         [Answer Gen Agent] ──► (Generate draft response citing review IDs)
                  │
                  ▼
      [Evidence Verify Agent] ─► (Audit groundedness & citation accuracy)
                  │
         ┌────────┴────────┐
         ▼                 ▼
  [Score >= 0.70]     [Score < 0.70]
         │                 │
         ▼                 ▼
 [Return Answer + Citations] ["There is insufficient evidence."]
```

---

## API Endpoints Design

| Method | Route | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/reviews/upload` | Upload review file (CSV/JSON), runs parser, translation, and indexers. |
| `POST` | `/api/v1/analysis/discover-themes` | Trigger MiniBatchKMeans clustering to auto-label feedback topics. |
| `POST` | `/api/v1/analysis/sentiment` | Batch aspect sentiment extractor (ABSA) for untagged reviews. |
| `GET` | `/api/v1/analysis/trends` | Fetch sentiment and volume trend data aggregated weekly, monthly, quarterly. |
| `POST` | `/api/v1/qa/query` | Ask natural language questions grounded strictly in customer reviews. |
| `GET` | `/api/v1/products` | Retrieve list of all ingested product lines. |
| `GET` | `/api/v1/themes` | Retrieve list of all active discovered themes. |
| `GET` | `/api/v1/reviews` | Paginated search list of reviews including ratings and aspect badges. |

---

## Setup & Running Guide

Ensure you have **Python 3.11+**, **Node.js 22+**, and **Docker** installed.

### Option 1: Running Locally (Fastest)

1. **Install and run the Backend**:
   ```bash
   # Go to backend folder
   cd backend
   # Install dependencies
   pip install -r requirements.txt
   # Launch FastAPI application (defaults to Mock LLM & SQLite Vector Index)
   python -m backend.app.main
   ```
   The backend server starts on: `http://localhost:8000`.

2. **Install and run the Frontend**:
   ```bash
   # Go to frontend folder
   cd ../frontend
   # Install package dependencies
   npm install
   # Run the Next.js development server
   npm run dev
   ```
   The PM Dashboard will be available at: `http://localhost:3000`.

3. **Try with seed data**:
   - Go to `http://localhost:3000`.
   - In the **Operations** card, select `Choose File` and select `sample_data.json` from the root project directory.
   - Click **Ingest**.
   - Click **Cluster Themes** to run the theme clustering agent.
   - Click **Extract Sentiment** to run the ABSA engine.
   - You can now visualize trends, browse reviews, and query the QA window!

### Option 2: Running with Docker Compose
To run the production-ready multi-service architecture locally:
```bash
docker-compose up --build
```
This boots up the FastAPI container (port 8000) and the Next.js container (port 3000) sharing network bridges and SQLite storage.

---

## Configuration Variables (`.env`)

Configure the LLM settings in the `.env` file at the project root:

- **Offline Mock Execution (Default)**:
  `LLM_PROVIDER=mock`
  No API keys needed. The system uses deterministic rule sets for testing.

- **Gemini Integration (Recommended)**:
  ```env
  LLM_PROVIDER=gemini
  GEMINI_API_KEY=your_gemini_api_key_here
  LLM_MODEL=gemini-1.5-flash
  ```

- **OpenAI Integration**:
  ```env
  LLM_PROVIDER=openai
  OPENAI_API_KEY=your_openai_key_here
  LLM_MODEL=gpt-4o
  ```

---

## Pipeline Quality Evaluation

We have integrated an automated evaluation bench in `backend/evaluation/evaluate.py`. It measures:
1. **Aspect sentiment accuracy (ABSA)**: Compares extracted sentiments against a test dataset.
2. **Precision & Recall @ K**: Measures retrieval relevance.
3. **Groundedness Verification**: Checks the hallucination suppression rate.

To run the pipeline benchmarks:
```bash
python -m backend.evaluation.evaluate
```

---

## Production Scalability Blueprint

For deploying this platform at scale (across the entire Havells product catalogue with millions of reviews):

1. **Celery Task Worker Queue**: Currently, the ingestion, theme discovery, and sentiment extraction run asynchronously within standard FastAPI request threads. For production scaling, move these processes to Celery workers backed by a Redis/RabbitMQ message broker.
2. **Qdrant Vector Cluster**: Swap out the default file-based vector index by setting `VECTOR_DB_TYPE=qdrant` and configuring the `VECTOR_DB_URL` in `.env` to connect to a distributed Qdrant cloud cluster with vector sharding.
3. **Incremental Indexing & Streaming Ingestion**: Set up an Apache Kafka or AWS Kinesis pipeline that listens to webhooks from e-commerce feeds (Amazon/Flipkart API). Reviews are streamed straight into a fast-insert ingestion queue (via Redis Streams) and processed incrementally.
4. **Caching Layer (Redis)**: Cache generated RAG answers and semantic search vectors to keep latency under 100ms for frequently asked business questions.
