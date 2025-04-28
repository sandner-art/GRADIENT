**SQLite (via SQLAlchemy)** for structured metadata/session state and **LanceDB** for local vector storage (specifically for RAG) 
Here's the rationale and how they fit:

**1. SQLite (via SQLAlchemy)**

*   **Purpose:** Store structured, relational data about experiments, presets, settings, potentially user annotations, queued tasks, etc.
*   **Why SQLite?**
    *   **Serverless:** It's a simple file-based database. No separate database server process is needed, making setup and deployment for a local desktop application trivial. The database is just a `.db` file (e.g., `data/gradient.db`).
    *   **Standard SQL:** Uses standard SQL syntax.
    *   **Mature & Robust:** Extremely well-tested and reliable for single-user or low-concurrency applications.
    *   **Good Python Integration:** Excellent libraries exist, especially SQLAlchemy.
*   **Why SQLAlchemy?**
    *   **ORM (Object-Relational Mapper):** Allows you to define your database tables as Python classes (models) and interact with the database using Python objects instead of writing raw SQL strings. This makes database interactions cleaner, more Pythonic, and less prone to SQL injection errors (though less of a concern for a local app).
    *   **Database Agnosticism (Partial):** While using specific SQLite features might limit portability, SQLAlchemy provides an abstraction layer that makes potentially migrating to another database (like PostgreSQL) easier in the future if needed.
    *   **Migration Tools:** Alembic (often used with SQLAlchemy) can help manage database schema changes over time as your application evolves.
*   **Data Stored Here:**
    *   `ExperimentRuns`: Run ID, timestamp, base parameters (foreign key or JSON), status (running, completed, failed), duration.
    *   `GeneratedOutputs`: Output ID, associated Run ID, file path (image, data), type (image, plot, latent), specific parameters for this output (seed, etc.), metadata extracted (if any).
    *   `ParameterPresets`: Preset name, parameter dictionary (JSON).
    *   `Workflows`: Workflow name, definition (JSON/YAML).
    *   `ModelRegistry` (Optional): Information about locally discovered models (path, type, metadata).
    *   `TaskQueue` (Optional): State of background tasks.
    *   `UserAnnotations` (Optional): Tags, ratings applied to outputs/runs.

**2. LanceDB**

*   **Purpose:** Store and efficiently query vector embeddings, primarily for **Retrieval-Augmented Generation (RAG)** functionality, but potentially also for semantic image search or finding similar experiments later.
*   **Why LanceDB?**
    *   **Built for Vector Search:** Optimized specifically for Approximate Nearest Neighbor (ANN) search on large vector datasets.
    *   **Serverless & Embeddable:** Like SQLite, it runs directly within your application process using local files. No separate server needed. This is crucial for a local-first application.
    *   **Python Native:** Designed with Python in mind.
    *   **Efficient Storage & Querying:** Uses the Lance file format, which is optimized for analytical queries and vector search on disk (memory-mapping), reducing RAM requirements compared to purely in-memory vector databases. Supports filtering alongside vector search.
    *   **Developer Experience:** Generally considered easy to integrate and use.
*   **Best Use - RAG:**
    *   **Scenario:** You want an LLM (like the Hermes model you loaded) to answer questions or generate text based on a specific set of documents (e.g., research papers, project notes, documentation) provided by the user.
    *   **Workflow:**
        1.  **Indexing:** The user points GRADIENT to a folder of documents (PDF, TXT, MD, etc.).
        2.  GRADIENT uses a text processing library (e.g., `langchain`, `llama-index`, or custom code) to chunk the documents into smaller, manageable pieces.
        3.  An **embedding model** (like `all-MiniLM-L6-v2`, `bge-base-en-v1.5`, etc., running locally via `sentence-transformers` or potentially a larger model) converts each text chunk into a numerical vector embedding.
        4.  These embeddings, along with the original text chunk and metadata (source document, chunk ID), are stored in a LanceDB table.
        5.  **Retrieval:** When the user asks the LLM a question relevant to the documents:
            *   The user's query is converted into a vector embedding using the *same* embedding model.
            *   LanceDB performs an efficient vector similarity search (ANN) to find the text chunk embeddings in the database that are most semantically similar to the query embedding.
            *   The corresponding original text chunks for the top N most similar embeddings are retrieved.
        6.  **Augmentation:** The retrieved text chunks are formatted and added as context to the prompt being sent to the main LLM (along with the user's original query).
        7.  **Generation:** The LLM generates an answer based on both the user's query *and* the retrieved context from the documents.
    *   **Benefits:** Allows LLMs to access and reason about specific, up-to-date, or private information not present in their original training data, significantly improving factual accuracy and relevance for specific domains. LanceDB makes the crucial retrieval step fast and efficient locally.
*   **Other Potential Uses:**
    *   **Semantic Search of Experiments:** Embed prompts or descriptions of experiment runs and find runs with similar goals.
    *   **Image Similarity Search:** Store image embeddings (e.g., from CLIP) generated by GRADIENT and allow searching for visually similar generated images.

**Implementation Plan Integration:**

*   **Phase 1 Refinement / Phase 2:** Introduce SQLAlchemy and the basic Experiment DB schema. Modify generation functions to log to this DB. Introduce metadata saving to images.
*   **Phase 3 / Phase 4:** Introduce LanceDB integration specifically when implementing RAG capabilities for the LLM or other semantic search features. This involves adding `sentence-transformers` dependency, text chunking logic, embedding generation, and the LanceDB query code.

This combination provides a robust, local-first approach to managing both structured metadata and unstructured vector data efficiently within GRADIENT.