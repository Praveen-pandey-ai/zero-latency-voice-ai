Recreate docs.npy

This file originally contained a precomputed numpy array of embeddings/docs used by the RAG demo.

To recreate the file run the ingestion script which will rebuild the index from source documents:

```bash
python rag/ingest.py
```

If you don't want to store large precomputed artifacts in the repo, keep this file removed and run the ingest step at runtime.
