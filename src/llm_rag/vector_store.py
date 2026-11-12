import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss
from config.settings import settings
from config.logging_config import logger

class MacroVectorStore:
    """
    High-Performance Institutional Vector Store powered by FAISS.
    Indexes Central Bank speeches, FOMC policy minutes, and SEC Filings
    with L2-normalized dense embeddings for exact cosine similarity ranking.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index: Optional[faiss.Index] = None
        self.documents: List[Dict[str, Any]] = []
        self.model = None
        self._init_embedding_model()

    def _init_embedding_model(self):
        """
        Loads SentenceTransformer if available; otherwise uses deterministic dense projection.
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.dimension = 384
            logger.info("Loaded SentenceTransformer 'all-MiniLM-L6-v2' for RAG embeddings.")
        except Exception as e:
            logger.warning(f"SentenceTransformer offline/unavailable: {e}. Utilizing high-dimensional TF-IDF projection.")
            self.model = None
            self.dimension = 384

    def _compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Encodes list of text chunks into normalized float32 embedding matrix.
        """
        if self.model is not None:
            try:
                embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
                # L2 normalize for cosine similarity via inner product
                faiss.normalize_L2(embeddings)
                return embeddings.astype(np.float32)
            except Exception as e:
                logger.warning(f"Model encode failed, falling back to TF-IDF projection: {e}")

        # Fallback: Scikit-learn TF-IDF with TruncatedSVD projection
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        vec = TfidfVectorizer(max_features=2000, stop_words="english")
        tfidf_mat = vec.fit_transform(texts).toarray()

        n_samples = len(texts)
        n_comp = min(self.dimension, n_samples, tfidf_mat.shape[1])
        if n_comp > 1 and tfidf_mat.shape[1] > 1:
            svd = TruncatedSVD(n_components=n_comp, random_state=settings.RANDOM_SEED)
            reduced = svd.fit_transform(tfidf_mat)
            if reduced.shape[1] < self.dimension:
                pad = np.zeros((n_samples, self.dimension - reduced.shape[1]))
                emb = np.hstack([reduced, pad])
            else:
                emb = reduced[:, :self.dimension]
        else:
            emb = np.random.default_rng(settings.RANDOM_SEED).normal(0, 0.1, (n_samples, self.dimension))

        emb = emb.astype(np.float32)
        faiss.normalize_L2(emb)
        return emb

    def build_index(self, doc_chunks: List[Dict[str, Any]]) -> int:
        """
        Builds FAISS index from document chunks.
        Each chunk is a dict: {"id": str, "title": str, "source": str, "content": str, "category": str, "date": str}
        """
        if not doc_chunks:
            logger.warning("Empty document list supplied to vector store.")
            return 0

        self.documents = doc_chunks
        texts = [f"{doc.get('title', '')}: {doc.get('content', '')}" for doc in doc_chunks]
        embeddings = self._compute_embeddings(texts)

        # IndexFlatIP calculates inner product = cosine similarity on normalized vectors
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)

        logger.info(f"FAISS index built with {self.index.ntotal} vectors (dim={self.dimension}).")
        return self.index.ntotal

    def search(
        self,
        query: str,
        top_k: int = 4,
        source_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes semantic search against the FAISS index.
        Returns top-k documents with cosine similarity score and metadata.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        q_emb = self._compute_embeddings([query])
        # Retrieve extra if filtering is applied
        retrieve_k = min(self.index.ntotal, top_k * 3 if source_filter else top_k)
        scores, indices = self.index.search(q_emb, retrieve_k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0 or idx >= len(self.documents):
                continue
            doc = self.documents[idx].copy()
            if source_filter and source_filter.lower() not in doc.get("source", "").lower():
                continue

            doc["similarity_score"] = round(float(score), 4)
            doc["rank"] = rank + 1
            results.append(doc)
            if len(results) >= top_k:
                break

        return results
