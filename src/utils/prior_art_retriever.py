"""
Prior Art Retriever — domain-specific tool for Tomocube patent search.

Embeds similar_patent_bq abstracts and retrieves the most relevant prior art
for a given claim or patent description. This grounds 102/103 rejections
in actual prior art rather than LLM hallucination.
"""

import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("vpatent.retriever")

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
EMBED_DIM = 1024
PATENT_KB_DB = "/data1/patent-kb/data/patent_kb.db"


class PriorArtRetriever:
    """Retrieve relevant prior art from Tomocube patent KB."""

    def __init__(self, db_path: str = PATENT_KB_DB,
                 index_path: str = None,
                 model_name: str = EMBEDDING_MODEL,
                 device: str = "cuda:0"):
        import os
        self.db_path = db_path
        self.model_name = model_name
        self.device = os.environ.get("BGE_DEVICE", device)
        self._model = None
        self._index = None  # numpy array of embeddings
        self._metadata = []  # list of dicts with patent info

        if index_path is None:
            index_path = str(Path(__file__).resolve().parents[2] / "data" / "prior_art_index.npz")
        self.index_path = index_path

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("Model loaded on %s", self.device)
        return self._model

    def _load_patents(self) -> list[dict]:
        """Load patents from SQLite DB."""
        db = sqlite3.connect(self.db_path)
        db.row_factory = sqlite3.Row
        cursor = db.cursor()

        cursor.execute("""
            SELECT publication_number, title, abstract, ipc_codes, match_source
            FROM similar_patent_bq
            WHERE abstract IS NOT NULL AND abstract != ''
        """)
        patents = [dict(row) for row in cursor.fetchall()]
        db.close()
        logger.info("Loaded %d patents from DB", len(patents))
        return patents

    def build_index(self):
        """Build embedding index from patent abstracts."""
        patents = self._load_patents()
        model = self._load_model()

        # Build texts for embedding (title + abstract)
        texts = []
        metadata = []
        for p in patents:
            text = f"{p['title']}. {p['abstract']}"
            texts.append(text)
            metadata.append({
                "publication_number": p["publication_number"],
                "title": p["title"],
                "abstract": p["abstract"][:500],
                "ipc_codes": p.get("ipc_codes", ""),
            })

        logger.info("Encoding %d patents...", len(texts))
        # bge models recommend "Represent this sentence: " prefix for queries
        embeddings = model.encode(texts, show_progress_bar=True,
                                  batch_size=32, normalize_embeddings=True)

        # Save index
        np.savez_compressed(self.index_path,
                           embeddings=embeddings,
                           metadata=json.dumps(metadata, ensure_ascii=False))

        self._index = embeddings
        self._metadata = metadata
        logger.info("Index built and saved: %s (%d patents, dim=%d)",
                    self.index_path, len(metadata), embeddings.shape[1])
        return len(metadata)

    def load_index(self):
        """Load pre-built index."""
        if not os.path.exists(self.index_path):
            logger.info("Index not found, building...")
            return self.build_index()

        data = np.load(self.index_path, allow_pickle=True)
        self._index = data["embeddings"]
        self._metadata = json.loads(str(data["metadata"]))
        logger.info("Index loaded: %d patents", len(self._metadata))
        return len(self._metadata)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Search for prior art relevant to the query.

        Args:
            query: Claim text, patent abstract, or technical description
            top_k: Number of results to return

        Returns:
            List of dicts with patent info + similarity score
        """
        if self._index is None:
            self.load_index()

        model = self._load_model()

        # bge query prefix
        query_text = f"Represent this sentence for searching relevant documents: {query}"
        query_emb = model.encode([query_text], normalize_embeddings=True)

        # Cosine similarity (embeddings are normalized, so dot product = cosine)
        scores = np.dot(self._index, query_emb.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            result = dict(self._metadata[idx])
            result["similarity"] = float(scores[idx])
            results.append(result)

        return results

    def search_for_claims(self, claims: list[dict], top_k: int = 5) -> dict:
        """
        Search prior art for multiple claims, return aggregated results.

        Returns:
            Dict with per-claim results and unique refs list.
        """
        if self._index is None:
            self.load_index()

        all_refs = {}
        per_claim = {}

        for claim in claims:
            claim_text = claim.get("text", "")
            claim_num = claim.get("claim_number", 0)

            results = self.search(claim_text, top_k=top_k)
            per_claim[claim_num] = results

            for r in results:
                ref_id = r["publication_number"]
                if ref_id not in all_refs or r["similarity"] > all_refs[ref_id]["similarity"]:
                    all_refs[ref_id] = r

        return {
            "per_claim": per_claim,
            "unique_refs": sorted(all_refs.values(), key=lambda x: -x["similarity"]),
            "total_unique_refs": len(all_refs),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    retriever = PriorArtRetriever()

    # Build index
    n = retriever.build_index()
    print(f"\nIndex built: {n} patents")

    # Test search
    query = "Three-dimensional refractive index tomography using optical diffraction tomography with deep learning phase retrieval"
    print(f"\nQuery: {query[:80]}...")
    results = retriever.search(query, top_k=5)
    print(f"\nTop 5 results:")
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['similarity']:.3f}] {r['publication_number']}: {r['title'][:70]}")
