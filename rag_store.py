import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Set

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings


class RAGStore:
	"""
	Simple RAG storage backed by Chroma + sentence-transformers.

	- Stores combined user/assistant exchanges as documents.
	- Supports topic-aware retrieval for augmenting context.
	"""

	def __init__(self, persist_directory: Optional[str] = None):
		base_dir = persist_directory or os.path.join(
			os.path.dirname(os.path.abspath(__file__)), "chroma_store"
		)
		os.makedirs(base_dir, exist_ok=True)

		self._client = chromadb.PersistentClient(
			path=base_dir,
			settings=Settings(anonymized_telemetry=False),
		)
		self._collection = self._client.get_or_create_collection(
			name="foreverchat_exchanges",
			metadata={"hnsw:space": "cosine"},
		)

		self._embed_model_name = os.environ.get(
			"EMBED_MODEL",
			"sentence-transformers/all-MiniLM-L6-v2",
		)
		self._embed_model: Optional[SentenceTransformer] = None

	def _get_model(self) -> SentenceTransformer:
		if self._embed_model is None:
			self._embed_model = SentenceTransformer(self._embed_model_name)
		return self._embed_model

	def add_exchange(
		self,
		user_message: str,
		assistant_response: str,
		topics: Optional[List[str]] = None,
		model: Optional[str] = None,
	) -> None:
		"""Store a single exchange as a retrievable document."""
		text = f"User: {user_message}\nAssistant: {assistant_response}"
		doc_id = str(uuid.uuid4())
		now = datetime.utcnow().isoformat()

		metadata: Dict[str, Any] = {
			"created_at": now,
			"model": model or "",
		}
		if topics:
			metadata["topics"] = topics

		emb = self._get_model().encode(text).tolist()

		self._collection.add(
			ids=[doc_id],
			documents=[text],
			embeddings=[emb],
			metadatas=[metadata],
		)

	def search(
		self,
		query: str,
		topics: Optional[List[str]] = None,
		k: int = 4,
	) -> List[Dict[str, Any]]:
		"""Return top-k relevant past exchanges."""
		if not query.strip():
			return []

		query_emb = self._get_model().encode(query).tolist()

		where: Optional[Dict[str, Any]] = None
		if topics:
			# Match documents that have any overlap with requested topics.
			where = {"topics": {"$in": topics}}

		results = self._collection.query(
			query_embeddings=[query_emb],
			n_results=k,
			where=where,
		)

		docs = results.get("documents", [[]])[0]
		metas = results.get("metadatas", [[]])[0]
		distances = results.get("distances", [[]])[0]

		out: List[Dict[str, Any]] = []
		for doc, meta, dist in zip(docs, metas, distances):
			out.append(
				{
					"text": doc,
					"metadata": meta,
					"score": float(dist),
				}
			)
		return out


rag_store = RAGStore()


