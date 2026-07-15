import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from backend.app.config import settings

logger = logging.getLogger("app.vector_store")

class VectorStoreService:
    def __init__(self):
        self.db_type = settings.VECTOR_DB_TYPE.lower()
        self.local_path = "voice_cve_vectors.json"
        self.vectors = {}  # review_id -> list of float
        self.payloads = {}  # review_id -> metadata dict
        
        if self.db_type == "local":
            self._load_local_store()
        else:
            # Qdrant client initialization could go here.
            # We fall back to local if qdrant is not importable or missing config.
            logger.info("Qdrant requested, verifying client...")
            try:
                import qdrant_client
                self.qclient = qdrant_client.QdrantClient(
                    url=settings.VECTOR_DB_URL,
                    api_key=settings.VECTOR_DB_API_KEY
                )
                logger.info("Successfully connected to Qdrant.")
            except Exception as e:
                logger.warning(f"Failed to connect to Qdrant: {e}. Falling back to local vector store.")
                self.db_type = "local"
                self._load_local_store()

    def _load_local_store(self):
        if os.path.exists(self.local_path):
            try:
                with open(self.local_path, "r") as f:
                    data = json.load(f)
                    self.vectors = data.get("vectors", {})
                    self.payloads = data.get("payloads", {})
                logger.info(f"Loaded {len(self.vectors)} vectors from local store: {self.local_path}")
            except Exception as e:
                logger.error(f"Error loading local vector store: {e}")
                self.vectors = {}
                self.payloads = {}
        else:
            logger.info("Created new local vector store.")

    def _save_local_store(self):
        try:
            with open(self.local_path, "w") as f:
                json.dump({
                    "vectors": self.vectors,
                    "payloads": self.payloads
                }, f)
            logger.info(f"Saved {len(self.vectors)} vectors to local store.")
        except Exception as e:
            logger.error(f"Error saving local vector store: {e}")

    async def upsert_reviews(
        self, 
        review_ids: List[str], 
        embeddings: List[List[float]], 
        payloads: List[Dict[str, Any]]
    ):
        """
        Upsert reviews and their vectors into the database.
        """
        if not review_ids or not embeddings:
            return

        if self.db_type == "local":
            for idx, r_id in enumerate(review_ids):
                self.vectors[r_id] = embeddings[idx]
                self.payloads[r_id] = payloads[idx]
            self._save_local_store()
        else:
            try:
                # Setup Qdrant collection if not exists
                from qdrant_client.models import Distance, VectorParams, PointStruct
                # Determine vector size from first embedding
                vector_size = len(embeddings[0])
                
                # Check collection
                collections = self.qclient.get_collections().collections
                col_names = [c.name for c in collections]
                if settings.VECTOR_DB_COLLECTION not in col_names:
                    self.qclient.create_collection(
                        collection_name=settings.VECTOR_DB_COLLECTION,
                        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                    )
                
                # Write points
                points = []
                for idx, r_id in enumerate(review_ids):
                    points.append(PointStruct(
                        id=r_id,
                        vector=embeddings[idx],
                        payload=payloads[idx]
                    ))
                
                self.qclient.upsert(
                    collection_name=settings.VECTOR_DB_COLLECTION,
                    points=points
                )
            except Exception as e:
                logger.error(f"Error upserting to Qdrant: {e}, falling back to local storage.")
                # Fallback to local
                for idx, r_id in enumerate(review_ids):
                    self.vectors[r_id] = embeddings[idx]
                    self.payloads[r_id] = payloads[idx]
                self._save_local_store()

    async def get_embeddings_for_reviews(self, review_ids: List[str]) -> Dict[str, List[float]]:
        """
        Retrieve stored vectors for a list of review IDs.
        """
        if self.db_type == "local":
            return {rid: self.vectors[rid] for rid in review_ids if rid in self.vectors}
        else:
            try:
                result = self.qclient.retrieve(
                    collection_name=settings.VECTOR_DB_COLLECTION,
                    ids=review_ids,
                    with_vectors=True
                )
                return {hit.id: hit.vector for hit in result if hit.vector}
            except Exception as e:
                logger.error(f"Error retrieving vectors from Qdrant: {e}")
                return {}

    async def search_reviews(
        self, 
        query_vector: List[float], 
        filter_metadata: Optional[Dict[str, Any]] = None, 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for reviews matching query_vector and metadata filters.
        """
        if not query_vector:
            return []

        if self.db_type == "local":
            # Perform Cosine Similarity in NumPy
            results = []
            if not self.vectors:
                return []

            q_vec = np.array(query_vector)
            q_norm = np.linalg.norm(q_vec)
            
            for r_id, emb in self.vectors.items():
                payload = self.payloads.get(r_id, {})
                
                # Metadata filtering
                if filter_metadata:
                    skip = False
                    for key, val in filter_metadata.items():
                        if val is not None:
                            # Handle lists (e.g. list of ratings)
                            if isinstance(val, list):
                                if payload.get(key) not in val:
                                    skip = True
                                    break
                            # Handle ranges (e.g., date range check)
                            elif key == "start_date" and payload.get("date"):
                                if payload.get("date") < val:
                                    skip = True
                                    break
                            elif key == "end_date" and payload.get("date"):
                                if payload.get("date") > val:
                                    skip = True
                                    break
                            elif payload.get(key) != val:
                                skip = True
                                break
                    if skip:
                        continue
                
                e_vec = np.array(emb)
                e_norm = np.linalg.norm(e_vec)
                
                if q_norm > 0 and e_norm > 0:
                    similarity = float(np.dot(q_vec, e_vec) / (q_norm * e_norm))
                else:
                    similarity = 0.0
                
                results.append({
                    "id": r_id,
                    "score": similarity,
                    "payload": payload
                })
            
            # Sort by score descending
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
        else:
            try:
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                
                # Build Qdrant filter
                qfilters = []
                if filter_metadata:
                    for key, val in filter_metadata.items():
                        if val is not None:
                            if isinstance(val, list):
                                # Qdrant supports MatchValue inside individual condition
                                # Or MatchAny for multiple matches
                                from qdrant_client.models import MatchAny
                                qfilters.append(FieldCondition(
                                    key=f"metadata.{key}",
                                    match=MatchAny(any=val)
                                ))
                            else:
                                qfilters.append(FieldCondition(
                                    key=f"metadata.{key}",
                                    match=MatchValue(value=val)
                                ))
                
                qfilter = Filter(must=qfilters) if qfilters else None
                
                search_result = self.qclient.search(
                    collection_name=settings.VECTOR_DB_COLLECTION,
                    query_vector=query_vector,
                    query_filter=qfilter,
                    limit=top_k
                )
                
                return [{
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload
                } for hit in search_result]
            except Exception as e:
                logger.error(f"Error querying Qdrant: {e}, falling back to local index.")
                # Run query on local fallback index
                self.db_type = "local"
                self._load_local_store()
                return await self.search_reviews(query_vector, filter_metadata, top_k)

vector_store_service = VectorStoreService()
