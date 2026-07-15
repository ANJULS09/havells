from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from backend.app.agents.base import BaseAgent
from backend.app.models import Review, Product
from backend.app.services.llm import llm_service
from backend.app.services.vector_store import vector_store_service

class RetrieverAgent(BaseAgent):
    def __init__(self):
        super().__init__("RetrieverAgent")

    async def retrieve_relevant_reviews(
        self,
        db: Session,
        question: str,
        product_id: Optional[str] = None,
        category: Optional[str] = None,
        rating_filter: Optional[List[int]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        top_k: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top K reviews relevant to the question, matching specified filters.
        """
        self.log_info(f"Retrieving reviews for question: '{question}'")
        
        # 1. Embed query text
        try:
            query_embeddings = await llm_service.get_embeddings([question])
            if not query_embeddings:
                self.log_error("Failed to generate embedding for query.")
                return []
            query_vector = query_embeddings[0]
        except Exception as e:
            self.log_error(f"Embedding query exception: {e}")
            return []

        # 2. Setup Vector DB metadata filter
        # Keep keys matching the schema of payload inserted during ingestion
        vector_filter = {}
        if product_id:
            vector_filter["product_id"] = product_id
        if category:
            vector_filter["category"] = category
        if rating_filter:
            vector_filter["rating"] = rating_filter
        if start_date:
            vector_filter["start_date"] = start_date.isoformat()
        if end_date:
            vector_filter["end_date"] = end_date.isoformat()

        # 3. Perform semantic vector search
        search_hits = await vector_store_service.search_reviews(
            query_vector=query_vector,
            filter_metadata=vector_filter,
            top_k=top_k
        )

        if not search_hits:
            self.log_warning("No matching reviews returned from vector search.")
            return []

        # 4. Fetch full review documents and relations from SQL
        retrieved_reviews = []
        for hit in search_hits:
            review_id = hit["id"]
            score = hit["score"]
            
            review = db.query(Review).join(Product).filter(Review.id == review_id).first()
            if review:
                retrieved_reviews.append({
                    "review_id": review.id,
                    "product_name": review.product.name,
                    "category": review.product.category,
                    "rating": review.rating,
                    "date": review.date,
                    "source": review.source,
                    "text": review.cleaned_text or review.raw_text,
                    "similarity_score": score
                })

        self.log_info(f"Retrieved {len(retrieved_reviews)} relevant reviews.")
        return retrieved_reviews

retriever_agent = RetrieverAgent()
