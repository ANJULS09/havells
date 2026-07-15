import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from backend.app.agents.base import BaseAgent
from backend.app.agents.retriever import retriever_agent
from backend.app.agents.verification import verification_agent
from backend.app.services.llm import llm_service
from backend.app.schemas import QAResponse, CitationItem

class AnswerGenerationAgent(BaseAgent):
    def __init__(self):
        super().__init__("AnswerGenerationAgent")

    def _extract_cited_ids(self, answer_text: str) -> List[str]:
        """
        Extracts cited Review IDs from the text (looking for formats like [UUID] or similar).
        """
        # Matches UUIDs in brackets, e.g., [4b1b369c-0975-4be4-a82f-2d7c079ad5be]
        pattern = r"\[([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\]"
        return re.findall(pattern, answer_text)

    async def answer_question(
        self,
        db: Session,
        question: str,
        product_id: Optional[str] = None,
        category: Optional[str] = None,
        rating_filter: Optional[List[int]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> QAResponse:
        """
        Executes complete RAG pipeline to answer user questions using only reviews.
        """
        self.log_info(f"Answering question: '{question}'")
        
        # 1. Retrieve relevant reviews
        reviews = await retriever_agent.retrieve_relevant_reviews(
            db=db,
            question=question,
            product_id=product_id,
            category=category,
            rating_filter=rating_filter,
            start_date=start_date,
            end_date=end_date,
            top_k=8
        )

        # Fallback if no reviews retrieved
        if not reviews:
            return QAResponse(
                answer="There is insufficient evidence in the available reviews.",
                citations=[],
                groundedness_score=0.0,
                confidence_score=0.0,
                reasoning_summary="No reviews matched the specified search criteria or filters.",
                retrieved_count=0,
                is_evidenced=False
            )

        # 2. Construct LLM Context
        reviews_context = []
        for r in reviews:
            reviews_context.append(
                f"Review ID: [{r['review_id']}]\n"
                f"Product Name: {r['product_name']}\n"
                f"Rating: {r['rating']}/5 stars\n"
                f"Date: {r['date'].strftime('%Y-%m-%d')}\n"
                f"Review: \"{r['text']}\"\n"
                f"----------------------------------------"
            )
        reviews_context_str = "\n".join(reviews_context)

        draft_prompt = (
            f"You are a Customer Voice Intelligence Agent for Havells. Answer the user's question using ONLY the provided reviews. "
            f"Do NOT make assumptions, extrapolate, or use outside knowledge. "
            f"Every fact or claim in your response MUST be directly supported by a review, and you MUST cite the corresponding Review ID in brackets, e.g. [ReviewID_Here].\n\n"
            f"If the reviews do not contain enough information to answer the question, respond EXACTLY with:\n"
            f"\"There is insufficient evidence in the available reviews.\"\n\n"
            f"User Question:\n\"{question}\"\n\n"
            f"Retrieved Reviews:\n\"\"\"\n{reviews_context_str}\n\"\"\"\n\n"
            f"Draft Answer:"
        )

        # 3. Generate Draft Response
        draft_answer = await llm_service.generate_response(
            prompt=draft_prompt,
            system_instruction="You are an expert customer feedback analyzer. Your response must be objective, grounded, and clearly cite source reviews."
        )
        draft_answer = draft_answer.strip()

        # Handle explicit lack of evidence in draft
        if "insufficient evidence" in draft_answer.lower():
            return QAResponse(
                answer="There is insufficient evidence in the available reviews.",
                citations=[],
                groundedness_score=1.0,
                confidence_score=1.0,
                reasoning_summary="The LLM determined that the retrieved reviews do not contain information answering the question.",
                retrieved_count=len(reviews),
                is_evidenced=False
            )

        # 4. Verify Draft
        verification_report = await verification_agent.verify_answer(
            question=question,
            draft_answer=draft_answer,
            retrieved_reviews=reviews
        )

        # 5. Check Groundedness Threshold (e.g. 0.70)
        is_grounded = (
            verification_report["groundedness_score"] >= 0.70 and 
            not verification_report["hallucinations_detected"]
        )

        if not is_grounded:
            self.log_warning("Draft answer failed verification checks. Outputting insufficient evidence fallback.")
            return QAResponse(
                answer="There is insufficient evidence in the available reviews.",
                citations=[],
                groundedness_score=verification_report["groundedness_score"],
                confidence_score=0.1,
                reasoning_summary=f"Audit failed. {verification_report['reasoning_summary']}",
                retrieved_count=len(reviews),
                is_evidenced=False
            )

        # 6. Extract and build CitationItem objects
        cited_ids = self._extract_cited_ids(draft_answer)
        citations = []
        unique_cited = set(cited_ids)

        for review_id in unique_cited:
            # Find the corresponding review in retrieved reviews
            match = next((r for r in reviews if r["review_id"] == review_id), None)
            if match:
                # We can extract a snippet from the review text that explains the citation
                citations.append(CitationItem(
                    review_id=review_id,
                    product_name=match["product_name"],
                    category=match["category"],
                    rating=match["rating"],
                    date=match["date"],
                    snippet=match["text"][:300] + ("..." if len(match["text"]) > 300 else ""),
                    source=match.get("source")
                ))

        # Calculate confidence score: base on average review semantic similarity score * groundedness
        avg_similarity = sum(r["similarity_score"] for r in reviews) / len(reviews) if reviews else 0.0
        confidence = avg_similarity * verification_report["groundedness_score"]

        return QAResponse(
            answer=draft_answer,
            citations=citations,
            groundedness_score=verification_report["groundedness_score"],
            confidence_score=round(confidence, 2),
            reasoning_summary=verification_report["reasoning_summary"],
            retrieved_count=len(reviews),
            is_evidenced=True
        )

answer_generation_agent = AnswerGenerationAgent()
