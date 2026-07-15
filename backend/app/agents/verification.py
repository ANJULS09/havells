import json
from typing import List, Dict, Any
from backend.app.agents.base import BaseAgent
from backend.app.services.llm import llm_service

class EvidenceVerificationAgent(BaseAgent):
    def __init__(self):
        super().__init__("EvidenceVerificationAgent")

    async def verify_answer(
        self,
        question: str,
        draft_answer: str,
        retrieved_reviews: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Verifies if the draft answer is fully grounded in the retrieved reviews.
        Returns evaluation scores and flags.
        """
        self.log_info("Starting evidence verification process...")
        
        if not retrieved_reviews:
            self.log_warning("No retrieved reviews to verify against.")
            return {
                "groundedness_score": 0.0,
                "hallucinations_detected": True,
                "citation_correctness": 0.0,
                "reasoning_summary": "No source documents were retrieved to support this answer."
            }

        # Construct verification context
        reviews_context = []
        for idx, r in enumerate(retrieved_reviews):
            reviews_context.append(
                f"Review ID: {r['review_id']}\n"
                f"Product: {r['product_name']} ({r['category']})\n"
                f"Rating: {r['rating']}/5\n"
                f"Content: \"{r['text']}\"\n"
                f"----------------------------------------"
            )
        reviews_context_str = "\n".join(reviews_context)

        prompt = (
            f"You are a Verification Agent. Your task is to audit a generated draft answer against retrieved source reviews. "
            f"Every fact in the answer must be directly supported by the text of the source reviews. If a claim is not supported, "
            f"or is an extrapolation/hallucination, mark it as unsupported.\n\n"
            f"User Question:\n\"{question}\"\n\n"
            f"Draft Answer:\n\"\"\"\n{draft_answer}\n\"\"\"\n\n"
            f"Source Reviews:\n\"\"\"\n{reviews_context_str}\n\"\"\"\n\n"
            f"Analyze the draft answer and provide:\n"
            f"1. A groundedness_score between 0.0 and 1.0 (where 1.0 means every sentence in the answer is completely supported by at least one review).\n"
            f"2. Whether hallucinations_detected is true (if there are claims in the answer that do not appear anywhere in the source reviews).\n"
            f"3. A citation_correctness score between 0.0 and 1.0 (checking if cited quotes actually match the text in the referenced review ID).\n"
            f"4. A reasoning_summary explaining your audit findings.\n\n"
            f"Return ONLY a JSON object in this exact format with no markdown wrappers:\n"
            f"{{\n"
            f"  \"groundedness_score\": 0.95,\n"
            f"  \"hallucinations_detected\": false,\n"
            f"  \"citation_correctness\": 1.0,\n"
            f"  \"reasoning_summary\": \"All statements are verified. Review ID X supports claim Y.\"\n"
            f"}}"
        )

        try:
            response = await llm_service.generate_response(
                prompt=prompt,
                system_instruction="You are an audit agent ensuring factual grounding and zero hallucination. Output only JSON."
            )

            # Clean code blocks from LLM markdown if present
            clean_response = response.strip()
            if clean_response.startswith("```"):
                lines = clean_response.splitlines()
                if lines[0].startswith("```json"):
                    clean_response = "\n".join(lines[1:-1])
                elif lines[0].startswith("```"):
                    clean_response = "\n".join(lines[1:-1])

            data = json.loads(clean_response)
            
            # Type coerce and fallback
            data["groundedness_score"] = float(data.get("groundedness_score", 0.0))
            data["hallucinations_detected"] = bool(data.get("hallucinations_detected", True))
            data["citation_correctness"] = float(data.get("citation_correctness", 0.0))
            data["reasoning_summary"] = str(data.get("reasoning_summary", "Audit completed."))

            self.log_info(f"Audit score: {data['groundedness_score']}. Hallucinations: {data['hallucinations_detected']}")
            return data

        except Exception as e:
            self.log_error(f"Error executing verification LLM check: {e}")
            return {
                "groundedness_score": 0.5,
                "hallucinations_detected": True,
                "citation_correctness": 0.5,
                "reasoning_summary": f"Verification system error: {e}. Defaulting to cautious evaluation."
            }

verification_agent = EvidenceVerificationAgent()
