import json
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from backend.app.agents.base import BaseAgent
from backend.app.models import Review, ReviewAspect, Theme
from backend.app.services.llm import llm_service

class AspectSentimentAgent(BaseAgent):
    def __init__(self):
        super().__init__("AspectSentimentAgent")

    async def analyze_review_aspects(self, review: Review, themes: List[Theme], db: Session) -> List[ReviewAspect]:
        """
        Analyzes a single review for aspect-based sentiment against the list of themes.
        Saves results to SQL DB.
        """
        if not themes:
            self.log_warning("No themes provided for aspect sentiment analysis. Skipping.")
            return []

        theme_names = [t.name for t in themes]
        review_text = review.cleaned_text or review.raw_text

        prompt = (
            f"Perform Aspect-Based Sentiment Analysis (ABSA) on the following customer review:\n"
            f"Review: \"{review_text}\"\n\n"
            f"Below is the list of active themes you should map to:\n"
            f"{json.dumps(theme_names, indent=2)}\n\n"
            f"For this review, extract ONLY the themes from the list above that are mentioned (either explicitly or implicitly).\n"
            f"For each theme mentioned, determine:\n"
            f"1. aspect: The exact theme name from the list above.\n"
            f"2. sentiment: One of 'Positive', 'Negative', 'Neutral', or 'Mixed'.\n"
            f"3. sentiment_score: A float between -1.0 (extremely negative) and 1.0 (extremely positive).\n"
            f"4. snippet: The exact words or clause from the review describing this aspect.\n\n"
            f"Return ONLY a JSON object in this exact format, with no extra characters:\n"
            f"{{\n"
            f"  \"aspects\": [\n"
            f"    {{\n"
            f"      \"aspect\": \"Theme Name\",\n"
            f"      \"sentiment\": \"Negative\",\n"
            f"      \"sentiment_score\": -0.8,\n"
            f"      \"snippet\": \"makes too much noise\"\n"
            f"    }}\n"
            f"  ]\n"
            f"}}"
        )

        try:
            response = await llm_service.generate_response(
                prompt=prompt,
                system_instruction="You are an Aspect-Based Sentiment Analysis engine. Output only JSON conforming to the requested schema."
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
            aspect_items = data.get("aspects", [])

            # Clear any existing aspects for this review to prevent duplicates
            db.query(ReviewAspect).filter(ReviewAspect.review_id == review.id).delete()
            db.commit()

            inserted_aspects = []
            for item in aspect_items:
                aspect_name = item.get("aspect")
                if aspect_name not in theme_names:
                    # Skip if the LLM hallucinated a theme name not in our official list
                    continue

                aspect_record = ReviewAspect(
                    review_id=review.id,
                    aspect=aspect_name,
                    sentiment=item.get("sentiment", "Neutral"),
                    sentiment_score=float(item.get("sentiment_score", 0.0)),
                    snippet=item.get("snippet", "")
                )
                db.add(aspect_record)
                inserted_aspects.append(aspect_record)

            db.commit()
            return inserted_aspects

        except Exception as e:
            self.log_error(f"Error performing aspect sentiment for review {review.id}: {e}")
            return []

    async def run_batch_sentiment_analysis(self, db: Session) -> Dict[str, Any]:
        """
        Runs aspect-based sentiment analysis in batch on all reviews that don't have aspect records.
        """
        self.log_info("Starting batch aspect sentiment analysis...")
        
        themes = db.query(Theme).filter(Theme.active == True).all()
        if not themes:
            self.log_warning("No themes discovered yet. Please run Theme Discovery first.")
            return {"status": "error", "message": "No themes discovered yet. Run Theme Discovery first."}

        # Query reviews that don't have any aspect records in the database
        reviews_to_analyze = db.query(Review).outerjoin(ReviewAspect).filter(ReviewAspect.id == None).all()
        
        total = len(reviews_to_analyze)
        self.log_info(f"Found {total} reviews needing aspect sentiment analysis.")
        
        processed_count = 0
        aspects_count = 0

        for review in reviews_to_analyze:
            aspects = await self.analyze_review_aspects(review, themes, db)
            processed_count += 1
            aspects_count += len(aspects)
            
            if processed_count % 10 == 0:
                self.log_info(f"Analyzed {processed_count}/{total} reviews...")

        self.log_info(f"Batch aspect analysis finished. Processed {processed_count} reviews, saved {aspects_count} aspect records.")
        return {
            "status": "success",
            "reviews_processed": processed_count,
            "aspects_created": aspects_count
        }

aspect_sentiment_agent = AspectSentimentAgent()
