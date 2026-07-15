import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from backend.app.agents.base import BaseAgent
from backend.app.models import Review, ReviewAspect, Product

class TrendAgent(BaseAgent):
    def __init__(self):
        super().__init__("TrendAgent")

    def _get_period_key(self, date: datetime, period_type: str) -> str:
        """
        Formats datetime into a period string key based on period type.
        """
        if period_type == "weekly":
            # Format as YYYY-Www (e.g. 2026-W28)
            # %W is week number of the year, starting on Monday
            return date.strftime("%Y-W%W")
        elif period_type == "monthly":
            # Format as YYYY-MM (e.g. 2026-07)
            return date.strftime("%Y-%m")
        elif period_type == "quarterly":
            # Format as YYYY-Qq (e.g. 2026-Q3)
            quarter = (date.month - 1) // 3 + 1
            return f"{date.year}-Q{quarter}"
        else:
            return date.strftime("%Y-%m-%d")

    async def calculate_trends(
        self,
        db: Session,
        period_type: str = "monthly",
        product_id: Optional[str] = None,
        category: Optional[str] = None,
        theme: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Aggregates reviews and aspect sentiments over time periods.
        """
        self.log_info(f"Calculating trends. Type: {period_type}, Product: {product_id}, Category: {category}, Theme: {theme}")
        
        # Build query
        query = db.query(Review).join(Product)
        
        if product_id:
            query = query.filter(Review.product_id == product_id)
        if category:
            query = query.filter(Product.category == category)
            
        reviews = query.all()
        if not reviews:
            self.log_warning("No reviews found matching trend filters.")
            return {
                "theme": theme,
                "product_id": product_id,
                "category": category,
                "period_type": period_type,
                "trends": []
            }

        # Convert to Pandas DataFrame for powerful time-series group-bys
        data = []
        for r in reviews:
            # Load aspects
            aspects_query = db.query(ReviewAspect).filter(ReviewAspect.review_id == r.id)
            if theme:
                aspects_query = aspects_query.filter(ReviewAspect.aspect == theme)
            
            aspects = aspects_query.all()
            
            # If a specific theme was requested and this review does not have it, skip
            if theme and not aspects:
                continue

            aspect_data = {asp.aspect: (asp.sentiment, asp.sentiment_score) for asp in aspects}

            data.append({
                "id": r.id,
                "date": r.date,
                "rating": r.rating,
                "aspects": aspect_data
            })

        if not data:
            return {
                "theme": theme,
                "product_id": product_id,
                "category": category,
                "period_type": period_type,
                "trends": []
            }

        df = pd.DataFrame(data)
        # Add period column
        df["period"] = df["date"].apply(lambda d: self._get_period_key(d, period_type))

        # Group by period
        grouped = df.groupby("period")
        trend_points = []

        for period, group in grouped:
            total_reviews = len(group)
            avg_rating = float(group["rating"].mean())
            
            # Count aspect sentiments and scores
            sentiment_distribution = {"Positive": 0, "Negative": 0, "Neutral": 0, "Mixed": 0}
            aspect_counts = {}
            aspect_sums = {}
            
            for index, row in group.iterrows():
                aspects_dict = row["aspects"]
                for aspect, (sent, score) in aspects_dict.items():
                    # Aggregating total aspects count
                    aspect_counts[aspect] = aspect_counts.get(aspect, 0) + 1
                    
                    # Accumulating scores for average calculation
                    aspect_sums[aspect] = aspect_sums.get(aspect, 0.0) + score
                    
                    # Distribute review overall/aspect sentiment counts
                    sentiment_distribution[sent] = sentiment_distribution.get(sent, 0) + 1

            # Compute aspect average sentiment scores (-1.0 to 1.0)
            aspect_sentiment = {}
            for aspect, count in aspect_counts.items():
                if count > 0:
                    aspect_sentiment[aspect] = float(aspect_sums[aspect] / count)

            trend_points.append({
                "period": str(period),
                "total_reviews": total_reviews,
                "average_rating": round(avg_rating, 2),
                "sentiment_distribution": sentiment_distribution,
                "aspect_counts": aspect_counts,
                "aspect_sentiment": {k: round(v, 2) for k, v in aspect_sentiment.items()}
            })

        # Sort chronologically by period
        trend_points.sort(key=lambda x: x["period"])

        return {
            "theme": theme,
            "product_id": product_id,
            "category": category,
            "period_type": period_type,
            "trends": trend_points
        }

trend_agent = TrendAgent()
