from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# Product schemas
class ProductBase(BaseModel):
    name: str
    category: str

class ProductCreate(ProductBase):
    pass

class ProductResponse(ProductBase):
    id: str
    created_at: datetime

    class Config:
        from_attributes = True

# Review Aspect schemas
class ReviewAspectBase(BaseModel):
    aspect: str
    sentiment: str
    sentiment_score: float
    snippet: Optional[str] = None

class ReviewAspectCreate(ReviewAspectBase):
    review_id: str

class ReviewAspectResponse(ReviewAspectBase):
    id: str
    review_id: str

    class Config:
        from_attributes = True

# Review schemas
class ReviewBase(BaseModel):
    rating: int
    date: datetime
    source: Optional[str] = None
    verified_purchase: Optional[bool] = False
    helpful_votes: Optional[int] = 0
    raw_text: str

class ReviewCreate(ReviewBase):
    product_name: str
    category: str

class ReviewResponse(ReviewBase):
    id: str
    product_id: str
    cleaned_text: Optional[str] = None
    language: str
    created_at: datetime
    aspects: List[ReviewAspectResponse] = []

    class Config:
        from_attributes = True

# Theme schemas
class ThemeBase(BaseModel):
    name: str
    description: Optional[str] = None
    confidence: float = 1.0
    active: bool = True

class ThemeCreate(ThemeBase):
    pass

class ThemeResponse(ThemeBase):
    id: str
    created_at: datetime

    class Config:
        from_attributes = True

# Ingestion upload schemas
class UploadStats(BaseModel):
    total_received: int
    total_inserted: int
    total_duplicates_skipped: int
    products_created: List[str]

# QA & RAG schemas
class QAQueryRequest(BaseModel):
    question: str
    product_id: Optional[str] = None
    category: Optional[str] = None
    rating_filter: Optional[List[int]] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

class CitationItem(BaseModel):
    review_id: str
    product_name: str
    category: str
    rating: int
    date: datetime
    snippet: str
    source: Optional[str] = None

class QAResponse(BaseModel):
    answer: str
    citations: List[CitationItem]
    groundedness_score: float
    confidence_score: float
    reasoning_summary: str
    retrieved_count: int
    is_evidenced: bool  # false if "insufficient evidence"

# Trend analysis schemas
class TrendPoint(BaseModel):
    period: str  # e.g., "2026-W01", "2026-M03"
    total_reviews: int
    average_rating: float
    sentiment_distribution: Dict[str, int]  # positive, negative, neutral
    aspect_counts: Dict[str, int]  # aspect -> count
    aspect_sentiment: Dict[str, float]  # aspect -> average sentiment score (-1 to 1)

class TrendReportResponse(BaseModel):
    theme: Optional[str] = None
    product_id: Optional[str] = None
    category: Optional[str] = None
    period_type: str  # weekly, monthly, quarterly
    trends: List[TrendPoint]
