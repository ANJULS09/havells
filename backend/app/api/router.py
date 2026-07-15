from fastapi import APIRouter, Depends, UploadFile, File, Query, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
from backend.app.database import get_db
from backend.app.models import Product, Theme, Review, ReviewAspect
from backend.app import schemas
from backend.app.agents.ingestion import ingestion_agent
from backend.app.agents.theme_discovery import theme_discovery_agent
from backend.app.agents.sentiment import aspect_sentiment_agent
from backend.app.agents.trend import trend_agent
from backend.app.agents.answer_gen import answer_generation_agent

router = APIRouter()

@router.post("/reviews/upload", response_model=schemas.UploadStats)
async def upload_reviews_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload and ingest customer reviews from CSV or JSON.
    """
    contents = await file.read()
    try:
        stats = await ingestion_agent.ingest_reviews(contents, file.filename, db)
        return stats
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

@router.post("/analysis/discover-themes")
async def trigger_theme_discovery(
    target_num_themes: int = Query(8, ge=2, le=20),
    db: Session = Depends(get_db)
):
    """
    Run MiniBatchKMeans clustering to discover topics in reviews.
    """
    try:
        themes = await theme_discovery_agent.discover_themes(db, target_num_themes)
        return {"status": "success", "discovered_count": len(themes), "themes": themes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Theme discovery failed: {e}")

@router.post("/analysis/sentiment")
async def trigger_aspect_sentiment(db: Session = Depends(get_db)):
    """
    Extract aspect-based sentiment (ABSA) for reviews.
    """
    try:
        stats = await aspect_sentiment_agent.run_batch_sentiment_analysis(db)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Aspect sentiment analysis failed: {e}")

@router.get("/analysis/trends", response_model=schemas.TrendReportResponse)
async def get_trends_report(
    period_type: str = Query("monthly", regex="^(weekly|monthly|quarterly)$"),
    product_id: Optional[str] = None,
    category: Optional[str] = None,
    theme: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Calculate rating and aspect trends over weeks, months, or quarters.
    """
    try:
        report = await trend_agent.calculate_trends(
            db=db,
            period_type=period_type,
            product_id=product_id,
            category=category,
            theme=theme
        )
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trend calculation failed: {e}")

@router.post("/qa/query", response_model=schemas.QAResponse)
async def query_cve_system(
    query: schemas.QAQueryRequest,
    db: Session = Depends(get_db)
):
    """
    Ask natural language questions grounded in customer reviews.
    """
    try:
        response = await answer_generation_agent.answer_question(
            db=db,
            question=query.question,
            product_id=query.product_id,
            category=query.category,
            rating_filter=query.rating_filter,
            start_date=query.start_date,
            end_date=query.end_date
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"QA query processing failed: {e}")

@router.get("/products", response_model=List[schemas.ProductResponse])
def get_products(db: Session = Depends(get_db)):
    """
    List all ingested products.
    """
    return db.query(Product).all()

@router.get("/themes", response_model=List[schemas.ThemeResponse])
def get_themes(db: Session = Depends(get_db)):
    """
    List all discovered themes.
    """
    return db.query(Theme).filter(Theme.active == True).all()

@router.get("/reviews")
def get_reviews(
    product_id: Optional[str] = None,
    category: Optional[str] = None,
    rating: Optional[int] = None,
    limit: int = Query(20, le=100),
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """
    Retrieve and search matching review rows.
    """
    query = db.query(Review).join(Product)
    if product_id:
        query = query.filter(Review.product_id == product_id)
    if category:
        query = query.filter(Product.category == category)
    if rating:
        query = query.filter(Review.rating == rating)
        
    total = query.count()
    items = query.order_by(Review.date.desc()).offset(offset).limit(limit).all()
    
    # Structure response manually to include aspects list
    result = []
    for r in items:
        aspects = db.query(ReviewAspect).filter(ReviewAspect.review_id == r.id).all()
        result.append({
            "id": r.id,
            "product_name": r.product.name,
            "category": r.product.category,
            "rating": r.rating,
            "date": r.date,
            "source": r.source,
            "verified_purchase": r.verified_purchase,
            "helpful_votes": r.helpful_votes,
            "raw_text": r.raw_text,
            "cleaned_text": r.cleaned_text,
            "language": r.language,
            "aspects": [{"aspect": a.aspect, "sentiment": a.sentiment, "sentiment_score": a.sentiment_score, "snippet": a.snippet} for a in aspects]
        })
        
    return {"total": total, "reviews": result}
