import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from backend.app.database import Base

def generate_uuid():
    return str(uuid.uuid4())

class Product(Base):
    __tablename__ = "products"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False, index=True)
    category = Column(String(255), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    reviews = relationship("Review", back_populates="product", cascade="all, delete-orphan")

class Review(Base):
    __tablename__ = "reviews"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    product_id = Column(String(36), ForeignKey("products.id"), nullable=False)
    rating = Column(Integer, nullable=False)
    date = Column(DateTime, nullable=False, index=True)
    source = Column(String(100), nullable=True)
    verified_purchase = Column(Boolean, default=False)
    helpful_votes = Column(Integer, default=0)
    raw_text = Column(Text, nullable=False)
    cleaned_text = Column(Text, nullable=True)
    language = Column(String(10), default="en")
    created_at = Column(DateTime, default=datetime.utcnow)

    product = relationship("Product", back_populates="reviews")
    aspects = relationship("ReviewAspect", back_populates="review", cascade="all, delete-orphan")

class ReviewAspect(Base):
    __tablename__ = "review_aspects"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    review_id = Column(String(36), ForeignKey("reviews.id"), nullable=False)
    aspect = Column(String(100), nullable=False, index=True)  # e.g., "Motor Noise", "Appearance"
    sentiment = Column(String(20), nullable=False)  # Positive, Negative, Neutral, Mixed
    sentiment_score = Column(Float, nullable=False)  # -1.0 to 1.0 (or 0.0 to 1.0 confidence)
    snippet = Column(Text, nullable=True)  # Quote from the review supporting this aspect
    created_at = Column(DateTime, default=datetime.utcnow)

    review = relationship("Review", back_populates="aspects")

class Theme(Base):
    __tablename__ = "themes"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(100), nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)
    confidence = Column(Float, default=1.0)
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
