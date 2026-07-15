import csv
import json
import io
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Tuple
from sqlalchemy.orm import Session
from backend.app.agents.base import BaseAgent
from backend.app.agents.cleaning import cleaning_agent
from backend.app.models import Product, Review
from backend.app.services.llm import llm_service
from backend.app.services.vector_store import vector_store_service

class IngestionAgent(BaseAgent):
    def __init__(self):
        super().__init__("IngestionAgent")

    def _normalize_row_keys(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize column variations from different datasets into standard keys.
        """
        normalized = {}
        row_lower = {k.lower().replace(" ", "_").replace("-", "_"): v for k, v in row.items()}
        
        # Product Name
        normalized["product_name"] = (
            row_lower.get("product_name") or 
            row_lower.get("product") or 
            row_lower.get("title") or 
            row_lower.get("product_title") or 
            "Unknown Product"
        )
        
        # Category
        normalized["category"] = (
            row_lower.get("category") or 
            row_lower.get("product_category") or 
            row_lower.get("class") or 
            "General"
        )

        # Review text
        normalized["raw_text"] = (
            row_lower.get("raw_text") or 
            row_lower.get("review") or 
            row_lower.get("text") or 
            row_lower.get("review_text") or 
            row_lower.get("body") or 
            row_lower.get("content") or 
            ""
        )
        
        # Rating
        rating_val = row_lower.get("rating") or row_lower.get("stars") or row_lower.get("score") or 5
        try:
            normalized["rating"] = int(float(rating_val))
        except ValueError:
            normalized["rating"] = 5

        # Date
        date_val = row_lower.get("date") or row_lower.get("timestamp") or row_lower.get("review_date")
        if date_val:
            for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%d-%m-%Y", "%m/%d/%Y", "%Y/%m/%d"):
                try:
                    normalized["date"] = datetime.strptime(str(date_val).strip(), fmt)
                    break
                except ValueError:
                    continue
            if "date" not in normalized:
                normalized["date"] = datetime.utcnow()
        else:
            normalized["date"] = datetime.utcnow()

        # Source
        normalized["source"] = row_lower.get("source") or row_lower.get("platform") or "upload"

        # Verified Purchase
        verified_val = row_lower.get("verified") or row_lower.get("verified_purchase") or False
        if isinstance(verified_val, str):
            normalized["verified_purchase"] = verified_val.lower() in ("true", "1", "yes", "verified")
        else:
            normalized["verified_purchase"] = bool(verified_val)

        # Helpful Votes
        helpful_val = row_lower.get("helpful") or row_lower.get("helpful_votes") or row_lower.get("votes") or 0
        try:
            normalized["helpful_votes"] = int(float(helpful_val))
        except ValueError:
            normalized["helpful_votes"] = 0

        return normalized

    async def ingest_reviews(self, file_content: bytes, filename: str, db: Session) -> Dict[str, Any]:
        """
        Parses bytes of a file (CSV or JSON), processes them, and saves to SQL and Vector DBs.
        """
        self.log_info(f"Ingesting file: {filename}")
        rows = []
        
        # Parse based on extension
        if filename.endswith(".json"):
            try:
                # Can be single JSON array or JSON Lines
                content_str = file_content.decode("utf-8")
                try:
                    data = json.loads(content_str)
                    if isinstance(data, list):
                        rows = data
                    else:
                        rows = [data]
                except json.JSONDecodeError:
                    # Try reading JSON Lines
                    rows = [json.loads(line) for line in content_str.splitlines() if line.strip()]
            except Exception as e:
                self.log_error(f"Error parsing JSON: {e}")
                raise ValueError("Invalid JSON format.")
        else:
            # Assume CSV
            try:
                content_str = file_content.decode("utf-8-sig")
                csv_reader = csv.DictReader(io.StringIO(content_str))
                rows = list(csv_reader)
            except Exception as e:
                self.log_error(f"Error parsing CSV: {e}")
                raise ValueError("Invalid CSV format.")

        total_received = len(rows)
        self.log_info(f"Parsed {total_received} rows. Starting cleaning & insertion...")

        total_inserted = 0
        total_duplicates_skipped = 0
        products_cache = {}  # name -> Product DB object
        created_product_names = set()

        # Gather reviews for batch embedding vector insertion
        batch_reviews = []
        batch_embeddings_texts = []
        batch_payloads = []

        for idx, raw_row in enumerate(rows):
            normalized = self._normalize_row_keys(raw_row)
            if not normalized["raw_text"]:
                continue

            # Ensure product exists
            prod_name = normalized["product_name"]
            prod_cat = normalized["category"]
            
            if prod_name not in products_cache:
                # Query DB
                product = db.query(Product).filter(Product.name == prod_name).first()
                if not product:
                    product = Product(name=prod_name, category=prod_cat)
                    db.add(product)
                    db.commit()
                    db.refresh(product)
                    created_product_names.add(prod_name)
                    self.log_info(f"Created new product: {prod_name} in category: {prod_cat}")
                products_cache[prod_name] = product
            
            product = products_cache[prod_name]

            # Check if this exact review text already exists for this product (De-duplication)
            existing_review = db.query(Review).filter(
                Review.product_id == product.id,
                Review.raw_text == normalized["raw_text"]
            ).first()
            if existing_review:
                total_duplicates_skipped += 1
                continue

            # Clean and translate
            cleaned_text, lang, is_spam = await cleaning_agent.clean_and_translate(normalized["raw_text"])
            if is_spam:
                continue

            # Insert Review to Relational DB
            review = Review(
                product_id=product.id,
                rating=normalized["rating"],
                date=normalized["date"],
                source=normalized["source"],
                verified_purchase=normalized["verified_purchase"],
                helpful_votes=normalized["helpful_votes"],
                raw_text=normalized["raw_text"],
                cleaned_text=cleaned_text,
                language=lang
            )
            db.add(review)
            db.commit()
            db.refresh(review)
            total_inserted += 1

            # Prepare for Vector DB ingestion
            batch_reviews.append(review)
            batch_embeddings_texts.append(cleaned_text if cleaned_text else normalized["raw_text"])
            batch_payloads.append({
                "product_id": product.id,
                "product_name": product.name,
                "category": product.category,
                "rating": normalized["rating"],
                "date": normalized["date"].isoformat(),
                "source": normalized["source"]
            })

            # Batch process vector DB writes every 20 records
            if len(batch_reviews) >= 20:
                await self._process_vector_batch(batch_reviews, batch_embeddings_texts, batch_payloads)
                batch_reviews = []
                batch_embeddings_texts = []
                batch_payloads = []

        # Process any remaining vectors in final batch
        if batch_reviews:
            await self._process_vector_batch(batch_reviews, batch_embeddings_texts, batch_payloads)

        self.log_info(f"Ingestion completed. Total parsed: {total_received}, Inserted: {total_inserted}, Duplicates skipped: {total_duplicates_skipped}")
        
        return {
            "total_received": total_received,
            "total_inserted": total_inserted,
            "total_duplicates_skipped": total_duplicates_skipped,
            "products_created": list(created_product_names)
        }

    async def _process_vector_batch(
        self, 
        reviews: List[Review], 
        texts: List[str], 
        payloads: List[Dict[str, Any]]
    ):
        """Helper to batch create embeddings and write to vector DB"""
        try:
            embeddings = await llm_service.get_embeddings(texts)
            review_ids = [r.id for r in reviews]
            await vector_store_service.upsert_reviews(review_ids, embeddings, payloads)
            self.log_info(f"Ingested {len(reviews)} vectors to vector DB.")
        except Exception as e:
            self.log_error(f"Error processing vector ingestion batch: {e}")

ingestion_agent = IngestionAgent()
