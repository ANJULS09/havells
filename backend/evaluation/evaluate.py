import asyncio
import time
import uuid
from datetime import datetime
from sqlalchemy.orm import Session
from backend.app.database import SessionLocal, Base, engine
from backend.app.models import Product, Review, Theme, ReviewAspect
from backend.app.agents.cleaning import cleaning_agent
from backend.app.agents.theme_discovery import theme_discovery_agent
from backend.app.agents.sentiment import aspect_sentiment_agent
from backend.app.agents.answer_gen import answer_generation_agent
from backend.app.services.vector_store import vector_store_service
from backend.app.services.llm import llm_service

# Define test dataset for evaluation
EVAL_PRODUCT_NAME = "Havells Test Fan X"
EVAL_PRODUCT_CATEGORY = "Fans"

EVAL_REVIEWS = [
    {
        "text": "The motor is extremely silent and works very well. Sturdy blade design.",
        "expected_aspects": {
            "Motor Noise": "Positive",
            "Build Quality": "Positive"
        }
    },
    {
        "text": "Poor packaging, the box was damaged. The fan motor makes a loud rattling sound.",
        "expected_aspects": {
            "Packaging": "Negative",
            "Motor Noise": "Negative"
        }
    },
    {
        "text": "Installation was very difficult and took 2 hours. But it looks beautiful in our living room.",
        "expected_aspects": {
            "Ease of Installation": "Negative",
            "Appearance": "Positive"
        }
    }
]

async def seed_evaluation_data(db: Session) -> Product:
    # Ensure test product exists
    product = db.query(Product).filter(Product.name == EVAL_PRODUCT_NAME).first()
    if not product:
        product = Product(name=EVAL_PRODUCT_NAME, category=EVAL_PRODUCT_CATEGORY)
        db.add(product)
        db.commit()
        db.refresh(product)
    
    # Clean old test reviews
    db.query(Review).filter(Review.product_id == product.id).delete()
    db.commit()

    # Seed reviews and vectors
    reviews = []
    vector_texts = []
    payloads = []
    for item in EVAL_REVIEWS:
        review = Review(
            product_id=product.id,
            rating=4,
            date=datetime.utcnow(),
            source="test_runner",
            raw_text=item["text"],
            cleaned_text=item["text"],
            language="en"
        )
        db.add(review)
        db.commit()
        db.refresh(review)
        reviews.append(review)
        vector_texts.append(item["text"])
        payloads.append({
            "product_id": product.id,
            "product_name": product.name,
            "category": product.category,
            "rating": 4,
            "date": review.date.isoformat(),
            "source": "test_runner"
        })

    # Embed & insert vectors
    embeddings = await llm_service.get_embeddings(vector_texts)
    await vector_store_service.upsert_reviews([r.id for r in reviews], embeddings, payloads)
    
    # Seed matching test themes if they don't exist
    required_themes = ["Motor Noise", "Build Quality", "Packaging", "Ease of Installation", "Appearance"]
    for t_name in required_themes:
        theme = db.query(Theme).filter(Theme.name == t_name).first()
        if not theme:
            theme = Theme(name=t_name, description=f"Test theme for {t_name}")
            db.add(theme)
            db.commit()

    return product

async def evaluate_pipeline():
    print("========================================")
    print("STARTING PIPELINE EVALUATION SUITE")
    print("========================================")
    
    # Ensure tables are created
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    try:
        # 1. Seed data
        print("[INFO] Seeding test database reviews...")
        product = await seed_evaluation_data(db)
        
        # 2. Benchmark aspect sentiment extraction
        print("\n--- Benchmarking Aspect Sentiment extraction (ABSA) ---")
        start_time = time.time()
        
        test_reviews = db.query(Review).filter(Review.product_id == product.id).all()
        themes = db.query(Theme).all()
        
        correct_sentiment_hits = 0
        total_expected_aspects = 0
        total_extracted_aspects = 0
        
        for idx, review in enumerate(test_reviews):
            extracted = await aspect_sentiment_agent.analyze_review_aspects(review, themes, db)
            expected = EVAL_REVIEWS[idx]["expected_aspects"]
            
            total_expected_aspects += len(expected)
            total_extracted_aspects += len(extracted)
            
            # Map extraction
            extracted_map = {asp.aspect: asp.sentiment for asp in extracted}
            for exp_aspect, exp_sent in expected.items():
                if exp_aspect in extracted_map and extracted_map[exp_aspect] == exp_sent:
                    correct_sentiment_hits += 1

        absa_latency = (time.time() - start_time) / len(test_reviews)
        absa_precision = correct_sentiment_hits / total_extracted_aspects if total_extracted_aspects > 0 else 0.0
        absa_recall = correct_sentiment_hits / total_expected_aspects if total_expected_aspects > 0 else 0.0
        absa_f1 = 2 * (absa_precision * absa_recall) / (absa_precision + absa_recall) if (absa_precision + absa_recall) > 0 else 0.0

        print(f"ABSA Latency per review: {absa_latency:.3f}s")
        print(f"Aspect Sentiment Precision: {absa_precision * 100:.1f}%")
        print(f"Aspect Sentiment Recall: {absa_recall * 100:.1f}%")
        print(f"Aspect Sentiment F1-Score: {absa_f1 * 100:.1f}%")

        # 3. Benchmark RAG Retrieval and Grounding
        print("\n--- Benchmarking Grounded QA Retrieval (RAG) & Verification ---")
        
        questions = [
            ("What are the motor noise issues with the fan?", True),  # Grounded query (should find noisy motor review)
            ("Did anyone complain about delivery box packaging damage?", True),  # Grounded query
            ("What are the air filter dander purification features?", False)  # Unsupported query (no reviews have air purifiers here)
        ]
        
        for question, expects_grounding in questions:
            start_q_time = time.time()
            qa_res = await answer_generation_agent.answer_question(db, question, product_id=product.id)
            latency = time.time() - start_q_time
            
            print(f"\nQuestion: \"{question}\"")
            print(f"Latency: {latency:.3f}s")
            print(f"Is Evidenced/Grounded: {qa_res.is_evidenced}")
            print(f"Groundedness score: {qa_res.groundedness_score * 100:.1f}%")
            print(f"Confidence score: {qa_res.confidence_score * 100:.1f}%")
            print(f"Citations returned: {len(qa_res.citations)}")
            print(f"Answer snippet: {qa_res.answer[:150]}...")
            
            # Check correctness
            if expects_grounding and not qa_res.is_evidenced:
                print(">> [WARNING] Expected grounding but got insufficient evidence fallback.")
            elif not expects_grounding and qa_res.is_evidenced:
                print(">> [ALERT] Potential hallucination: Generated answer for unsupported query!")
            else:
                print(">> [PASS] Grounding behaves as expected.")

        print("\n========================================")
        print("EVALUATION COMPLETE")
        print("========================================")
        
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(evaluate_pipeline())
