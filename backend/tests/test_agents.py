import pytest
from datetime import datetime
from backend.app.agents.cleaning import cleaning_agent
from backend.app.agents.ingestion import ingestion_agent
from backend.app.services.vector_store import vector_store_service

def test_cleaning_agent_html_removal():
    """Verify HTML tags are stripped and entities decoded."""
    raw = "<p>This is a <b>great</b> fan &amp; very stylish!</p>"
    cleaned = cleaning_agent.clean_text(raw)
    assert "great" in cleaned
    assert "great</b>" not in cleaned
    assert "&amp;" not in cleaned
    assert "great fan & very stylish!" in cleaned

def test_cleaning_agent_emoji_removal():
    """Verify emoji sequences are cleaned out."""
    raw = "Love this heater! 🔥⭐⭐⭐⭐ Highly recommend 👍"
    cleaned = cleaning_agent.clean_text(raw)
    assert "Love this heater!" in cleaned
    assert "🔥" not in cleaned
    assert "👍" not in cleaned

def test_cleaning_agent_spam_detection():
    """Verify spam checks identify promo links or key repetitions."""
    assert cleaning_agent.detect_spam("Buy now at http://cheap-fans.com/deal!!!") is True
    assert cleaning_agent.detect_spam("Good build quality.") is False
    assert cleaning_agent.detect_spam("abc") is True  # too short
    assert cleaning_agent.detect_spam("aaaaaaa") is True  # character repetition

def test_ingestion_normalize_row_keys():
    """Verify column mapping variations consolidate into uniform schema fields."""
    row = {
        "Product Title": "Havells Fan",
        "Stars": "4.5",
        "Review_Text": "Awesome fan!",
        "Date": "2026-07-15",
        "Verified": "yes",
        "Helpful": "5"
    }
    norm = ingestion_agent._normalize_row_keys(row)
    assert norm["product_name"] == "Havells Fan"
    assert norm["rating"] == 4
    assert norm["raw_text"] == "Awesome fan!"
    assert norm["verified_purchase"] is True
    assert norm["helpful_votes"] == 5
    assert isinstance(norm["date"], datetime)

@pytest.mark.asyncio
async def test_local_vector_store_crud():
    """Verify vector store upserting and search filtering functions."""
    review_ids = ["rev-1", "rev-2"]
    # Mock embeddings (size 768)
    v1 = [0.1] * 768
    v2 = [0.9] * 768
    
    payloads = [
        {"product_id": "prod-a", "category": "Fans", "rating": 5, "source": "test"},
        {"product_id": "prod-b", "category": "Heaters", "rating": 2, "source": "test"}
    ]
    
    await vector_store_service.upsert_reviews(review_ids, [v1, v2], payloads)
    
    # Check that search returns correct item with filters
    search_results = await vector_store_service.search_reviews(
        query_vector=[0.1] * 768,
        filter_metadata={"category": "Fans"},
        top_k=1
    )
    
    assert len(search_results) == 1
    assert search_results[0]["id"] == "rev-1"
    assert search_results[0]["payload"]["product_id"] == "prod-a"
