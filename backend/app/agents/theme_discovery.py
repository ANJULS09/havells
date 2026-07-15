import json
import numpy as np
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sklearn.cluster import MiniBatchKMeans
from backend.app.agents.base import BaseAgent
from backend.app.models import Review, Theme
from backend.app.services.llm import llm_service
from backend.app.services.vector_store import vector_store_service

class ThemeDiscoveryAgent(BaseAgent):
    def __init__(self):
        super().__init__("ThemeDiscoveryAgent")

    async def discover_themes(self, db: Session, target_num_themes: int = 8) -> List[Dict[str, Any]]:
        """
        Retrieves all reviews, clusters their embeddings, labels clusters using LLM, and persists them.
        """
        self.log_info("Starting theme discovery...")
        
        # 1. Fetch reviews
        reviews = db.query(Review).all()
        if not reviews:
            self.log_warning("No reviews found in database. Skipping theme discovery.")
            return []

        review_ids = [r.id for r in reviews]
        
        # 2. Get embeddings
        embedding_dict = await vector_store_service.get_embeddings_for_reviews(review_ids)
        
        # Check if any reviews are missing embeddings, generate them
        missing_ids = [rid for rid in review_ids if rid not in embedding_dict]
        if missing_ids:
            self.log_info(f"Generating embeddings for {len(missing_ids)} reviews in theme discovery...")
            missing_reviews = [r for r in reviews if r.id in missing_ids]
            missing_texts = [r.cleaned_text or r.raw_text for r in missing_reviews]
            
            try:
                new_embeddings = await llm_service.get_embeddings(missing_texts)
                # Save back to vector store
                payloads = [{
                    "product_id": r.product_id,
                    "rating": r.rating,
                    "date": r.date.isoformat(),
                    "source": r.source
                } for r in missing_reviews]
                await vector_store_service.upsert_reviews(missing_ids, new_embeddings, payloads)
                
                # Merge into dict
                for idx, rid in enumerate(missing_ids):
                    embedding_dict[rid] = new_embeddings[idx]
            except Exception as e:
                self.log_error(f"Error generating missing embeddings: {e}")

        # Re-align reviews and embeddings
        valid_reviews = []
        vectors = []
        for r in reviews:
            if r.id in embedding_dict:
                valid_reviews.append(r)
                vectors.append(embedding_dict[r.id])

        if len(valid_reviews) < 3:
            self.log_warning(f"Not enough reviews ({len(valid_reviews)}) to cluster. Needs at least 3 reviews.")
            return []

        X = np.array(vectors)
        
        # Determine number of clusters
        num_clusters = min(target_num_themes, len(valid_reviews) // 2)
        num_clusters = max(2, num_clusters)
        
        self.log_info(f"Clustering {len(valid_reviews)} reviews into {num_clusters} themes...")
        
        # 3. Perform K-Means Clustering
        kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, batch_size=100, n_init="auto")
        labels = kmeans.fit_predict(X)
        
        # Group reviews by cluster
        clusters = {i: [] for i in range(num_clusters)}
        for idx, label in enumerate(labels):
            clusters[label].append(valid_reviews[idx])

        discovered_themes = []
        
        # 4. Label each cluster using LLM
        for cluster_id, cluster_reviews in clusters.items():
            if not cluster_reviews:
                continue

            self.log_info(f"Labeling cluster {cluster_id} with {len(cluster_reviews)} reviews...")
            
            # Sample up to 5 representative reviews from the cluster
            # (Ideally we select reviews closest to the centroid, but a random or length-based selection is also very robust)
            # Let's sort by length and select middle-length reviews to avoid short/empty or excessively long reviews
            sorted_reviews = sorted(cluster_reviews, key=lambda r: len(r.cleaned_text or r.raw_text))
            sample_size = min(5, len(sorted_reviews))
            step = max(1, len(sorted_reviews) // sample_size)
            sampled = [sorted_reviews[i] for i in range(0, len(sorted_reviews), step)][:sample_size]

            # Construct review texts
            review_snippets_str = "\n".join([
                f"- Review #{idx+1} (Rating: {r.rating}/5): \"{r.cleaned_text or r.raw_text}\"" 
                for idx, r in enumerate(sampled)
            ])

            label_prompt = (
                f"You are analyzing a cluster of customer reviews for appliances (fans, purifiers, water heaters, etc.).\n"
                f"Here are a few representative reviews from this cluster:\n\n"
                f"{review_snippets_str}\n\n"
                f"Identify the common theme or topic uniting these reviews. Select a short, professional label of 1-3 words "
                f"(e.g., 'Motor Noise', 'Heating Issue', 'Build Quality', 'Delivery Delay', 'Ease of Installation').\n"
                f"Also write a brief description (1 sentence) explaining what issues or compliments are contained in this theme.\n\n"
                f"Return ONLY a JSON object in this exact format:\n"
                f"{{\n"
                f"  \"name\": \"Theme Name Here\",\n"
                f"  \"description\": \"Description here.\"\n"
                f"}}"
            )

            try:
                response = await llm_service.generate_response(
                    prompt=label_prompt,
                    system_instruction="You are a data intelligence assistant. Your output must be a clean JSON snippet with keys 'name' and 'description'."
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
                theme_name = data.get("name", f"Theme {cluster_id}").strip()
                theme_desc = data.get("description", "Auto-discovered theme.").strip()
                
                # 5. Persist theme to Relational Database
                theme = db.query(Theme).filter(Theme.name == theme_name).first()
                if not theme:
                    theme = Theme(name=theme_name, description=theme_desc, confidence=1.0)
                    db.add(theme)
                    db.commit()
                    db.refresh(theme)
                
                discovered_themes.append({
                    "id": theme.id,
                    "name": theme.name,
                    "description": theme.description,
                    "reviews_count": len(cluster_reviews)
                })
                
            except Exception as e:
                self.log_error(f"Failed to label cluster {cluster_id}: {e}")
                
        self.log_info(f"Theme discovery completed. Discovered {len(discovered_themes)} themes.")
        return discovered_themes

theme_discovery_agent = ThemeDiscoveryAgent()
