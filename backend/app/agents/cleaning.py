import re
import html
from typing import Dict, Any, Tuple
from backend.app.agents.base import BaseAgent
from backend.app.services.llm import llm_service

# Emoji matching regex
EMOJI_REGEX = re.compile(
    "["
    "\U00010000-\U0010ffff"  # Supplemental Planes
    "\u2600-\u27bf"          # Miscellaneous Symbols and Dingbats
    "\u200d"                 # Zero Width Joiner (used in emoji sequences)
    "]+", 
    flags=re.UNICODE
)

class CleaningAgent(BaseAgent):
    def __init__(self):
        super().__init__("CleaningAgent")

    def clean_text(self, text: str) -> str:
        """
        Remove HTML tags, strip emojis, decode HTML entities, and normalize whitespace.
        """
        if not text:
            return ""
        
        # 1. Decode HTML entities (e.g., &amp; -> &)
        text = html.unescape(text)
        
        # 2. Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        
        # 3. Remove Emojis
        text = EMOJI_REGEX.sub("", text)
        
        # 4. Remove spam/promotional links
        text = re.sub(r"http\S+|www\.\S+", "", text)
        
        # 5. Clean extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        return text

    def detect_spam(self, text: str) -> bool:
        """
        Identify if a review is spam (empty, promo links, random character repetition, etc.)
        """
        if not text or len(text.strip()) < 5:
            return True
            
        # Check if text is just repeated characters (e.g. "aaaaaaa")
        if re.search(r"(.)\1{6,}", text):
            return True
            
        # Promos
        if "buy now at" in text.lower() or "use coupon code" in text.lower() or "click here" in text.lower():
            return True
            
        return False

    async def clean_and_translate(self, raw_text: str) -> Tuple[str, str, bool]:
        """
        Cleans the review text, detects the language, and translates to English if necessary.
        Returns:
            Tuple[cleaned_text, language, is_spam]
        """
        is_spam = self.detect_spam(raw_text)
        if is_spam:
            return "", "unknown", True

        cleaned = self.clean_text(raw_text)
        
        # Detect language
        language = "en"
        try:
            from langdetect import detect
            language = detect(cleaned)
        except Exception:
            # Fallback check for common Hindi words (Hinglish/Hindi)
            hindi_keywords = ["achha", "kharab", "bekar", "badiya", "hai", "nahi", "tha", "hua", "kharab", "kam", "hi"]
            words = set(cleaned.lower().split())
            if words.intersection(hindi_keywords):
                language = "hi"

        # If not English, translate to English
        if language != "en" and len(cleaned) > 5:
            self.log_info(f"Non-English review detected (lang={language}). Translating...")
            translation_prompt = (
                f"Translate the following customer product review into clear, natural English. "
                f"Maintain the original sentiment, rating tone, and specific product details. "
                f"Provide ONLY the English translation and nothing else.\n\n"
                f"Review:\n\"{cleaned}\""
            )
            try:
                translated = await llm_service.generate_response(
                    prompt=translation_prompt,
                    system_instruction="You are a professional translator translating user feedback from regional Indian languages (Hindi, Hinglish, Tamil, etc.) to English."
                )
                cleaned = translated.strip().strip('"')
                self.log_info("Translation successful.")
            except Exception as e:
                self.log_error(f"Translation failed: {e}. Retaining cleaned text.")
        
        return cleaned, language, False

cleaning_agent = CleaningAgent()
