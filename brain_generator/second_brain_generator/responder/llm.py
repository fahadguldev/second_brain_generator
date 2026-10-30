"""Gemini API client handling key rotation, multi-model fallbacks, and query embedding."""

from __future__ import annotations

import logging
import random
import time
from typing import List, Optional, Tuple
import numpy as np

from ..utils.env import get_gemini_keys
from ..embedder.embedder import _deterministic_mock_vector
from .types import GenerationConfig
from .prompts import GANJISWAG_SYSTEM_INSTRUCTION

logger = logging.getLogger(__name__)


class GeminiResponderClient:
    """
    Manages robust interaction with Gemini models for embedding queries
    and generating long-form Roman Urdu responses with automatic key rotation.
    """

    def __init__(self, config: Optional[GenerationConfig] = None) -> None:
        self.config = config or GenerationConfig()
        self.keys = get_gemini_keys()
        # Start at random offset to distribute requests evenly across available keys
        self._current_key_idx = random.randint(0, max(0, len(self.keys) - 1)) if self.keys else 0

    def _next_key(self) -> Optional[str]:
        """Rotates to the next available Gemini API key."""
        if not self.keys:
            return None
        key = self.keys[self._current_key_idx % len(self.keys)]
        self._current_key_idx += 1
        return key

    def embed_query(self, query: str) -> Tuple[np.ndarray, float]:
        """
        Generates 3072-dimensional embedding for query text.
        Falls back to deterministic offline mock vector if offline or API unavailable.
        """
        start_t = time.perf_counter()

        if self.config.offline or not self.keys:
            vec = _deterministic_mock_vector(query, dim=self.config.embedding_dimension)
            elapsed = time.perf_counter() - start_t
            return vec, elapsed

        # Attempt remote embedding with key rotation
        max_attempts = min(len(self.keys), 5)
        for attempt in range(max_attempts):
            key = self._next_key()
            if not key:
                break
            try:
                from google import genai
                from google.genai import types

                client = genai.Client(
                    api_key=key,
                    http_options=types.HttpOptions(timeout=12000),
                )
                model_name = self.config.embedding_model.replace("models/", "")
                result = client.models.embed_content(
                    model=model_name,
                    contents=query,
                    config=types.EmbedContentConfig(
                        output_dimensionality=self.config.embedding_dimension
                    ),
                )
                if result.embeddings and len(result.embeddings) > 0:
                    vec = np.array(result.embeddings[0].values, dtype=np.float32)
                    norm = np.linalg.norm(vec)
                    if norm > 1e-12:
                        vec /= norm
                    elapsed = time.perf_counter() - start_t
                    return vec, elapsed
            except Exception as exc:
                logger.debug("Embedding attempt %d failed on key: %s", attempt + 1, exc)
                continue

        # Offline fallback
        logger.info("Falling back to deterministic query vector.")
        vec = _deterministic_mock_vector(query, dim=self.config.embedding_dimension)
        elapsed = time.perf_counter() - start_t
        return vec, elapsed

    def generate_response(
        self,
        prompt: str,
        system_instruction: str = GANJISWAG_SYSTEM_INSTRUCTION,
    ) -> Tuple[str, str, float]:
        """
        Generates comprehensive Roman Urdu text via Gemini with model and key fallback.
        Returns: (generated_text, model_name_used, elapsed_seconds).
        """
        start_t = time.perf_counter()

        if self.config.offline or not self.keys:
            text = self._generate_offline_fallback(prompt)
            elapsed = time.perf_counter() - start_t
            return text, "offline-fallback", elapsed

        models_to_try = [self.config.model]
        for m in self.config.fallback_models:
            if m not in models_to_try:
                models_to_try.append(m)

        for model_candidate in models_to_try:
            # Try up to 3 distinct keys per candidate model
            for _ in range(min(len(self.keys), 3)):
                key = self._next_key()
                if not key:
                    break
                try:
                    from google import genai
                    from google.genai import types

                    client = genai.Client(
                        api_key=key,
                        http_options=types.HttpOptions(timeout=20000),
                    )
                    mod_name = model_candidate.replace("models/", "")

                    config = types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=self.config.temperature,
                        max_output_tokens=self.config.max_output_tokens,
                    )
                    res = client.models.generate_content(
                        model=mod_name,
                        contents=prompt,
                        config=config,
                    )
                    if res and res.text:
                        elapsed = time.perf_counter() - start_t
                        return res.text.strip(), mod_name, elapsed
                except Exception as err:
                    err_str = str(err).lower()
                    if "503" in err_str or "demand" in err_str:
                        logger.debug("Model %s 503 unavailable, skipping to next model.", model_candidate)
                        break
                    if "404" in err_str:
                        logger.debug("Model %s 404 not found, skipping to next model.", model_candidate)
                        break
                    # For 429, timeout, or other errors, try next key
                    continue

        # If all API calls fail, return rich offline fallback
        logger.warning("All Gemini generation attempts exhausted. Using offline response synthesis.")
        text = self._generate_offline_fallback(prompt)
        elapsed = time.perf_counter() - start_t
        return text, "offline-synthesizer", elapsed

    def _generate_offline_fallback(self, prompt: str) -> str:
        """
        High-quality offline synthesizer ensuring the pipeline never crashes
        even without internet or valid API keys.
        """
        return (
            "# GanjiSwag AI Response (Offline Mode)\n\n"
            "### 1. Seedhi Baat & Ground Reality\n"
            "Dekhein bhai, baat yeh hai ke jo sawal aapne poocha hai, iska seedha sa jawab yeh hai ke "
            "humaari society mein log hamesha shortcuts aur asaan raste dhoondte hain, jabke haqeeqat "
            "hamesha thori karwi aur mehnat talab hoti hai. Junaid Akram hamesha kehte hain ke jab tak "
            "aap apne gireban mein nahi jhankte aur khud ki responsibilities nahi lete, tab tak system ko "
            "bura kehne ka koi faida nahi.\n\n"
            "### 2. Samaj Ka Masla & Tafseeli Tajzia\n"
            "Hamare mulk mein sab se bara almiya yeh hai ke yahan bher chal (herd mentality) bohot aam hai. "
            "Har banda doosron ko dekh kar faislay karta hai — chahe career choose karna ho, degree karni ho, "
            "ya rishta dhoondna ho. Jab tak aap independent thinking develop nahi karenge, aap is toxic "
            "cycle ka shikar rahenge. Middle class ka masla yeh hai ke hum par logon ki expectations ka bohot "
            "zyada bojh hota hai, aur hum 'log kya kahenge' ke chakkar mein apni zindagi zaya kar dete hain.\n\n"
            "### 3. Ask Ganjiswag Ki Roshni Mein\n"
            "Podcast mein Junaid Akram ne barha yeh point uthaya hai ke practical life mein sirf kitabi batein "
            "ya degree kaam nahi aati. Dunyadari, networking, civic sense, aur apnay andar bardasht paida karna "
            "bohot zaroori hai. Karachi ki misaal lein ya poore Pakistan ki — jo log mehnat karte hain aur apne "
            "kaam se kaam rakhte hain, wohi aage nikalte hain.\n\n"
            "### 4. Amali Mashwaray Aur Solution\n"
            "1. **Pehle Khud Ko Pehchanein:** Jo kaam aap kar rahe hain, kya usme aapka dil hai ya sirf paisa dekh kar ghus gaye hain?\n"
            "2. **Skills Par Invest Karein:** Rozana kam az kam 1-2 ghante naye tools aur skills seekhne mein lagayein.\n"
            "3. **Social Media Detox:** Fuzool scrolling aur negativity se bachein. Apne waqt ki qadar karein.\n"
            "4. **Financial Literacy:** Paisa bachana aur usko sahi jagah invest karna seekhein.\n\n"
            "### 5. GanjiSwag Ka Aakhri Paighaam\n"
            "Bhai, zindagi aapki apni hai. Agar aap fail huye toh tana dene walay bhi yehi log honge, aur agar "
            "kamyab huye toh mithai khane walay bhi yehi log honge. Is liye apnay dimaag se sochein, mehnat karein, "
            "aur baaqi sab Allah par chor dein."
        )
