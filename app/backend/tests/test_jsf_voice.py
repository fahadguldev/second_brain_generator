import asyncio
import time
import unittest
from unittest.mock import patch, MagicMock
from src.jsf_keys import JSFKeyManager, KeyState
from src.voice_service import (
    is_voice_configured,
    get_voice_status,
    _compute_cache_key,
    synthesize_speech,
    VoiceConfigurationError,
    VoiceServiceError,
    AUDIO_CACHE_DIR,
)


class TestJSFKeyManager(unittest.TestCase):
    def test_key_pool_discovery_and_reserve(self):
        manager = JSFKeyManager(["key_1", "key_2", "key_3"])
        self.assertEqual(manager.total_keys(), 3)
        self.assertEqual(manager.active_keys_count(), 3)

        # Round-robin reservation
        k1 = manager.reserve_key()
        k2 = manager.reserve_key()
        k3 = manager.reserve_key()
        self.assertEqual(k1.api_key, "key_1")
        self.assertEqual(k2.api_key, "key_2")
        self.assertEqual(k3.api_key, "key_3")

    def test_key_rotation_on_429(self):
        manager = JSFKeyManager(["key_alpha", "key_beta"])
        k1 = manager.reserve_key()
        self.assertEqual(k1.api_key, "key_alpha")

        # Mark key_alpha as rate-limited with 30s cooldown
        manager.mark_key_failure("key_alpha", status_code=429, retry_after_seconds=30)
        self.assertEqual(manager.active_keys_count(), 1)

        # Next reserve must skip key_alpha and use key_beta
        k_next = manager.reserve_key()
        self.assertEqual(k_next.api_key, "key_beta")

    def test_key_rotation_on_auth_expiry_401(self):
        manager = JSFKeyManager(["expired_key", "healthy_backup_key"])
        # Expire first key
        manager.mark_key_failure("expired_key", status_code=401, reason="Unauthorized")

        # Must automatically select backup key
        k_active = manager.reserve_key()
        self.assertEqual(k_active.api_key, "healthy_backup_key")

    def test_all_keys_exhausted_raises(self):
        manager = JSFKeyManager(["single_key"])
        manager.mark_key_failure("single_key", status_code=429, retry_after_seconds=60)
        with self.assertRaises(RuntimeError):
            manager.reserve_key()


class TestJSFVoiceService(unittest.TestCase):
    def test_compute_cache_key_deterministic(self):
        key1 = _compute_cache_key("Hello world", "v1", 1.0)
        key2 = _compute_cache_key("  Hello world  ", "v1", 1.0)
        self.assertEqual(key1, key2)
        self.assertEqual(len(key1), 64)

    def test_synthesize_speech_uses_cache(self):
        text = "Unit test cached audio response for JSF Labs"
        voice = "jsf_mock_voice"
        speed = 1.0
        cache_key = _compute_cache_key(text, voice, speed)
        cache_file = AUDIO_CACHE_DIR / f"{cache_key}.wav"
        fake_wav = b"RIFF\x24\x00\x00\x00WAVEfmt mock-wav-data"
        cache_file.write_bytes(fake_wav)

        try:
            # Even if key manager is unconfigured, cache hit returns instantly!
            with patch("src.voice_service.is_voice_configured", return_value=True):
                result = asyncio.run(synthesize_speech(text, voice_id=voice, speed=speed))
                self.assertEqual(result, fake_wav)
        finally:
            if cache_file.exists():
                cache_file.unlink()

    def test_synthesize_speech_raises_when_unconfigured(self):
        with patch("src.voice_service.is_voice_configured", return_value=False):
            with self.assertRaises(VoiceConfigurationError):
                asyncio.run(synthesize_speech("Hello"))


if __name__ == "__main__":
    unittest.main()
