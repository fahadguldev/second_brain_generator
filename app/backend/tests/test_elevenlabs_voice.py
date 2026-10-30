import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
from fastapi.testclient import TestClient

from src.main import app
from src.elevenlabs_service import (
    ElevenLabsKeyManager,
    ElevenLabsServiceError,
    ElevenLabsConfigurationError,
    synthesize_speech_elevenlabs,
    get_convai_signed_url,
)
from src.voice_service import synthesize_speech, get_voice_status


class TestElevenLabsVoiceService(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_key_manager_rotation_and_recovery(self):
        manager = ElevenLabsKeyManager(["key_primary", "key_backup"])
        self.assertEqual(manager.total_keys(), 2)

        # First reservation uses key_primary
        k1 = manager.reserve_key()
        self.assertEqual(k1.api_key, "key_primary")

        # Mark key_primary rate-limited (429)
        manager.mark_failure("key_primary", status_code=429, reason="Quota exceeded", retry_after_seconds=60)
        self.assertFalse(k1.is_available)

        # Next reservation picks key_backup
        k2 = manager.reserve_key()
        self.assertEqual(k2.api_key, "key_backup")

        # Mark backup success
        manager.mark_success("key_backup")
        self.assertEqual(k2.successes, 1)

    def test_key_manager_all_exhausted_raises(self):
        manager = ElevenLabsKeyManager(["single_key"])
        manager.mark_failure("single_key", status_code=429, reason="Rate limit", retry_after_seconds=30)
        with self.assertRaises(ElevenLabsServiceError) as ctx:
            manager.reserve_key()
        self.assertEqual(ctx.exception.status_code, 429)

    @patch("src.elevenlabs_service.elevenlabs_key_manager")
    def test_synthesize_speech_elevenlabs_success(self, mock_km):
        key_mock = MagicMock()
        key_mock.api_key = "test_eleven_key_123456"
        mock_km.reserve_key.return_value = key_mock
        mock_km.has_keys = True
        mock_km.total_keys.return_value = 1

        mock_audio = b"\xff\xfb\x90\x00MockMP3Data"

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.content = mock_audio

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            result_bytes, media_type = asyncio.run(
                synthesize_speech_elevenlabs("Assalam o Alaikum, main Fahad bol raha hoon")
            )
            self.assertEqual(result_bytes, mock_audio)
            self.assertEqual(media_type, "audio/mpeg")
            mock_km.mark_success.assert_called_with("test_eleven_key_123456")

    @patch("src.elevenlabs_service.elevenlabs_key_manager")
    def test_get_convai_signed_url_success(self, mock_km):
        key_mock = MagicMock()
        key_mock.api_key = "test_convai_key"
        mock_km.reserve_key.return_value = key_mock
        mock_km.has_keys = True

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "signed_url": "wss://api.elevenlabs.io/v1/convai/conversation?agent_id=test_agent&signature=abc"
        }

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            result = asyncio.run(get_convai_signed_url(agent_id="test_agent"))
            self.assertIn("wss://api.elevenlabs.io", result["signed_url"])
            self.assertEqual(result["agent_id"], "test_agent")

    def test_voice_status_endpoint_returns_elevenlabs_info(self):
        with patch("src.voice_service.get_active_provider", return_value="elevenlabs"), \
             patch("src.voice_service.is_elevenlabs_configured", return_value=True):
            res = self.client.get("/api/voice/status")
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["provider"], "elevenlabs")
            self.assertTrue(data["configured"])

    def test_tts_endpoint_returns_mp3_for_elevenlabs(self):
        mock_audio = b"\xff\xfb\x90\x00MP3AudioChunk"
        with patch("src.routes.synthesize_speech", new_callable=AsyncMock) as mock_synth:
            mock_synth.return_value = mock_audio
            res = self.client.post("/api/tts", json={"text": "Test ElevenLabs voice", "provider": "elevenlabs"})
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.headers["content-type"], "audio/mpeg")
            self.assertEqual(res.content, mock_audio)

    def test_convai_signed_url_endpoint(self):
        with patch("src.routes.get_convai_signed_url", new_callable=AsyncMock) as mock_get_url:
            mock_get_url.return_value = {
                "signed_url": "wss://api.elevenlabs.io/v1/convai/conversation?agent_id=test_agent",
                "agent_id": "test_agent",
            }
            res = self.client.get("/api/voice/convai/signed-url?agent_id=test_agent")
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.json()["agent_id"], "test_agent")
            self.assertIn("wss://", res.json()["signed_url"])


if __name__ == "__main__":
    unittest.main()
