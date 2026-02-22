import pytest
from unittest.mock import MagicMock, patch
from typing import List

from app.services.inference.openai_engine import OpenAIEngine
from app.services.inference.base import GenerationConfig

class TestOpenAIEngine:
    def test_initialization(self):
        engine = OpenAIEngine(base_url="http://test.local/v1", api_key="test_key", model_name="test-model")
        assert engine.model_name == "test-model"
        assert engine.is_loaded is True

    def test_missing_base_url(self):
        with pytest.raises(ValueError):
            OpenAIEngine(base_url="", api_key="test_key", model_name="test-model")

    @patch('app.services.inference.openai_engine.OpenAI')
    def test_generate_success(self, mock_openai):
        # Setup mock
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Here is a test response."
        mock_choice.finish_reason = "stop"
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 6
        
        mock_client.chat.completions.create.return_value = mock_response
        
        engine = OpenAIEngine(base_url="http://test.local/v1", api_key="key", model_name="test-model")
        config = GenerationConfig(temperature=0.5, max_tokens=100, top_p=0.9, seed=42)
        
        result = engine.generate("Hello world", config)
        
        assert result.text == "Here is a test response."
        assert result.tokens_input == 5
        assert result.tokens_output == 6
        assert result.finish_reason == "stop"
        
        # Verify OpenAI called correctly
        mock_client.chat.completions.create.assert_called_once_with(
            model="test-model",
            messages=[{"role": "user", "content": "Hello world"}],
            max_tokens=100,
            temperature=0.5,
            top_p=0.9,
            seed=42
        )

    @patch('app.services.inference.openai_engine.OpenAI')
    # Note: Because of tenacious retry, we might want to mock time or disable retry, but let's test a ValueError so it doesn't retry
    def test_generate_failure(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.side_effect = ValueError("API Error")
        
        engine = OpenAIEngine(base_url="http://test.local/v1", api_key="key", model_name="test-model")
        config = GenerationConfig(temperature=0.5, max_tokens=100, top_p=0.9)
        
        with pytest.raises(Exception):
            engine.generate("Hello world", config)

    @patch('app.services.inference.openai_engine.OpenAIEngine.generate')
    def test_generate_batch(self, mock_generate):
        mock_generate.side_effect = [
            MagicMock(text="Response 1"),
            MagicMock(text="Response 2")
        ]
        
        engine = OpenAIEngine(base_url="http://test.local/v1", api_key="key", model_name="test-model")
        config = GenerationConfig(temperature=0.5, max_tokens=100, top_p=0.9)
        
        results = engine.generate_batch(["Prompt 1", "Prompt 2"], config)
        
        assert len(results) == 2
        # Since order is not guaranteed with ThreadPoolExecutor, we check texts
        texts = [r.text for r in results]
        assert "Response 1" in texts
        assert "Response 2" in texts
