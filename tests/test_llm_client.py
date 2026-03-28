"""
Tests for LLMClient provider routing.
All provider SDKs are mocked — no real API calls are made.
"""
import sys
from unittest.mock import MagicMock, patch

import pytest

from agent.llm_reasoner import LLMClient


def _make_genai_mock(response_text: str = "gemini response") -> MagicMock:
    mock_genai = MagicMock()
    mock_response = MagicMock()
    mock_response.text = response_text
    mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response
    return mock_genai


def _make_openai_mock(response_text: str = "openai response") -> MagicMock:
    mock_openai = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = response_text
    mock_openai.OpenAI.return_value.chat.completions.create.return_value.choices = [mock_choice]
    return mock_openai


def _make_anthropic_mock(response_text: str = "anthropic response") -> MagicMock:
    mock_anthropic = MagicMock()
    mock_content = MagicMock()
    mock_content.text = response_text
    mock_anthropic.Anthropic.return_value.messages.create.return_value.content = [mock_content]
    return mock_anthropic


def _make_hf_mock(response_text: str = "huggingface response") -> MagicMock:
    mock_hf = MagicMock()
    mock_hf.InferenceClient.return_value.text_generation.return_value = response_text
    return mock_hf


class TestLLMClientProviderRouting:
    def test_gemini_routes_to_generative_model(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        mock_genai = _make_genai_mock()
        with patch.dict(sys.modules, {"google.generativeai": mock_genai}):
            result = LLMClient().complete("sys", "user")
        mock_genai.GenerativeModel.assert_called_once_with("gemini-2.0-flash")
        assert result == "gemini response"

    def test_openai_routes_to_openai_client(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        mock_openai = _make_openai_mock()
        with patch.dict(sys.modules, {"openai": mock_openai}):
            result = LLMClient().complete("sys", "user")
        mock_openai.OpenAI.assert_called_once()
        assert result == "openai response"

    def test_anthropic_routes_to_anthropic_client(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        mock_anthropic = _make_anthropic_mock()
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            result = LLMClient().complete("sys", "user")
        mock_anthropic.Anthropic.assert_called_once()
        assert result == "anthropic response"

    def test_huggingface_routes_to_inference_client(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "huggingface")
        mock_hf = _make_hf_mock()
        with patch.dict(sys.modules, {"huggingface_hub": mock_hf}):
            result = LLMClient().complete("sys", "user")
        mock_hf.InferenceClient.assert_called_once()
        assert result == "huggingface response"

    def test_unknown_provider_raises_value_error(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "unknown_provider")
        with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
            LLMClient().complete("sys", "user")

    def test_default_provider_is_gemini_when_env_not_set(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        mock_genai = _make_genai_mock()
        with patch.dict(sys.modules, {"google.generativeai": mock_genai}):
            result = LLMClient().complete("sys", "user")
        mock_genai.GenerativeModel.assert_called_once_with("gemini-2.0-flash")
        assert result == "gemini response"
