"""
Tests for Phase 6: ReAct Agent

Tests cover:
- CalculatorTool: safe math evaluation via AST
- WikipediaSearchTool: search functionality (mocked)
- RetrievalTool: knowledge base search (mocked)
- ReActAgent: parsing, loop detection, full run with mock engine
- ReActPromptTemplate: answer extraction
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from app.services.agent_service import (
    Tool,
    CalculatorTool,
    WikipediaSearchTool,
    RetrievalTool,
    ReActAgent,
    ToolResult,
    TraceStep,
    AgentResult,
)
from app.services.inference.prompting import ReActPromptTemplate


# =============================================================================
# CalculatorTool Tests
# =============================================================================

class TestCalculatorTool:
    def setup_method(self):
        self.calc = CalculatorTool()
    
    def test_basic_addition(self):
        result = self.calc.execute("2 + 3")
        assert result.success is True
        assert result.output == "5"
    
    def test_multiplication(self):
        result = self.calc.execute("7 * 8")
        assert result.success is True
        assert result.output == "56"
    
    def test_float_division(self):
        result = self.calc.execute("10 / 3")
        assert result.success is True
        assert float(result.output) == pytest.approx(3.333, abs=0.01)
    
    def test_parentheses(self):
        result = self.calc.execute("(2 + 3) * 4")
        assert result.success is True
        assert result.output == "20"
    
    def test_power(self):
        result = self.calc.execute("2 ** 10")
        assert result.success is True
        assert result.output == "1024"
    
    def test_caret_as_power(self):
        result = self.calc.execute("2 ^ 3")
        assert result.success is True
        assert result.output == "8"
    
    def test_negative_number(self):
        result = self.calc.execute("-5 + 3")
        assert result.success is True
        assert result.output == "-2"
    
    def test_modulo(self):
        result = self.calc.execute("17 % 5")
        assert result.success is True
        assert result.output == "2"
    
    def test_empty_input(self):
        result = self.calc.execute("")
        assert result.success is False
    
    def test_invalid_expression(self):
        result = self.calc.execute("hello world")
        assert result.success is False
        assert "error" in result.output.lower()
    
    def test_division_by_zero(self):
        result = self.calc.execute("10 / 0")
        assert result.success is False
    
    def test_large_exponent_rejected(self):
        result = self.calc.execute("2 ** 1000")
        assert result.success is False
        assert "too large" in result.output.lower() or "error" in result.output.lower()
    
    def test_tool_metadata(self):
        assert self.calc.name == "calculator"
        assert "math" in self.calc.description.lower()


# =============================================================================
# WikipediaSearchTool Tests (Mocked)
# =============================================================================

class TestWikipediaSearchTool:
    def setup_method(self):
        self.wiki = WikipediaSearchTool(cache_dir="./test_cache_wiki")
    
    def test_tool_metadata(self):
        assert self.wiki.name == "wikipedia_search"
        assert "wikipedia" in self.wiki.description.lower()
    
    def test_empty_query(self):
        result = self.wiki.execute("")
        assert result.success is False
    
    @patch("app.services.agent_service.requests.get")
    def test_successful_search(self, mock_get):
        # Mock search response
        search_response = MagicMock()
        search_response.json.return_value = {
            "query": {"search": [{"title": "Python (programming language)"}]}
        }
        search_response.raise_for_status = MagicMock()
        
        # Mock extract response
        extract_response = MagicMock()
        extract_response.json.return_value = {
            "query": {
                "pages": {
                    "123": {
                        "extract": "Python is a programming language."
                    }
                }
            }
        }
        extract_response.raise_for_status = MagicMock()
        
        mock_get.side_effect = [search_response, extract_response]
        
        result = self.wiki.execute("Python programming")
        assert result.success is True
        assert "Python" in result.output
    
    @patch("app.services.agent_service.requests.get")
    def test_no_results(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {"query": {"search": []}}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        result = self.wiki.execute("xyznonexistentquery123")
        assert result.success is False
    
    def teardown_method(self):
        import shutil
        import os
        if os.path.exists("./test_cache_wiki"):
            shutil.rmtree("./test_cache_wiki")


# =============================================================================
# RetrievalTool Tests
# =============================================================================

class TestRetrievalTool:
    def test_no_pipeline(self):
        tool = RetrievalTool(rag_pipeline=None)
        result = tool.execute("test query")
        assert result.success is False
        assert "not available" in result.output.lower()
    
    def test_tool_metadata(self):
        tool = RetrievalTool()
        assert tool.name == "retrieval"
        assert "knowledge base" in tool.description.lower()
    
    def test_empty_query(self):
        tool = RetrievalTool()
        result = tool.execute("")
        assert result.success is False
    
    def test_successful_retrieval(self):
        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_chunk = MagicMock()
        mock_chunk.text = "The Eiffel Tower was built in 1889."
        mock_chunk.title = "Eiffel Tower"
        mock_result = MagicMock()
        mock_result.chunks = [mock_chunk]
        mock_pipeline.retrieve.return_value = mock_result
        
        tool = RetrievalTool(rag_pipeline=mock_pipeline)
        result = tool.execute("Eiffel Tower history")
        assert result.success is True
        assert "1889" in result.output


# =============================================================================
# ReActAgent Tests
# =============================================================================

@dataclass
class MockGenerationResult:
    """Mock for GenerationResult."""
    text: str
    tokens_input: int = 100
    tokens_output: int = 50
    latency_ms: float = 100.0
    finish_reason: str = "stop"
    gpu_memory_mb: float = None


class MockEngine:
    """Mock inference engine for testing."""
    
    def __init__(self, responses: list):
        self._responses = responses
        self._call_count = 0
    
    def generate(self, prompt, config):
        if self._call_count < len(self._responses):
            text = self._responses[self._call_count]
        else:
            text = "Answer: I don't know"
        self._call_count += 1
        return MockGenerationResult(text=text)


class TestReActAgentParsing:
    """Test the ReActAgent's _parse_response method."""
    
    def setup_method(self):
        engine = MockEngine([])
        self.agent = ReActAgent(engine=engine, tools=[], gen_config=None)
    
    def test_parse_answer(self):
        text = "Thought: I know the answer.\nAnswer: The capital of France is Paris."
        result = self.agent._parse_response(text)
        assert result["answer"] == "The capital of France is Paris."
        assert result["thought"] == "I know the answer."
    
    def test_parse_action(self):
        text = 'Thought: I need to search for this.\nAction: wikipedia_search("Paris history")'
        result = self.agent._parse_response(text)
        assert result["action"] == "wikipedia_search"
        assert result["action_input"] == "Paris history"
        assert result["answer"] is None
    
    def test_parse_action_single_quotes(self):
        text = "Thought: searching.\nAction: calculator('2 + 3')"
        result = self.agent._parse_response(text)
        assert result["action"] == "calculator"
        assert result["action_input"] == "2 + 3"
    
    def test_parse_no_action_no_answer(self):
        text = "I'm thinking about this question..."
        result = self.agent._parse_response(text)
        assert result["action"] is None
        assert result["answer"] is None


class TestReActAgentLoopDetection:
    def test_no_loop_short_trace(self):
        engine = MockEngine([])
        agent = ReActAgent(engine=engine, tools=[], gen_config=None)
        
        trace = [
            TraceStep(step=1, action="wiki", action_input="test"),
        ]
        assert agent._detect_loop(trace) is False
    
    def test_loop_detected(self):
        engine = MockEngine([])
        agent = ReActAgent(engine=engine, tools=[], gen_config=None)
        
        trace = [
            TraceStep(step=1, action="wiki", action_input="same query"),
            TraceStep(step=2, action="wiki", action_input="same query"),
            TraceStep(step=3, action="wiki", action_input="same query"),
        ]
        assert agent._detect_loop(trace) is True
    
    def test_no_loop_different_actions(self):
        engine = MockEngine([])
        agent = ReActAgent(engine=engine, tools=[], gen_config=None)
        
        trace = [
            TraceStep(step=1, action="wiki", action_input="query 1"),
            TraceStep(step=2, action="calc", action_input="2 + 3"),
            TraceStep(step=3, action="wiki", action_input="query 2"),
        ]
        assert agent._detect_loop(trace) is False


class TestReActAgentRun:
    def test_direct_answer(self):
        """Agent immediately answers without tools."""
        engine = MockEngine([
            "Thought: I know this.\nAnswer: Paris"
        ])
        agent = ReActAgent(engine=engine, tools=[], max_iterations=5, gen_config=None)
        
        result = agent.run("What is the capital of France?")
        assert result.success is True
        assert result.answer == "Paris"
        assert result.tool_calls == 0
        assert result.termination_reason == "answer"
    
    def test_tool_then_answer(self):
        """Agent uses a tool then answers."""
        calc = CalculatorTool()
        engine = MockEngine([
            'Thought: I need to calculate this.\nAction: calculator("2 + 3")',
            "Thought: The result is 5.\nAnswer: 5",
        ])
        agent = ReActAgent(
            engine=engine, tools=[calc], max_iterations=5, gen_config=None
        )
        
        result = agent.run("What is 2 + 3?")
        assert result.success is True
        assert result.answer == "5"
        assert result.tool_calls == 1
    
    def test_max_iterations_reached(self):
        """Agent hits max iterations without answering."""
        engine = MockEngine([
            'Thought: Let me search.\nAction: calculator("1 + 1")',
            'Thought: Let me try again.\nAction: calculator("2 + 2")',
            'Thought: Still trying.\nAction: calculator("3 + 3")',
        ])
        calc = CalculatorTool()
        agent = ReActAgent(
            engine=engine, tools=[calc], max_iterations=3, gen_config=None
        )
        
        result = agent.run("Test question")
        assert result.success is False
        assert result.termination_reason == "max_iterations"
        assert result.total_iterations == 3
    
    def test_unknown_tool(self):
        """Agent tries to use unknown tool."""
        engine = MockEngine([
            'Thought: searching.\nAction: unknown_tool("query")',
            "Thought: I know.\nAnswer: 42",
        ])
        agent = ReActAgent(engine=engine, tools=[], max_iterations=5, gen_config=None)
        
        result = agent.run("Test")
        assert result.success is True
        assert result.answer == "42"
    
    def test_trace_serialization(self):
        """Agent trace can be serialized to dict."""
        engine = MockEngine([
            "Thought: I know.\nAnswer: 42"
        ])
        agent = ReActAgent(engine=engine, tools=[], max_iterations=5, gen_config=None)
        
        result = agent.run("Test")
        trace_dict = result.trace_as_dict()
        assert isinstance(trace_dict, list)
        assert len(trace_dict) > 0
        assert "step" in trace_dict[0]
        assert "thought" in trace_dict[0]


# =============================================================================
# ReActPromptTemplate Tests
# =============================================================================

class TestReActPromptTemplate:
    def test_parse_answer_format(self):
        response = "Some reasoning...\nAnswer: The Eiffel Tower"
        result = ReActPromptTemplate.parse_response(response)
        assert result == "The Eiffel Tower"
    
    def test_parse_empty(self):
        result = ReActPromptTemplate.parse_response("")
        assert result == "[No response generated]"
    
    def test_parse_no_answer_pattern(self):
        response = "The answer is Paris."
        result = ReActPromptTemplate.parse_response(response)
        assert "Paris" in result
    
    def test_parse_answer_with_period(self):
        response = "Answer: Paris."
        result = ReActPromptTemplate.parse_response(response)
        assert result == "Paris"
