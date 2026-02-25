"""
Agent Service (Phase 6)

ReAct (Reasoning + Acting) agent implementation.
Supports tool-using agents with Thought/Action/Observation loop.

Tools:
- WikipediaSearchTool: Search Wikipedia via REST API (cached)
- CalculatorTool: Safe math evaluation via Python AST
- RetrievalTool: Vector search via existing Qdrant/RAG pipeline

Architecture:
    Question → [ReAct Loop] → Thought → Action(tool) → Observation
             → Thought → Action(tool) → Observation
             → Thought → Answer
"""

import ast
import hashlib
import json
import logging
import operator
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ToolResult:
    """Result from a tool execution."""
    output: str
    success: bool
    execution_time_ms: float = 0.0


@dataclass
class TraceStep:
    """A single step in the agent's trace."""
    step: int
    thought: Optional[str] = None
    action: Optional[str] = None
    action_input: Optional[str] = None
    observation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation,
        }


@dataclass
class AgentResult:
    """Final result from agent execution."""
    answer: str
    trace: List[TraceStep]
    tool_calls: int
    total_iterations: int
    success: bool  # True if agent reached an answer, False if forced stop
    termination_reason: str  # "answer", "max_iterations", "loop_detected", "error"
    total_latency_ms: float = 0.0
    total_tokens_input: int = 0
    total_tokens_output: int = 0
    
    def trace_as_dict(self) -> List[Dict[str, Any]]:
        return [step.to_dict() for step in self.trace]


# =============================================================================
# Tool Base Class
# =============================================================================

class Tool(ABC):
    """Base class for agent tools."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name (used in Action: lines)."""
        ...
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description for the prompt."""
        ...
    
    @abstractmethod
    def execute(self, input_text: str) -> ToolResult:
        """Execute the tool with the given input."""
        ...


# =============================================================================
# Wikipedia Search Tool
# =============================================================================

class WikipediaSearchTool(Tool):
    """
    Search Wikipedia via REST API.
    
    Returns the first few sentences of the most relevant article.
    Caches results to disk to avoid rate limiting.
    """
    
    SEARCH_URL = "https://en.wikipedia.org/w/api.php"
    
    def __init__(self, cache_dir: Optional[str] = None, max_sentences: int = 3):
        if cache_dir is None:
            from app.core.config import settings
            cache_dir = str(settings.data_dir / "cache" / "wiki")
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._max_sentences = max_sentences
    
    @property
    def name(self) -> str:
        return "wikipedia_search"
    
    @property
    def description(self) -> str:
        return (
            "Search Wikipedia for factual information. "
            "Input should be a search query string. "
            "Returns the first few sentences of the most relevant article."
        )
    
    def execute(self, input_text: str) -> ToolResult:
        start = time.perf_counter()
        query = input_text.strip().strip('"\'')
        
        if not query:
            return ToolResult(
                output="Error: Empty search query",
                success=False,
                execution_time_ms=0.0,
            )
        
        # Check cache
        cache_key = hashlib.md5(query.lower().encode()).hexdigest()
        cache_file = self._cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                elapsed = (time.perf_counter() - start) * 1000
                return ToolResult(
                    output=cached["text"],
                    success=True,
                    execution_time_ms=elapsed,
                )
            except Exception:
                pass  # Cache miss, proceed with API call
        
        try:
            # Step 1: Search for the page
            search_params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": 1,
                "format": "json",
            }
            resp = requests.get(
                self.SEARCH_URL, params=search_params, timeout=10
            )
            resp.raise_for_status()
            data = resp.json()
            
            results = data.get("query", {}).get("search", [])
            if not results:
                elapsed = (time.perf_counter() - start) * 1000
                return ToolResult(
                    output=f"No Wikipedia article found for: {query}",
                    success=False,
                    execution_time_ms=elapsed,
                )
            
            title = results[0]["title"]
            
            # Step 2: Get the extract (first few sentences)
            extract_params = {
                "action": "query",
                "titles": title,
                "prop": "extracts",
                "exintro": True,
                "explaintext": True,
                "exsentences": self._max_sentences,
                "format": "json",
            }
            resp = requests.get(
                self.SEARCH_URL, params=extract_params, timeout=10
            )
            resp.raise_for_status()
            pages = resp.json().get("query", {}).get("pages", {})
            
            extract = ""
            for page in pages.values():
                extract = page.get("extract", "")
                break
            
            if not extract:
                extract = f"Wikipedia article '{title}' found but no extract available."
            
            # Cache the result
            try:
                cache_file.write_text(
                    json.dumps({"query": query, "title": title, "text": extract}),
                    encoding="utf-8",
                )
            except Exception:
                pass  # Non-critical
            
            elapsed = (time.perf_counter() - start) * 1000
            return ToolResult(
                output=extract,
                success=True,
                execution_time_ms=elapsed,
            )
        
        except requests.RequestException as e:
            elapsed = (time.perf_counter() - start) * 1000
            return ToolResult(
                output=f"Wikipedia API error: {str(e)}",
                success=False,
                execution_time_ms=elapsed,
            )


# =============================================================================
# Calculator Tool (Safe AST-based)
# =============================================================================

class CalculatorTool(Tool):
    """
    Safe math calculator using Python's AST module.
    
    Supports: +, -, *, /, **, %, parentheses, int/float literals.
    Does NOT use eval() — parses the expression tree safely.
    """
    
    # Allowed binary operators
    _OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    
    # Allowed unary operators
    _UNARY_OPS = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }
    
    @property
    def name(self) -> str:
        return "calculator"
    
    @property
    def description(self) -> str:
        return (
            "Evaluate a mathematical expression. "
            "Input should be a valid math expression like '2 + 3 * 4' or '(10 / 2) ** 3'. "
            "Supports: +, -, *, /, **, %, parentheses."
        )
    
    def execute(self, input_text: str) -> ToolResult:
        start = time.perf_counter()
        expr = input_text.strip().strip('"\'')
        
        if not expr:
            return ToolResult(
                output="Error: Empty expression",
                success=False,
                execution_time_ms=0.0,
            )
        
        try:
            result = self._safe_eval(expr)
            elapsed = (time.perf_counter() - start) * 1000
            
            # Format nicely
            if isinstance(result, float) and result == int(result):
                formatted = str(int(result))
            else:
                formatted = str(result)
            
            return ToolResult(
                output=formatted,
                success=True,
                execution_time_ms=elapsed,
            )
        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            elapsed = (time.perf_counter() - start) * 1000
            return ToolResult(
                output=f"Calculation error: {str(e)}",
                success=False,
                execution_time_ms=elapsed,
            )
    
    def _safe_eval(self, expr: str) -> float:
        """Safely evaluate a math expression using AST."""
        # Clean up common patterns
        expr = expr.replace("^", "**")  # caret → power
        expr = expr.replace("×", "*").replace("÷", "/")
        
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError:
            raise ValueError(f"Invalid expression: {expr}")
        
        return self._eval_node(tree.body)
    
    def _eval_node(self, node: ast.AST) -> float:
        """Recursively evaluate an AST node."""
        if isinstance(node, ast.Constant):  # Python 3.8+
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError(f"Unsupported constant: {node.value}")
        
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in self._OPS:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            # Safety: limit power operations
            if op_type == ast.Pow and abs(right) > 100:
                raise ValueError("Exponent too large (max 100)")
            return self._OPS[op_type](left, right)
        
        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in self._UNARY_OPS:
                raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
            operand = self._eval_node(node.operand)
            return self._UNARY_OPS[op_type](operand)
        
        else:
            raise ValueError(f"Unsupported expression element: {type(node).__name__}")


# =============================================================================
# Retrieval Tool (Qdrant/RAG)
# =============================================================================

class RetrievalTool(Tool):
    """
    Search the knowledge base using the existing RAG pipeline.
    
    Wraps QdrantStore + EmbeddingService for vector search.
    """
    
    def __init__(self, rag_pipeline=None, top_k: int = 3):
        self._pipeline = rag_pipeline
        self._top_k = top_k
    
    @property
    def name(self) -> str:
        return "retrieval"
    
    @property
    def description(self) -> str:
        return (
            "Search the knowledge base for relevant information. "
            "Input should be a search query. "
            "Returns the most relevant text passages from the knowledge base."
        )
    
    def execute(self, input_text: str) -> ToolResult:
        start = time.perf_counter()
        query = input_text.strip().strip('"\'')
        
        if not query:
            return ToolResult(
                output="Error: Empty search query",
                success=False,
                execution_time_ms=0.0,
            )
        
        if self._pipeline is None:
            elapsed = (time.perf_counter() - start) * 1000
            return ToolResult(
                output="Retrieval tool not available (no RAG pipeline configured)",
                success=False,
                execution_time_ms=elapsed,
            )
        
        try:
            result = self._pipeline.retrieve(
                question=query,
                method="naive",
                top_k=self._top_k,
            )
            
            if not result.chunks:
                elapsed = (time.perf_counter() - start) * 1000
                return ToolResult(
                    output="No relevant documents found.",
                    success=False,
                    execution_time_ms=elapsed,
                )
            
            # Format chunks as text
            passages = []
            for i, chunk in enumerate(result.chunks, 1):
                passages.append(f"[{i}] ({chunk.title}): {chunk.text[:300]}")
            
            elapsed = (time.perf_counter() - start) * 1000
            return ToolResult(
                output="\n".join(passages),
                success=True,
                execution_time_ms=elapsed,
            )
        
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return ToolResult(
                output=f"Retrieval error: {str(e)}",
                success=False,
                execution_time_ms=elapsed,
            )


# =============================================================================
# ReAct Agent
# =============================================================================

class ReActAgent:
    """
    ReAct (Reasoning + Acting) agent.
    
    Runs a Thought/Action/Observation loop to answer questions
    using available tools. Implements loop detection and trace logging.
    
    ReAct format:
        Thought: I need to find information about X.
        Action: wikipedia_search("X")
        Observation: [tool output]
        Thought: Now I know the answer.
        Answer: [final answer]
    """
    
    def __init__(
        self,
        engine,
        tools: List[Tool],
        max_iterations: int = 5,
        gen_config=None,
    ):
        """
        Args:
            engine: InferenceEngine for LLM calls
            tools: List of available Tool objects
            max_iterations: Maximum Thought/Action loops before forced stop
            gen_config: GenerationConfig for LLM calls
        """
        self._engine = engine
        self._tools = {tool.name: tool for tool in tools}
        self._max_iterations = max_iterations
        self._gen_config = gen_config
    
    def _build_system_prompt(self, question: str, history: List[TraceStep]) -> str:
        """Build the full prompt with tools description, examples, and history."""
        # Tool descriptions
        tool_desc = "\n".join(
            f"  - {t.name}: {t.description}" for t in self._tools.values()
        )
        
        # Build the prompt
        prompt_parts = [
            "You are a helpful AI assistant that can use tools to answer questions.",
            "You have access to the following tools:",
            tool_desc,
            "",
            "To use a tool, respond in this EXACT format:",
            'Thought: [your reasoning about what to do next]',
            'Action: tool_name("input")',
            "",
            "After receiving the observation, continue reasoning.",
            "When you have enough information to answer, respond:",
            'Thought: I now know the answer.',
            'Answer: [your final answer]',
            "",
            "IMPORTANT RULES:",
            "- Always start with a Thought.",
            "- Use exactly one Action per step.",
            "- Give a concise final Answer.",
            "- Do NOT repeat the same action with the same input.",
            "",
            "Example:",
            'Thought: I need to find when the Eiffel Tower was built.',
            'Action: wikipedia_search("Eiffel Tower construction date")',
            'Observation: The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It was constructed from 1887 to 1889.',
            'Thought: I now know the answer.',
            'Answer: The Eiffel Tower was constructed from 1887 to 1889.',
            "",
            f"Now answer this question: {question}",
        ]
        
        # Add history
        if history:
            prompt_parts.append("")
            for step in history:
                if step.thought:
                    prompt_parts.append(f"Thought: {step.thought}")
                if step.action and step.action_input is not None:
                    prompt_parts.append(f'Action: {step.action}("{step.action_input}")')
                if step.observation:
                    prompt_parts.append(f"Observation: {step.observation}")
        
        return "\n".join(prompt_parts)
    
    def _parse_response(self, text: str) -> Dict[str, Optional[str]]:
        """
        Parse the LLM response to extract Thought, Action, and Answer.
        
        Returns dict with keys: thought, action, action_input, answer
        """
        result = {
            "thought": None,
            "action": None,
            "action_input": None,
            "answer": None,
        }
        
        # Extract Thought
        thought_match = re.search(r"Thought:\s*(.+?)(?=\n(?:Action|Answer)|$)", text, re.DOTALL)
        if thought_match:
            result["thought"] = thought_match.group(1).strip()
        
        # Extract Answer (check this before Action)
        answer_match = re.search(r"Answer:\s*(.+?)(?=\n|$)", text, re.DOTALL)
        if answer_match:
            result["answer"] = answer_match.group(1).strip()
            return result  # If we have an answer, stop
        
        # Extract Action: tool_name("input") or tool_name(input)
        action_match = re.search(
            r'Action:\s*(\w+)\s*\(\s*["\']?(.*?)["\']?\s*\)',
            text
        )
        if action_match:
            result["action"] = action_match.group(1).strip()
            result["action_input"] = action_match.group(2).strip()
        else:
            # Try simpler format: Action: tool_name input
            simple_match = re.search(r"Action:\s*(\w+)\s+(.+?)(?=\n|$)", text)
            if simple_match:
                result["action"] = simple_match.group(1).strip()
                result["action_input"] = simple_match.group(2).strip().strip('"\'')
        
        return result
    
    def _detect_loop(self, trace: List[TraceStep]) -> bool:
        """Detect if the agent is stuck in a loop (same action 3+ times)."""
        if len(trace) < 3:
            return False
        
        # Check last 3 actions
        recent_actions = []
        for step in trace[-3:]:
            if step.action and step.action_input is not None:
                recent_actions.append(f"{step.action}:{step.action_input}")
        
        if len(recent_actions) >= 3 and len(set(recent_actions)) == 1:
            return True
        
        return False
    
    def run(self, question: str) -> AgentResult:
        """
        Run the ReAct loop to answer a question.
        
        Args:
            question: The question to answer
            
        Returns:
            AgentResult with answer, trace, and metadata
        """
        start_time = time.perf_counter()
        trace: List[TraceStep] = []
        tool_call_count = 0
        total_tokens_in = 0
        total_tokens_out = 0
        
        for iteration in range(self._max_iterations):
            # Check for loops
            if self._detect_loop(trace):
                logger.warning(f"Loop detected at iteration {iteration + 1}")
                answer = self._extract_best_answer(trace)
                return AgentResult(
                    answer=answer,
                    trace=trace,
                    tool_calls=tool_call_count,
                    total_iterations=iteration + 1,
                    success=False,
                    termination_reason="loop_detected",
                    total_latency_ms=(time.perf_counter() - start_time) * 1000,
                    total_tokens_input=total_tokens_in,
                    total_tokens_output=total_tokens_out,
                )
            
            # Build prompt and generate
            prompt = self._build_system_prompt(question, trace)
            
            try:
                gen_result = self._engine.generate(prompt, self._gen_config)
                total_tokens_in += gen_result.tokens_input
                total_tokens_out += gen_result.tokens_output
            except Exception as e:
                logger.error(f"LLM generation failed at iteration {iteration + 1}: {e}")
                answer = self._extract_best_answer(trace)
                return AgentResult(
                    answer=answer,
                    trace=trace,
                    tool_calls=tool_call_count,
                    total_iterations=iteration + 1,
                    success=False,
                    termination_reason="error",
                    total_latency_ms=(time.perf_counter() - start_time) * 1000,
                    total_tokens_input=total_tokens_in,
                    total_tokens_output=total_tokens_out,
                )
            
            # Parse the response
            parsed = self._parse_response(gen_result.text)
            
            step = TraceStep(
                step=iteration + 1,
                thought=parsed["thought"],
            )
            
            # Check for final answer
            if parsed["answer"]:
                step.action = None
                step.observation = None
                trace.append(step)
                
                return AgentResult(
                    answer=parsed["answer"],
                    trace=trace,
                    tool_calls=tool_call_count,
                    total_iterations=iteration + 1,
                    success=True,
                    termination_reason="answer",
                    total_latency_ms=(time.perf_counter() - start_time) * 1000,
                    total_tokens_input=total_tokens_in,
                    total_tokens_output=total_tokens_out,
                )
            
            # Execute action
            if parsed["action"] and parsed["action"] in self._tools:
                step.action = parsed["action"]
                step.action_input = parsed["action_input"] or ""
                
                tool = self._tools[parsed["action"]]
                tool_result = tool.execute(step.action_input)
                step.observation = tool_result.output
                tool_call_count += 1
                
                logger.info(
                    f"Step {iteration + 1}: {step.action}({step.action_input}) "
                    f"→ {'OK' if tool_result.success else 'FAIL'} "
                    f"({tool_result.execution_time_ms:.0f}ms)"
                )
            elif parsed["action"]:
                # Unknown tool
                step.action = parsed["action"]
                step.action_input = parsed.get("action_input", "")
                step.observation = (
                    f"Error: Unknown tool '{parsed['action']}'. "
                    f"Available tools: {', '.join(self._tools.keys())}"
                )
            else:
                # No action and no answer — model is confused
                step.observation = (
                    "Please respond with either an Action using one of the "
                    f"available tools ({', '.join(self._tools.keys())}), "
                    "or provide your final Answer."
                )
            
            trace.append(step)
        
        # Max iterations reached
        answer = self._extract_best_answer(trace)
        return AgentResult(
            answer=answer,
            trace=trace,
            tool_calls=tool_call_count,
            total_iterations=self._max_iterations,
            success=False,
            termination_reason="max_iterations",
            total_latency_ms=(time.perf_counter() - start_time) * 1000,
            total_tokens_input=total_tokens_in,
            total_tokens_output=total_tokens_out,
        )
    
    def _extract_best_answer(self, trace: List[TraceStep]) -> str:
        """Extract the best answer from trace when agent didn't conclude."""
        # Try to find any Answer-like text in the last thought
        if trace:
            last = trace[-1]
            if last.thought:
                # Check if the thought contains an answer
                answer_match = re.search(
                    r"(?:the answer is|answer:)\s*(.+?)(?:\.|$)",
                    last.thought,
                    re.IGNORECASE,
                )
                if answer_match:
                    return answer_match.group(1).strip()
                return last.thought
            
            # Use last observation as fallback
            if last.observation:
                return last.observation[:200]
        
        return "[Agent could not determine an answer]"
