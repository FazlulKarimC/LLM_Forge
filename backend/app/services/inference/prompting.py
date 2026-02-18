"""
Prompt Templates

Prompt formatting utilities for different reasoning methods.
- NaivePromptTemplate: Simple Q&A (Phase 2)
- CoTPromptTemplate: Chain-of-Thought reasoning (Phase 4)
- RAGPromptTemplate: Context-augmented generation (Phase 5)
- ReActPromptTemplate: Agent answer extraction (Phase 6)
"""

import re
from typing import Optional, List, Dict


class NaivePromptTemplate:
    """
    Naive prompt template for simple Q&A.
    
    Format: Question: {question}\nAnswer:
    """
    
    @staticmethod
    def format(question: str) -> str:
        """
        Format a question into a naive prompt.
        
        Args:
            question: The question to ask
        
        Returns:
            Formatted prompt
        """
        return f"Question: {question}\nAnswer:"
    
    @staticmethod
    def parse_response(response: str) -> str:
        """
        Parse and clean the model's response.
        
        Handles edge cases like extra whitespace, newlines, etc.
        
        Args:
            response: Raw model output
        
        Returns:
            Cleaned answer
        """
        # Strip leading/trailing whitespace
        cleaned = response.strip()
        
        # If response contains newlines, take only the first line
        # (model sometimes continues generating)
        if "\n" in cleaned:
            cleaned = cleaned.split("\n")[0].strip()
        
        # Remove common artifacts
        cleaned = cleaned.replace("Answer:", "").strip()
        
        # Handle empty responses
        if not cleaned:
            return "[No response generated]"
        
        return cleaned


class CoTPromptTemplate:
    """
    Chain-of-Thought prompt template for reasoning tasks.
    
    Supports:
    - Zero-shot CoT: Appends "Let's think step by step" trigger
    - Few-shot CoT: Prepends worked examples before the question
    
    Answer extraction looks for explicit patterns like
    "The answer is", "Answer:", "Therefore," then falls back
    to the last sentence.
    """
    
    COT_TRIGGER = "Let's think step by step."
    
    # Patterns for extracting the final answer (order = priority)
    ANSWER_PATTERNS = [
        r"[Tt]he\s+answer\s+is[:\s]+(.+?)[\.\n]",
        r"[Aa]nswer[:\s]+(.+?)[\.\n]",
        r"[Tt]herefore[,:\s]+(?:the answer is\s+)?(.+?)[\.\n]",
        r"[Ss]o[,:\s]+the answer is\s+(.+?)[\.\n]",
        r"[Ii]n conclusion[,:\s]+(.+?)[\.\n]",
    ]
    
    @staticmethod
    def format(
        question: str,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Format a question into a CoT prompt.
        
        Args:
            question: The question to ask
            few_shot_examples: Optional list of dicts with keys:
                - question: example question
                - reasoning: step-by-step reasoning
                - answer: final answer
        
        Returns:
            Formatted prompt with CoT trigger
        """
        parts: list[str] = []
        
        # Few-shot examples (if provided)
        if few_shot_examples:
            for ex in few_shot_examples:
                parts.append(
                    f"Question: {ex['question']}\n"
                    f"{CoTPromptTemplate.COT_TRIGGER}\n"
                    f"{ex['reasoning']}\n"
                    f"The answer is {ex['answer']}.\n"
                )
        
        # Target question
        parts.append(
            f"Question: {question}\n"
            f"{CoTPromptTemplate.COT_TRIGGER}\n"
        )
        
        return "\n".join(parts)
    
    @staticmethod
    def parse_response(response: str) -> str:
        """
        Extract the final answer from a CoT reasoning chain.
        
        Strategy:
        1. Try regex patterns: "The answer is ...", "Answer: ...", 
           "Therefore, ...", etc.
        2. Fallback: take the last non-empty sentence.
        3. Clean artifacts and whitespace.
        
        Args:
            response: Raw model output (may contain multi-line reasoning)
        
        Returns:
            Extracted and cleaned answer string
        """
        if not response or not response.strip():
            return "[No response generated]"
        
        text = response.strip()
        
        # Ensure text ends with a period for pattern matching
        text_for_matching = text if text.endswith(".") else text + "."
        
        # Try each pattern in priority order
        for pattern in CoTPromptTemplate.ANSWER_PATTERNS:
            matches = re.findall(pattern, text_for_matching)
            if matches:
                # Take the last match (final answer in the chain)
                answer = matches[-1].strip()
                answer = CoTPromptTemplate._clean_answer(answer)
                if answer:
                    return answer
        
        # Fallback: last non-empty sentence
        sentences = re.split(r'[.!?]\s+', text)
        for sentence in reversed(sentences):
            cleaned = sentence.strip().rstrip(".!?").strip()
            if cleaned and len(cleaned) > 1:
                return CoTPromptTemplate._clean_answer(cleaned)
        
        # Ultimate fallback: return cleaned full text, first line
        first_line = text.split("\n")[0].strip()
        cleaned = CoTPromptTemplate._clean_answer(first_line)
        return cleaned if cleaned else "[No response generated]"
    
    @staticmethod
    def _clean_answer(answer: str) -> str:
        """
        Clean extracted answer text.
        
        Removes common artifacts, extra whitespace, and trailing punctuation.
        """
        # Remove markdown-style formatting
        cleaned = answer.strip()
        cleaned = cleaned.strip("*_`\"'")
        
        # Remove trailing punctuation
        cleaned = cleaned.rstrip(".,;:!?")
        
        # Remove common prefixes
        for prefix in ["the answer is", "answer:", "therefore,"]:
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        
        return cleaned.strip()


class RAGPromptTemplate:
    """
    RAG prompt template for context-augmented generation (Phase 5).
    
    Prepends retrieved context chunks before the question.
    Uses the same parse logic as NaivePromptTemplate.
    """

    @staticmethod
    def format(question: str, context_chunks: List[str]) -> str:
        """
        Format a question with retrieved context.
        
        Args:
            question: The question to ask
            context_chunks: List of relevant text chunks from retrieval
        
        Returns:
            Formatted prompt with context
        """
        if not context_chunks:
            return NaivePromptTemplate.format(question)

        context = "\n\n".join(context_chunks)
        return (
            f"Context:\n{context}\n\n"
            f"Based on the context above, answer the following question.\n"
            f"Question: {question}\nAnswer:"
        )

    @staticmethod
    def parse_response(response: str) -> str:
        """Parse response — same as NaivePromptTemplate."""
        return NaivePromptTemplate.parse_response(response)


class ReActPromptTemplate:
    """
    ReAct prompt template for agent-based reasoning (Phase 6).
    
    The actual prompt building is handled by ReActAgent._build_system_prompt.
    This class provides parse_response to extract the final answer.
    """

    @staticmethod
    def parse_response(response: str) -> str:
        """
        Extract the final answer from agent output.
        
        Looks for 'Answer: ...' pattern first, then falls back
        to CoT-style extraction.
        """
        if not response or not response.strip():
            return "[No response generated]"
        
        text = response.strip()
        
        # Try Answer: pattern (ReAct format)
        answer_match = re.search(r"Answer:\s*(.+?)(?:\n|$)", text, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip().rstrip(".,;:!?")
            if answer:
                return answer
        
        # Fallback to CoT-style parsing
        return CoTPromptTemplate.parse_response(text)


