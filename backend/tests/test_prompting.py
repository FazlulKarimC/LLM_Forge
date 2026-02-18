"""
Tests for prompt templates (Naive + CoT).

Covers formatting, parsing, and edge cases for both prompt strategies.
"""

import json
import os
import pytest
from app.services.inference.prompting import NaivePromptTemplate, CoTPromptTemplate


# =============================================================================
# NaivePromptTemplate Tests
# =============================================================================

class TestNaivePromptTemplate:
    def test_format(self):
        result = NaivePromptTemplate.format("What is Python?")
        assert result == "Question: What is Python?\nAnswer:"
    
    def test_parse_simple(self):
        assert NaivePromptTemplate.parse_response("Python is a programming language") == "Python is a programming language"
    
    def test_parse_strips_whitespace(self):
        assert NaivePromptTemplate.parse_response("  hello  ") == "hello"
    
    def test_parse_takes_first_line(self):
        assert NaivePromptTemplate.parse_response("Paris\nFrance\nEurope") == "Paris"
    
    def test_parse_removes_answer_prefix(self):
        assert NaivePromptTemplate.parse_response("Answer: Tokyo") == "Tokyo"
    
    def test_parse_empty(self):
        assert NaivePromptTemplate.parse_response("") == "[No response generated]"
        assert NaivePromptTemplate.parse_response("   ") == "[No response generated]"


# =============================================================================
# CoTPromptTemplate Tests
# =============================================================================

class TestCoTPromptTemplate:
    # --- Format Tests ---
    
    def test_format_zero_shot(self):
        result = CoTPromptTemplate.format("What is the capital of France?")
        assert "Question: What is the capital of France?" in result
        assert "Let's think step by step." in result
    
    def test_format_few_shot(self):
        examples = [
            {
                "question": "What is 2+2?",
                "reasoning": "2 plus 2 equals 4.",
                "answer": "4",
            }
        ]
        result = CoTPromptTemplate.format("What is 3+3?", examples)
        assert "Question: What is 2+2?" in result
        assert "2 plus 2 equals 4." in result
        assert "The answer is 4." in result
        assert "Question: What is 3+3?" in result
    
    def test_format_multiple_examples(self):
        examples = [
            {"question": "Q1", "reasoning": "R1", "answer": "A1"},
            {"question": "Q2", "reasoning": "R2", "answer": "A2"},
        ]
        result = CoTPromptTemplate.format("Q3", examples)
        assert "Question: Q1" in result
        assert "Question: Q2" in result
        assert "Question: Q3" in result
        assert result.count("Let's think step by step.") == 3
    
    def test_format_no_examples(self):
        result = CoTPromptTemplate.format("Test?", None)
        assert "Question: Test?" in result
        assert "Let's think step by step." in result
    
    # --- Parse Tests ---
    
    def test_parse_answer_is_pattern(self):
        response = "Let me think. The capital is in Europe. The answer is Paris."
        assert CoTPromptTemplate.parse_response(response) == "Paris"
    
    def test_parse_therefore_pattern(self):
        response = "The wall fell in 1989. Therefore, the answer is 1989."
        assert CoTPromptTemplate.parse_response(response) == "1989"
    
    def test_parse_answer_colon_pattern(self):
        response = "After reasoning... Answer: Gold"
        assert CoTPromptTemplate.parse_response(response) == "Gold"
    
    def test_parse_so_pattern(self):
        response = "Mars is red due to iron oxide. So, the answer is Mars."
        assert CoTPromptTemplate.parse_response(response) == "Mars"
    
    def test_parse_last_match_wins(self):
        # When multiple patterns match, last one should win (final answer)
        response = "The answer is not Mercury. The answer is Venus."
        assert CoTPromptTemplate.parse_response(response) == "Venus"
    
    def test_parse_fallback_last_sentence(self):
        response = "Thinking about this carefully. The element is called gold"
        assert CoTPromptTemplate.parse_response(response) == "The element is called gold"
    
    def test_parse_empty(self):
        assert CoTPromptTemplate.parse_response("") == "[No response generated]"
        assert CoTPromptTemplate.parse_response("   ") == "[No response generated]"
        assert CoTPromptTemplate.parse_response(None) == "[No response generated]"
    
    def test_parse_cleans_artifacts(self):
        response = "The answer is **Canberra**."
        result = CoTPromptTemplate.parse_response(response)
        assert result == "Canberra"
    
    def test_parse_multiline_reasoning(self):
        response = """First, let me consider the facts.
The Berlin Wall was built in 1961.
It divided East and West Berlin.
The wall fell on November 9, 1989.
Therefore, the answer is 1989."""
        assert CoTPromptTemplate.parse_response(response) == "1989"
    
    # --- Clean Answer Tests ---
    
    def test_clean_answer_strips_punctuation(self):
        assert CoTPromptTemplate._clean_answer("Paris.") == "Paris"
        assert CoTPromptTemplate._clean_answer("Paris!") == "Paris"
        assert CoTPromptTemplate._clean_answer("Paris,") == "Paris"
    
    def test_clean_answer_strips_formatting(self):
        assert CoTPromptTemplate._clean_answer("**Paris**") == "Paris"
        assert CoTPromptTemplate._clean_answer("`Paris`") == "Paris"
    
    def test_clean_answer_removes_prefix(self):
        assert CoTPromptTemplate._clean_answer("The answer is Paris") == "Paris"
        assert CoTPromptTemplate._clean_answer("Answer: Paris") == "Paris"


# =============================================================================
# Integration: CoT examples file
# =============================================================================

class TestCoTExamplesFile:
    def test_cot_examples_file_exists(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "configs", "cot_examples.json"
        )
        assert os.path.exists(path), f"cot_examples.json not found at {path}"
    
    def test_cot_examples_valid_json(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "configs", "cot_examples.json"
        )
        with open(path, "r", encoding="utf-8") as f:
            examples = json.load(f)
        
        assert isinstance(examples, list)
        assert len(examples) >= 3
        
        for ex in examples:
            assert "question" in ex
            assert "reasoning" in ex
            assert "answer" in ex
            assert len(ex["reasoning"]) > 20  # Should be substantial reasoning
    
    def test_cot_examples_work_with_template(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "configs", "cot_examples.json"
        )
        with open(path, "r", encoding="utf-8") as f:
            examples = json.load(f)
        
        prompt = CoTPromptTemplate.format("Test question?", examples)
        assert "Test question?" in prompt
        assert "Let's think step by step." in prompt
        # Should contain all example answers
        for ex in examples:
            assert ex["answer"] in prompt
