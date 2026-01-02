"""
Prompt Templates

Simple prompt formatting utilities for different reasoning methods.
Phase 2: Only naive prompting.
"""


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
