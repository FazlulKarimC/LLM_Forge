"""
Synthetic Test Dataset Generation (P2 #14)

Generates QA pairs from knowledge base chunks using a free HF instruct model.
Reduces manual effort for creating evaluation datasets.
"""

import logging
import os
import json
import random
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Budget: max QA pairs per generation request
DEFAULT_MAX_PAIRS = 30

QA_GENERATION_PROMPT = """Given the following text passage, generate {n} question-answer pairs that test understanding of the content. The questions should be specific and answerable from the passage.

**Passage:**
{passage}

Generate exactly {n} QA pairs. Respond ONLY in this JSON format:
[{{"question": "...", "answer": "..."}}, ...]"""


class SyntheticDatasetService:
    """
    Service for generating synthetic evaluation datasets from knowledge base chunks.
    
    Uses free HF Inference API to generate QA pairs from text passages.
    """
    
    def __init__(
        self,
        model_id: str = "mistralai/Mistral-7B-Instruct-v0.3",
    ):
        self.model_id = model_id
    
    async def generate_from_chunks(
        self,
        chunks: List[str],
        pairs_per_chunk: int = 3,
        max_chunks: Optional[int] = 10,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate QA pairs from knowledge base chunks.
        
        Args:
            chunks: List of text passages to generate QA pairs from
            pairs_per_chunk: Number of QA pairs to generate per chunk
            max_chunks: Maximum number of chunks to process (budget cap)
            seed: Random seed for reproducible chunk selection
            
        Returns:
            Dictionary with generated QA pairs and metadata
        """
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            return {"error": "HF_TOKEN not set", "pairs": []}
        
        if seed is not None:
            random.seed(seed)
        
        # Budget cap: limit chunks processed
        if max_chunks and len(chunks) > max_chunks:
            chunks = random.sample(chunks, max_chunks)
        
        total_pairs = min(len(chunks) * pairs_per_chunk, DEFAULT_MAX_PAIRS)
        
        logger.info(
            f"[SYNTHETIC] Generating ~{total_pairs} QA pairs "
            f"from {len(chunks)} chunks"
        )
        
        all_pairs: List[Dict[str, str]] = []
        errors = 0
        
        for i, chunk in enumerate(chunks):
            if len(all_pairs) >= DEFAULT_MAX_PAIRS:
                break
            
            try:
                pairs = await self._generate_from_chunk(
                    chunk, pairs_per_chunk, hf_token
                )
                for pair in pairs:
                    pair["source_chunk_index"] = i
                    pair["source_text"] = chunk[:200] + "..." if len(chunk) > 200 else chunk
                all_pairs.extend(pairs)
            except Exception as e:
                logger.warning(f"[SYNTHETIC] Failed on chunk {i}: {e}")
                errors += 1
        
        # Assign IDs
        for idx, pair in enumerate(all_pairs):
            pair["id"] = f"synthetic_{idx:04d}"
        
        return {
            "pairs": all_pairs[:DEFAULT_MAX_PAIRS],
            "total_generated": len(all_pairs),
            "chunks_processed": len(chunks),
            "errors": errors,
            "model": self.model_id,
            "method": "synthetic_generation",
        }
    
    async def _generate_from_chunk(
        self,
        chunk: str,
        n: int,
        hf_token: str,
    ) -> List[Dict[str, str]]:
        """Generate n QA pairs from a single text chunk."""
        import httpx
        
        prompt = QA_GENERATION_PROMPT.format(
            passage=chunk[:1500],  # Truncate for token limits
            n=n,
        )
        
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(
                f"https://api-inference.huggingface.co/models/{self.model_id}",
                headers={"Authorization": f"Bearer {hf_token}"},
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 500,
                        "temperature": 0.7,
                        "return_full_text": False,
                    },
                },
            )
            
            if resp.status_code != 200:
                raise Exception(f"HF API returned {resp.status_code}")
            
            data = resp.json()
            if isinstance(data, list) and data:
                generated = data[0].get("generated_text", "")
            else:
                return []
        
        # Parse JSON array from response
        json_start = generated.find("[")
        json_end = generated.rfind("]") + 1
        if json_start >= 0 and json_end > json_start:
            parsed = json.loads(generated[json_start:json_end])
            if isinstance(parsed, list):
                return [
                    {"question": p["question"], "answer": p["answer"]}
                    for p in parsed
                    if isinstance(p, dict) and "question" in p and "answer" in p
                ]
        
        return []
