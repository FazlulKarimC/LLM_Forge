# ADR 005: Budget-Capped LLM-as-Judge 

**Status**: Accepted  
**Date**: 2024-06-15

### Context
Evaluating output coherence, helpfulness, and factuality via LLM-as-a-Judge requires secondary inference passes on the primary LLM's answers. On a dataset of 100 questions, applying an LLM-judge creates 100 additional inference calls. The standard industry practice relies on GPT-4, which becomes prohibitively expensive at scale.

Given our free-tier constraints (e.g., Hugging Face Serverless endpoints, Groq rate limits), executing 100 judge passes sequentially hits throttling caps (HTTP 429), and batch parallelizing them exacerbates the rate limiting.

### Decision
We implemented a **sampled evaluation strategy** combined with a free `mistralai/Mistral-7B-Instruct-v0.3` judge.

Rather than evaluating every single generated answer, the `LLMJudgeService` randomly samples a strict limit `S` (default 20) of outputs per experiment. The final aggregate "Coherence Score" is the mean of this sampled subset rather than the complete test set.

### Consequences
- **Positive:** Bounded costs/execution times regardless of underlying dataset size (e.g., MMLU-500 versus ToyQA-20 both only trigger 20 judge calls).
- **Positive:** Avoids strict IP rate limits.
- **Negative:** Sampling introduces greater variance into the subjective scores. We validated mathematically that `N=20` provides a standard error of approximately ±0.3 points on our 5-point scale, which is acceptable for relative ranking (A vs B) but unsuitable for absolute benchmark publications.

### Constraints Addressed
- Free Tier API Budget Constraints 
- Execution time limits (the judge is a slow blocking operation if unbatched).
