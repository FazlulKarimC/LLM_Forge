# ADR 003: McNemar's Test for Paired Comparison

**Status**: Accepted  
**Date**: 2024-05-20

### Context
A primary feature of LLMForge is comparing reasoning methods (e.g., Naive vs. Chain-of-Thought) on identical datasets. Often, developers only report absolute accuracy differences (e.g., "CoT scored 72%, Naive scored 65%"). On small evaluation sets (N=50 to N=100), a difference of a few percentage points can easily occur purely due to random chance. 

We need a statistical method to determine if Method A is *significantly* better than Method B.

### Decision
We implemented **McNemar's test** combined with **bootstrap confidence intervals** inside the `StatisticalService`.

Why not a standard Student's t-test or Chi-squared test?
1. The outputs are binary (Correct vs. Incorrect).
2. The samples are **paired** (both Method A and Method B evaluate the exact same `example_id` in the dataset). 

McNemar's test is specifically designed for paired nominal data. It calculates significance based solely on the discordant pairs (questions where Method A succeeded but Method B failed, and vice versa).

### Consequences
- **Positive:** Provides rigorous mathematical evidence when claiming one prompting technique outperforms another.
- **Positive:** Demonstrates depth in statistical evaluation methodologies to technical interviewers.
- **Negative:** McNemar's test requires retrieving individual run comparisons (row-by-row matching via `example_id`), making the database query more complex than simply fetching aggregate averages.
