$API = "http://localhost:8000/api/v1"
$headers = @{ "Content-Type" = "application/json" }

$experiments = @(
    @{
        name = "[TEST] Naive - TriviaQA - Phi3.5"
        description = "Baseline naive prompting on factual recall"
        config = @{
            model_name = "microsoft/Phi-3.5-mini-instruct"
            reasoning_method = "naive"
            dataset_name = "trivia_qa"
            num_samples = 5
            hyperparameters = @{ temperature = 0.1; max_tokens = 128; top_p = 0.9; seed = 42 }
        }
    },
    @{
        name = "[TEST] CoT - TriviaQA - Phi3.5"
        description = "Chain-of-thought reasoning on factual recall"
        config = @{
            model_name = "microsoft/Phi-3.5-mini-instruct"
            reasoning_method = "cot"
            dataset_name = "trivia_qa"
            num_samples = 5
            hyperparameters = @{ temperature = 0.3; max_tokens = 256; top_p = 0.9; seed = 42 }
        }
    },
    @{
        name = "[TEST] CoT - Math - Phi3.5"
        description = "Chain-of-thought on math word problems"
        config = @{
            model_name = "microsoft/Phi-3.5-mini-instruct"
            reasoning_method = "cot"
            dataset_name = "math_reasoning"
            num_samples = 5
            hyperparameters = @{ temperature = 0.1; max_tokens = 512; top_p = 0.9; seed = 42 }
        }
    },
    @{
        name = "[TEST] Naive - MultiHop - Llama3.2"
        description = "Naive baseline on multi-hop reasoning with Llama"
        config = @{
            model_name = "meta-llama/Llama-3.2-3B-Instruct"
            reasoning_method = "naive"
            dataset_name = "multi_hop"
            num_samples = 5
            hyperparameters = @{ temperature = 0.1; max_tokens = 256; top_p = 0.9; seed = 42 }
        }
    },
    @{
        name = "[TEST] CoT - MultiHop - Llama3.2"
        description = "CoT on multi-hop reasoning with Llama"
        config = @{
            model_name = "meta-llama/Llama-3.2-3B-Instruct"
            reasoning_method = "cot"
            dataset_name = "multi_hop"
            num_samples = 5
            hyperparameters = @{ temperature = 0.3; max_tokens = 512; top_p = 0.9; seed = 42 }
        }
    },
    @{
        name = "[TEST] ReAct - ReactBench - Phi3.5"
        description = "ReAct agent on tool-requiring questions"
        config = @{
            model_name = "microsoft/Phi-3.5-mini-instruct"
            reasoning_method = "react"
            dataset_name = "react_bench"
            num_samples = 5
            hyperparameters = @{ temperature = 0.3; max_tokens = 512; top_p = 0.9; seed = 42 }
            agent = @{ max_iterations = 5; tools = @("wikipedia_search", "calculator", "retrieval") }
        }
    },
    @{
        name = "[TEST] Naive - Commonsense - Phi3.5"
        description = "Naive on commonsense reasoning"
        config = @{
            model_name = "microsoft/Phi-3.5-mini-instruct"
            reasoning_method = "naive"
            dataset_name = "commonsense_qa"
            num_samples = 5
            hyperparameters = @{ temperature = 0.1; max_tokens = 128; top_p = 0.9; seed = 42 }
        }
    },
    @{
        name = "[TEST] Naive - Sample - Mistral7B"
        description = "Smoke test with Mistral 7B on sample dataset"
        config = @{
            model_name = "mistralai/Mistral-7B-Instruct-v0.3"
            reasoning_method = "naive"
            dataset_name = "sample"
            num_samples = 5
            hyperparameters = @{ temperature = 0.1; max_tokens = 256; top_p = 0.9; seed = 42 }
        }
    }
)

$created = @()
foreach ($exp in $experiments) {
    $body = $exp | ConvertTo-Json -Depth 10
    try {
        $result = Invoke-RestMethod -Uri "$API/experiments" -Method POST -Headers $headers -Body $body
        Write-Host "CREATED: $($result.name) -> $($result.id)" -ForegroundColor Green
        $created += @{ id = $result.id; name = $result.name }
    } catch {
        Write-Host "FAILED: $($exp.name) -> $_" -ForegroundColor Red
    }
    Start-Sleep -Milliseconds 200
}

Write-Host "`n=== CREATED $($created.Count) EXPERIMENTS ===" -ForegroundColor Cyan
$created | ForEach-Object { Write-Host "  $($_.id) | $($_.name)" }
