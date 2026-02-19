param([string]$ExpId, [string]$ExpName, [int]$TimeoutSec = 300)

$API = "http://localhost:8000/api/v1"

# 1. Trigger run
try {
    $run = Invoke-RestMethod -Uri "$API/experiments/$ExpId/run" -Method POST
    Write-Host "[RUN] Triggered: $ExpName" -ForegroundColor Yellow
} catch {
    Write-Host "[RUN] Failed to trigger: $_" -ForegroundColor Red
    return @{ status = "trigger_failed"; error = "$_" }
}

# 2. Poll until done or timeout
$start = Get-Date
$lastStatus = ""
do {
    Start-Sleep -Seconds 5
    $exp = Invoke-RestMethod -Uri "$API/experiments/$ExpId"
    if ($exp.status -ne $lastStatus) {
        Write-Host "  -> status: $($exp.status)" -ForegroundColor Cyan
        $lastStatus = $exp.status
    }
    $elapsed = ((Get-Date) - $start).TotalSeconds
} while ($exp.status -in @("queued","running") -and $elapsed -lt $TimeoutSec)

# 3. Collect metrics if completed
$result = @{
    name   = $ExpName
    id     = $ExpId
    status = $exp.status
    error  = $exp.error_message
    started_at   = $exp.started_at
    completed_at = $exp.completed_at
}

if ($exp.status -eq "completed") {
    try {
        $metrics = Invoke-RestMethod -Uri "$API/results/$ExpId/metrics"
        $result.accuracy  = [math]::Round($metrics.quality.accuracy_substring * 100, 1)
        $result.f1        = [math]::Round($metrics.quality.accuracy_f1 * 100, 1)
        $result.latency_p50 = [math]::Round($metrics.performance.latency_p50, 0)
        $result.latency_p95 = [math]::Round($metrics.performance.latency_p95, 0)
        $result.total_runs  = $metrics.cost.total_runs
        $result.total_tokens = $metrics.cost.total_tokens_input + $metrics.cost.total_tokens_output
        Write-Host "[DONE] $ExpName | acc=$($result.accuracy)% f1=$($result.f1)% p50=$($result.latency_p50)ms" -ForegroundColor Green
    } catch {
        Write-Host "[WARN] Metrics fetch failed: $_" -ForegroundColor DarkYellow
    }
} elseif ($exp.status -eq "failed") {
    Write-Host "[FAIL] $ExpName | $($exp.error_message)" -ForegroundColor Red
} else {
    Write-Host "[TIMEOUT] $ExpName still $($exp.status) after $TimeoutSec s" -ForegroundColor Red
}

return $result
