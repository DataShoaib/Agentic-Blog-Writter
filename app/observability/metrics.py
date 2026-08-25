from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

GRAPH_RUNS = Counter("agentic_graph_runs_total", "Total graph executions started.")
GRAPH_COMPLETED = Counter("agentic_graph_completed_total", "Total successful graph executions.")
GRAPH_FAILURES = Counter("agentic_graph_failures_total", "Total failed graph executions.")
GRAPH_LATENCY = Histogram("agentic_graph_latency_seconds", "Graph execution latency in seconds.")

LLM_CALLS = Counter("agentic_llm_calls_total", "Total LLM calls.", ["operation"])
LLM_FAILURES = Counter("agentic_llm_failures_total", "Total failed LLM calls.", ["operation"])
LLM_RETRIES = Counter("agentic_llm_retries_total", "Retried LLM attempts after transient failures.", ["operation"])
LLM_MODEL_FALLBACKS = Counter("agentic_llm_model_fallbacks_total", "Switches to a fallback chat model.", ["operation"])

JOB_SUBMISSIONS = Counter("agentic_job_submissions_total", "Total accepted job submissions.")
JOB_FAILURES = Counter("agentic_job_failures_total", "Total background job failures.")
JOB_QUEUE_DEPTH = Gauge("agentic_job_queue_depth", "Approximate in-process queued jobs.")

CACHE_HITS = Counter("agentic_cache_hits_total", "Cache hits.")
CACHE_MISSES = Counter("agentic_cache_misses_total", "Cache misses.")
RATE_LIMIT_REJECTIONS = Counter("agentic_rate_limit_rejections_total", "Requests rejected by rate limiting.")
AUTH_FAILURES = Counter("agentic_auth_failures_total", "Failed authentication attempts.")
