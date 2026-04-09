# Video Presentation Script: Citation Guardrail Engine

## 1. Introduction (0:00 - 0:30)
* **[Camera ON]**
* "Hi team, I'm Valter. Thank you for the opportunity to present my solution for the Citation Guardrail Engine."
* "My primary goal with this implementation was to build a highly deterministic, observable, and resilient microservice. I deliberately chose a Pragmatic Clean Architecture to strictly separate the HTTP layer, the business rules (R1 to R5), and the matching strategies."

## 2. Architecture & Code Walkthrough (0:30 - 2:00)
* **[Screen Share: IDE showing project structure & guardrail_engine.py]**
* "As you can see in the `domain/guardrail_engine.py`, the core logic does not rely on complex frameworks. It evaluates the boolean flags sequentially."
* "For the matching layer, I implemented a Strategy Pattern. The default is a Hybrid strategy."
* **[Highlight infra/embeddings.py or .env file]**
* "As requested, the system supports swapping between local HuggingFace models and OpenAI APIs. This is cleanly handled via the `EmbeddingProviderFactory` and environment variables, ensuring the core engine remains completely unaware of the underlying infrastructure."

```python
# Example of the clean factory instantiation
provider = EmbeddingProviderFactory.create(settings)
```

* **[Highlight ADR-003 / Fallback Logic]**
* "One structural decision I want to highlight is the Graceful Degradation. If the embedding provider (HuggingFace or OpenAI) times out or fails, the system catches the exception and immediately falls back to the deterministic Keyword Strategy. It returns a 200 OK with a specific metric flag rather than dropping a 500 Internal Server Error on the user."

## 3. Live Evaluation Run (2:00 - 3:00)
* **[Screen Share: Terminal]**
* "Let's run the evaluation against the extended golden set."
* **[Action: Run the evaluation script]**

```bash
python scripts/eval.py
```

* "As you can see, we have a 100% pass rate across the 10+ cases. I added edge cases specifically for identical scores handling, empty candidate lists, and simulated embedding timeouts to prove the fallback works."

## 4. Answering the Core Questions (3:00 - End)
* **[Camera ON (Stop Screen Share)]**
* "Moving to the required questions:"

### Q1: Why didn't you just ask the synthesis LLM to include the citation directly in its response?
* "Relying on the synthesis LLM for citation injection introduces unacceptable non-determinism. LLMs are probabilistic text generators; they can hallucinate URLs, duplicate citations despite being prompted not to, or subtly alter the link structure. Furthermore, using a large context-window LLM just to perform conditional string concatenation is highly inefficient in terms of cost and latency. By using a post-processing guardrail, we guarantee 100% adherence to business rules, zero hallucinated URLs, and sub-100ms execution times."

### Q2: What is the most dangerous failure mode of your system and how would you detect it in production?
* "The most dangerous failure mode is 'Silent Degradation'. If our embedding API fails, the system correctly falls back to lexical keyword matching. However, if this outage persists silently, we might start injecting incorrect citations based purely on keyword overlap (false positives). **This is extremely dangerous because citing completely irrelevant pages directly destroys user trust in the RAG system's accuracy.**"
* "To detect this in production, I exposed the `reason` field in the response. I would set up a Datadog or Prometheus monitor on the `embedding_failed_fallback_keyword` metric. If this fallback accounts for more than 1% of our traffic over a 5-minute rolling window, it should trigger a P2 alert to the on-call engineer."

### Q3: If you had one more week, what would you add first and why?
* **[Speak clearly and deliberately—this part is dense]**
* "If I had one more week, I would implement two architectural patterns to evolve this guardrail from a simple matching script into an enterprise-grade AI component:"
* **[Pause]**
* "First, for matching precision, I would introduce a **Cross-Encoder Re-ranker** for ambiguity resolution. Currently, the semantic strategy relies on a Bi-encoder with Cosine Similarity. While fast, it lacks deep relational understanding. By implementing a `ReRankerProtocol` that offloads a Cross-Encoder's predict method to a ThreadPoolExecutor, we could cross-score the `llm_answer` directly against the candidate link descriptions. This would practically eliminate false positives on boundary matches without blocking the async event loop."
* **[Pause]**
* "Second, for systemic resilience, I would add an **Adaptive Concurrency and Backpressure Manager**. Our current fallback handles sudden embedding API failures, but in a production environment, we need to prevent them entirely. By implementing an AIMD (Additive Increase, Multiplicative Decrease) concurrency controller, the engine could monitor hardware telemetry and API latency in real-time. If the remote embedding provider starts degrading, the system would proactively apply backpressure and route traffic to the deterministic lexical strategy before a cascading system failure occurs."
* "Thank you for your time reviewing my code. I look forward to discussing it further."