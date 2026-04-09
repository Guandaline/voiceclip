# Citation Guardrail Engine — Achitetural Planning

## Objective

Design and implement a **Citation Guardrail Engine** that post-processes LLM responses and enforces deterministic citation rules.

The system must:

- Inject citations only when appropriate
- Avoid duplication
- Never hallucinate links
- Be observable and measurable
- Gracefully handle failures

---

## Scope

### In Scope

- Guardrail decision engine (R1–R5)
- Matching strategies (keyword, semantic, hybrid)
- Embedding integration (HF + OpenAI switch)
- API endpoints (/guardrail, /health)
- Evaluation pipeline (golden set + accuracy)

### Out of Scope

- Frontend
- Persistence layer
- Authentication
- Deployment
- Model training

---

## Project Structure

```text
app/
  main.py

  api/
    guardrail_router.py
    health_router.py
    schemas.py

  domain/
    guardrail_engine.py
    models.py

  matching/
    base.py
    keyword.py
    semantic.py
    hybrid.py

  infrastructure/
    embeddings.py

scripts/
  eval.py

tests/
  golden_set.json

docs/
  adr/

README.md
```

---

## Core Responsibilities

### API Layer

**Files:**
- `guardrail_router.py`
- `health_router.py`
- `schemas.py`

**Responsibilities:**
- Validate request/response
- Handle HTTP routing
- Delegate to domain layer

---

### Domain Layer

**File:**
- `guardrail_engine.py`

**Responsibilities:**
- Enforce rules R1–R5
- Orchestrate matching
- Build final response
- Track metrics

**Properties:**
- Pure business logic
- No external dependencies

---

### Matching Layer

**Files:**
- `base.py`
- `keyword.py`
- `semantic.py`
- `hybrid.py`

**Responsibilities:**
- Select best candidate link
- Provide similarity scores

---

### Infrastructure Layer

**File:**
- `embeddings.py`

**Responsibilities:**
- Provide embeddings (HF/OpenAI)
- Abstract external providers

---

## Core Execution Flow (R1–R5)

```text
1. if is_chitchat → skipped_chitchat
2. if not kb_grounded → skipped_ungrounded
3. if candidate_links is empty → skipped_no_match
4. run matching strategy
5. if best_score < threshold → skipped_no_match
6. if URL already present → already_present
7. else → injected
```

---

## Matching Strategy

### Default: Hybrid

```text
score = 0.7 * semantic + 0.3 * keyword
```

### Strategies

- Keyword → lexical overlap
- Semantic → embeddings + cosine similarity
- Hybrid → combination of both

---

## Ambiguity Handling

```python
if abs(top1_score - top2_score) < margin:
    return skipped_no_match
```

**Goal:**
- Avoid incorrect citation injection
- Prefer precision over recall

---

## Failure Handling

If embedding fails:

```text
- fallback to keyword strategy
- record reason: embedding_failed_fallback_keyword
```

**Guarantee:**
- No request failure (no 500)

---

## Observability

### Metrics

```json
{
  "latency_ms": 120,
  "llm_calls": 1
}
```

---

### Health Counters

```json
{
  "injected": X,
  "skipped_chitchat": X,
  "skipped_ungrounded": X,
  "skipped_no_match": X,
  "already_present": X
}
```

---

## Evaluation Strategy

### Golden Set

Located at:

```text
tests/golden_set.json
```

Includes:

- Required seed cases
- Additional edge cases:
  - ambiguous matches
  - empty candidates
  - near-threshold scores
  - duplicate URLs
  - embedding failures

---

### Eval Script

```bash
python scripts/eval.py
```

Output:

```text
case_id | expected | actual | PASS/FAIL
accuracy: 0.90
```

---

## Implementation Plan

### Step 1 — Core Models
- CandidateLink
- CitationDecision
- MatchResult
- Metrics

---

### Step 2 — Matching Strategies
- Implement keyword
- Implement semantic
- Add fallback logic
- Implement hybrid

---

### Step 3 — Guardrail Engine
- Implement R1–R5 flow
- Add ambiguity handling
- Add citation injection logic

---

### Step 4 — API Layer
- Implement `/guardrail`
- Implement `/health`

---

### Step 5 — Evaluation
- Create golden set (10+ cases)
- Implement eval script
- Validate accuracy

---

## Key Design Decisions

- Rules are deterministic, not LLM-driven
- Matching is pluggable but simple
- Observability is first-class
- System prioritizes correctness over recall

---

## Success Criteria

- All R1–R5 rules correctly enforced
- 100% pass on seed cases
- High accuracy on extended golden set
- No runtime failures
- Clear, explainable decisions


