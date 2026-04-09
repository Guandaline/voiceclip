import time
from functools import lru_cache
from fastapi import APIRouter, Depends
from voiceflip.api.schemas import (
    GuardrailRequest,
    GuardrailResponse,
    CitationDecisionSchema,
    MetricsSchema,
)
from voiceflip.domain.guardrail_engine import GuardrailEngine
from voiceflip.domain.models import CitationStatus
from voiceflip.matching import (
    HybridMatchingStrategy,
    SemanticMatchingStrategy,
    KeywordMatchingStrategy,
)
from voiceflip.infra.embeddings import EmbeddingProviderFactory, EmbeddingSettings

router = APIRouter(prefix="/guardrail", tags=["Guardrails"])


@lru_cache()
def get_engine() -> GuardrailEngine:
    """
    Dependency provider for the Guardrail Engine.
    Ensures single initialization of the embedding models via caching
    and aligns matching thresholds with the evaluation parameters.
    """
    settings = EmbeddingSettings()
    provider = EmbeddingProviderFactory.create(settings)
    semantic = SemanticMatchingStrategy(provider)
    keyword = KeywordMatchingStrategy()
    strategy = HybridMatchingStrategy(semantic, keyword)

    return GuardrailEngine(
        strategy=strategy, score_threshold=0.2, ambiguity_margin=0.01
    )


@router.post("", response_model=GuardrailResponse)
async def process_guardrail(
    request: GuardrailRequest, engine: GuardrailEngine = Depends(get_engine)
):
    """
    Evaluates an LLM answer and injects citations based on KB grounding rules.
    Ensures that the endpoint never returns 500, preserving the original answer.
    """
    start_time = time.perf_counter()
    try:
        from voiceflip.api.state import HEALTH_STATS

        final_answer, decision, metrics = engine.process(request)

        if decision.status.value in HEALTH_STATS:
            HEALTH_STATS[decision.status.value] += 1

        return GuardrailResponse(
            final_answer=final_answer,
            citation_decision=CitationDecisionSchema(
                status=decision.status,
                matched_label=decision.matched_label,
                strategy_used=decision.strategy_used,
                similarity_score=decision.similarity_score,
                reason=decision.reason,
            ),
            metrics=MetricsSchema(
                latency_ms=metrics.latency_ms, llm_calls=metrics.llm_calls
            ),
        )
    except Exception as e:
        latency: int = int((time.perf_counter() - start_time) * 1000)
        return GuardrailResponse(
            final_answer=request.llm_answer,
            citation_decision=CitationDecisionSchema(
                status=CitationStatus.SKIPPED_NO_MATCH,
                matched_label=None,
                strategy_used="system_error_fallback",
                similarity_score=0.0,
                reason=f"Internal System Error: {str(e)}",
            ),
            metrics=MetricsSchema(latency_ms=latency, llm_calls=0),
        )
