import time
from typing import List, Tuple, Optional, Dict, Any

from voiceflip.api.schemas import GuardrailRequest
from voiceflip.domain.models import (
    CitationDecision,
    CitationStatus,
    EngineMetrics,
    MatchResult,
)
from voiceflip.matching.base import BaseMatchingStrategy
from voiceflip.matching import KeywordMatchingStrategy
from voiceflip.domain.policies.base import GuardrailPolicy
from voiceflip.domain.policies.rules import (
    ChitchatPolicy,
    GroundingPolicy,
    ThresholdPolicy,
    DuplicationPolicy,
)


class GuardrailEngine:
    """
    Policy Orchestrator that executes validation rules and matching strategies.
    Enforces graceful degradation by falling back to lexical matching upon strategy failure.
    """

    def __init__(
        self,
        strategy: BaseMatchingStrategy,
        score_threshold: float = 0.3,
        ambiguity_margin: float = 0.05,
    ):
        """
        Initializes the engine with its matching strategy and validation policies.
        """
        self.strategy: BaseMatchingStrategy = strategy
        self.fallback_strategy: BaseMatchingStrategy = KeywordMatchingStrategy()
        self.context: Dict[str, Any] = {
            "threshold": score_threshold,
            "margin": ambiguity_margin,
        }

        self.pre_match_policies: List[GuardrailPolicy] = [
            ChitchatPolicy(),
            GroundingPolicy(),
        ]
        self.post_match_policies: List[GuardrailPolicy] = [
            ThresholdPolicy(),
            DuplicationPolicy(),
        ]

    def _get_metrics(self, start_time: float, llm_calls: int = 0) -> EngineMetrics:
        """
        Calculates telemetry for the execution.
        """
        latency: int = int((time.perf_counter() - start_time) * 1000)
        return EngineMetrics(latency_ms=latency, llm_calls=llm_calls)

    def process(
        self, request: GuardrailRequest
    ) -> Tuple[str, CitationDecision, EngineMetrics]:
        """
        Orchestrates policies and matching strategy execution with global fallback.
        """
        start_time: float = time.perf_counter()

        for policy in self.pre_match_policies:
            result: Optional[CitationDecision] = policy.validate(
                request, engine_context=self.context
            )
            if result:
                return request.llm_answer, result, self._get_metrics(start_time)

        if not request.candidate_links:
            return (
                request.llm_answer,
                CitationDecision(
                    status=CitationStatus.SKIPPED_NO_MATCH,
                    reason="No candidate links provided",
                ),
                self._get_metrics(start_time),
            )

        match_res = self.strategy.find_best_match(
            request.query,
            request.candidate_links,
            margin=self.context.get("margin", 0.0),
        )

        if match_res.reason and (
            "failure" in match_res.reason.lower() or "error" in match_res.reason.lower()
        ):
            fallback_res = self.fallback_strategy.find_best_match(
                request.query,
                request.candidate_links,
                margin=self.context.get("margin", 0.0),
            )
            match_res = MatchResult(
                label=fallback_res.label,
                url=fallback_res.url,
                score=fallback_res.score,
                strategy_used="engine_fallback_keyword",
                reason="Primary strategy failed. Fallback applied.",
            )

        llm_calls: int = (
            1
            if any(s in match_res.strategy_used for s in ["semantic", "hybrid"])
            else 0
        )

        for policy in self.post_match_policies:
            result = policy.validate(request, match_res, self.context)
            if result:
                enriched: CitationDecision = CitationDecision(
                    status=result.status,
                    matched_label=result.matched_label,
                    strategy_used=match_res.strategy_used,
                    similarity_score=result.similarity_score,
                    reason=result.reason,
                )
                return (
                    request.llm_answer,
                    enriched,
                    self._get_metrics(start_time, llm_calls),
                )

        final_answer: str = (
            f"{request.llm_answer}\n\n"
            f"For more information, see [{match_res.label}]({match_res.url})."
        )

        decision: CitationDecision = CitationDecision(
            status=CitationStatus.INJECTED,
            matched_label=match_res.label,
            similarity_score=match_res.score,
            strategy_used=match_res.strategy_used,
            reason=f"Rule R4: Injected via {match_res.strategy_used}",
        )

        return final_answer, decision, self._get_metrics(start_time, llm_calls)
