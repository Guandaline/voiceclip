from typing import List, Tuple
from voiceflip.api.schemas import CandidateLink
from voiceflip.domain.models import MatchResult
from ..base import BaseMatchingStrategy
from .keyword import KeywordMatchingStrategy
from .semantic import SemanticMatchingStrategy


class HybridMatchingStrategy(BaseMatchingStrategy):
    """
    Advanced Hybrid Strategy that re-ranks candidates using weighted scoring.
    """

    def __init__(
        self,
        semantic_strat: SemanticMatchingStrategy,
        keyword_strat: KeywordMatchingStrategy,
    ):
        """
        Initializes the hybrid approach.
        """
        self.semantic = semantic_strat
        self.keyword = keyword_strat

    def find_best_match(
        self, query: str, candidates: List[CandidateLink], margin: float = 0.0
    ) -> MatchResult:
        """
        Executes a weighted re-ranking by combining semantic and lexical scores.
        """
        if not candidates:
            return MatchResult(label=None, url=None, score=0.0, strategy_used="hybrid")

        try:
            semantic_result = self.semantic.find_best_match(
                query, candidates, margin=0.0
            )

            if semantic_result.reason and "failure" in semantic_result.reason.lower():
                keyword_fallback = self.keyword.find_best_match(
                    query, candidates, margin=margin
                )
                return MatchResult(
                    label=keyword_fallback.label,
                    url=keyword_fallback.url,
                    score=keyword_fallback.score,
                    strategy_used="hybrid_fallback_keyword",
                    reason=semantic_result.reason,
                )

            scored_candidates: List[Tuple[float, CandidateLink]] = []
            for candidate in candidates:
                ind_sem = self.semantic.find_best_match(query, [candidate], margin=0.0)
                ind_key = self.keyword.find_best_match(query, [candidate], margin=0.0)

                hybrid_score = (ind_sem.score * 0.7) + (ind_key.score * 0.3)
                scored_candidates.append((hybrid_score, candidate))

            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            best_score, best_candidate = scored_candidates[0]

            if len(scored_candidates) > 1:
                runner_up_score = scored_candidates[1][0]
                if abs(best_score - runner_up_score) <= margin:
                    return MatchResult(
                        label=None,
                        url=None,
                        score=0.0,
                        strategy_used="hybrid",
                        reason="Ambiguous match: top candidates within margin.",
                    )

            return MatchResult(
                label=best_candidate.label if best_candidate else None,
                url=str(best_candidate.url) if best_candidate else None,
                score=best_score,
                strategy_used="hybrid",
                reason="Weighted re-ranking applied successfully.",
            )

        except Exception as e:
            keyword_res = self.keyword.find_best_match(query, candidates, margin=margin)
            return MatchResult(
                label=keyword_res.label,
                url=keyword_res.url,
                score=keyword_res.score,
                strategy_used="hybrid_critical_fallback",
                reason=f"Unexpected error in hybrid matching: {str(e)}",
            )
