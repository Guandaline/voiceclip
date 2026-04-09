import math
from typing import List, Tuple
from voiceflip.infra.embeddings import EmbeddingProvider
from voiceflip.api.schemas import CandidateLink
from voiceflip.domain.models import MatchResult
from ..base import BaseMatchingStrategy


class SemanticMatchingStrategy(BaseMatchingStrategy):
    """
    Semantic strategy using vector embeddings and Cosine Similarity.
    """

    def __init__(self, provider: EmbeddingProvider):
        """
        Injects the embedding provider.
        """
        self._provider = provider

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """
        Calculates the cosine similarity between two vectors.
        """
        dot_product = sum(a * b for a, b in zip(v1, v2))
        magnitude1 = math.sqrt(sum(a * a for a in v1))
        magnitude2 = math.sqrt(sum(b * b for b in v2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        return dot_product / (magnitude1 * magnitude2)

    def find_best_match(
        self, query: str, candidates: List[CandidateLink], margin: float = 0.0
    ) -> MatchResult:
        """
        Finds the candidate with the highest semantic similarity to the query.
        """
        if not candidates:
            return MatchResult(
                label=None, url=None, score=0.0, strategy_used="semantic"
            )

        try:
            query_vector = self._provider.get_embedding(query)
            descriptions = [c.description for c in candidates]
            candidate_vectors = self._provider.get_batch_embeddings(descriptions)

            scored_candidates: List[Tuple[float, CandidateLink]] = []
            for i, c_vector in enumerate(candidate_vectors):
                score = self._cosine_similarity(query_vector, c_vector)
                scored_candidates.append((score, candidates[i]))

            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            best_score, best_candidate = scored_candidates[0]

            if len(scored_candidates) > 1:
                runner_up_score = scored_candidates[1][0]
                if abs(best_score - runner_up_score) <= margin:
                    return MatchResult(
                        label=None,
                        url=None,
                        score=0.0,
                        strategy_used="semantic",
                        reason="Ambiguous match: top candidates within margin.",
                    )

            return MatchResult(
                label=best_candidate.label,
                url=str(best_candidate.url),
                score=best_score,
                strategy_used="semantic",
            )

        except Exception as e:
            return MatchResult(
                label=None,
                url=None,
                score=0.0,
                strategy_used="semantic",
                reason=f"Embedding failure: {str(e)}",
            )
