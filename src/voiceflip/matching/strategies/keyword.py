import re
from typing import List, Set, Tuple
from voiceflip.api.schemas import CandidateLink
from voiceflip.domain.models import MatchResult
from ..base import BaseMatchingStrategy


class KeywordMatchingStrategy(BaseMatchingStrategy):
    """
    Lexical strategy based on keyword overlap between query and candidates.
    Fast, deterministic, and requires no external APIs.
    """

    def _tokenize(self, text: str) -> Set[str]:
        """
        Simple tokenizer that normalizes text and extracts words.
        """
        return set(re.findall(r"\w+", text.lower()))

    def find_best_match(
        self, query: str, candidates: List[CandidateLink], margin: float = 0.0
    ) -> MatchResult:
        """
        Finds the candidate with the highest keyword intersection.
        """
        if not candidates:
            return MatchResult(label=None, url=None, score=0.0, strategy_used="keyword")

        query_tokens = self._tokenize(query)
        scored_candidates: List[Tuple[float, CandidateLink]] = []

        for candidate in candidates:
            candidate_text = f"{candidate.label} {' '.join(candidate.keywords)}"
            candidate_tokens = self._tokenize(candidate_text)

            if not query_tokens:
                score = 0.0
            else:
                intersection = query_tokens.intersection(candidate_tokens)
                score = len(intersection) / len(query_tokens.union(candidate_tokens))

            scored_candidates.append((score, candidate))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_candidate = scored_candidates[0]

        if len(scored_candidates) > 1:
            runner_up_score = scored_candidates[1][0]
            if abs(best_score - runner_up_score) <= margin:
                return MatchResult(
                    label=None,
                    url=None,
                    score=0.0,
                    strategy_used="keyword",
                    reason="Ambiguous match: top candidates within margin.",
                )

        return MatchResult(
            label=best_candidate.label if best_candidate else None,
            url=str(best_candidate.url) if best_candidate else None,
            score=best_score,
            strategy_used="keyword",
        )
