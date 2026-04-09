from abc import ABC, abstractmethod
from typing import List
from voiceflip.api.schemas import CandidateLink
from voiceflip.domain.models import MatchResult


class BaseMatchingStrategy(ABC):
    """
    Abstract interface for link matching algorithms.
    """

    @abstractmethod
    def find_best_match(
        self, query: str, candidates: List[CandidateLink], margin: float = 0.0
    ) -> MatchResult:
        """
        Analyzes candidates and returns the most relevant match based on the query.
        Applies ambiguity logic if the top candidates score within the margin.
        """
        pass
