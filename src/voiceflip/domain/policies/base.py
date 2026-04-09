from abc import ABC, abstractmethod
from typing import Optional, Tuple
from voiceflip.api.schemas import GuardrailRequest
from voiceflip.domain.models import CitationDecision, MatchResult


class GuardrailPolicy(ABC):
    """
    Abstract base class for all guardrail validation policies.
    """

    @abstractmethod
    def validate(
        self,
        request: GuardrailRequest,
        match_res: Optional[MatchResult] = None,
        engine_context: Optional[dict] = None,
    ) -> Optional[CitationDecision]:
        """
        Evaluates the request against a specific business rule.

        Returns a CitationDecision if the rule triggers a skip/action,
        otherwise returns None to continue the chain.
        """
        pass
