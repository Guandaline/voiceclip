import re
from typing import Optional
from voiceflip.api.schemas import GuardrailRequest
from voiceflip.domain.models import CitationDecision, CitationStatus, MatchResult
from .base import GuardrailPolicy


class ChitchatPolicy(GuardrailPolicy):
    """Rule R1: Prevents citations in conversational queries."""

    def validate(
        self,
        request: GuardrailRequest,
        match_res: Optional[MatchResult] = None,
        engine_context: Optional[dict] = None,
    ):
        if request.is_chitchat:
            return CitationDecision(
                status=CitationStatus.SKIPPED_CHITCHAT,
                reason="Rule R1: identified as chitchat",
            )
        return None


class GroundingPolicy(GuardrailPolicy):
    """Rule R2: Ensures KB grounding before citing."""

    def validate(
        self,
        request: GuardrailRequest,
        match_res: Optional[MatchResult] = None,
        engine_context: Optional[dict] = None,
    ):
        if not request.grounding.kb_grounded:
            return CitationDecision(
                status=CitationStatus.SKIPPED_UNGROUNDED,
                reason="Rule R2: not grounded in KB",
            )
        return None


class ThresholdPolicy(GuardrailPolicy):
    """Rule R5: Validates match confidence against a threshold."""

    def validate(
        self,
        request: GuardrailRequest,
        match_res: Optional[MatchResult] = None,
        engine_context: Optional[dict] = None,
    ):
        threshold = engine_context.get("threshold", 0.3)
        if not match_res or match_res.score < threshold:
            score = match_res.score if match_res else 0.0
            return CitationDecision(
                status=CitationStatus.SKIPPED_NO_MATCH,
                similarity_score=score,
                reason=f"Rule R5: score {score:.2f} below threshold {threshold}",
            )
        return None


class DuplicationPolicy(GuardrailPolicy):
    """Rule R3: Prevents redundant citations."""

    def validate(
        self,
        request: GuardrailRequest,
        match_res: Optional[MatchResult] = None,
        engine_context: Optional[dict] = None,
    ):
        if not match_res or not match_res.url:
            return None

        escaped_url = re.escape(match_res.url)
        if re.search(escaped_url, request.llm_answer):
            return CitationDecision(
                status=CitationStatus.ALREADY_PRESENT,
                matched_label=match_res.label,
                similarity_score=match_res.score,
                reason="Rule R3: URL already present",
            )
        return None
