from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional
from voiceflip.domain.models import CitationStatus


class GroundingInfo(BaseModel):
    """
    Flags indicating the source and validity of the LLM response.
    """

    is_grounded: bool = Field(
        ..., description="Whether the answer is based on any context"
    )
    kb_grounded: bool = Field(
        ...,
        description="Specifically whether the answer is grounded in the Knowledge Base",
    )


class CandidateLink(BaseModel):
    """
    A potential citation source provided by the retrieval layer.
    """

    label: str = Field(..., description="User-facing text for the link")
    url: HttpUrl = Field(..., description="The canonical URL for the citation")
    keywords: List[str] = Field(
        default_factory=list, description="Keywords for lexical matching"
    )
    description: str = Field(..., description="Text description for semantic matching")


class GuardrailRequest(BaseModel):
    """
    Input schema for the citation guardrail microservice.
    """

    query: str = Field(..., description="The original user input")
    llm_answer: str = Field(..., description="The raw response from the synthesis LLM")
    grounding: GroundingInfo = Field(..., description="Grounding metadata")
    is_chitchat: bool = Field(
        ..., description="Flag for conversational/non-informative queries"
    )
    candidate_links: List[CandidateLink] = Field(
        default_factory=list, description="Available citation candidates"
    )


class CitationDecisionSchema(BaseModel):
    """
    Final output decision with observability data.
    """

    status: CitationStatus = Field(..., description="Outcome of the R1-R5 rules")
    matched_label: Optional[str] = Field(
        None, description="The label of the citation if injected/already_present"
    )
    strategy_used: Optional[str] = Field(
        None, description="Strategy that made the match"
    )
    similarity_score: float = Field(
        0.0, ge=0.0, le=1.0, description="Match confidence score"
    )
    reason: str = Field(..., description="Human-readable reason for the decision")


class MetricsSchema(BaseModel):
    """
    Technical metrics for monitoring.
    """

    latency_ms: int = Field(0, description="Execution time in milliseconds")
    llm_calls: int = Field(0, description="Number of external embedding/LLM calls made")


class GuardrailResponse(BaseModel):
    """
    The final response containing the (potentially) modified answer and metadata.
    """

    final_answer: str = Field(
        ..., description="The answer including citations if applicable"
    )
    citation_decision: CitationDecisionSchema = Field(
        ..., description="The logic breakdown"
    )
    metrics: MetricsSchema = Field(..., description="System performance data")
