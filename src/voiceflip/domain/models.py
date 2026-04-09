from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum


class CitationStatus(str, Enum):
    """
    Defines the final state of the citation logic based on R1-R5 rules.
    """

    INJECTED = "injected"
    SKIPPED_CHITCHAT = "skipped_chitchat"
    SKIPPED_UNGROUNDED = "skipped_ungrounded"
    SKIPPED_NO_MATCH = "skipped_no_match"
    ALREADY_PRESENT = "already_present"


@dataclass(frozen=True)
class MatchResult:
    """
    Internal result from a matching strategy before rule enforcement.
    """

    label: Optional[str] = field(
        default=None, metadata={"description": "The label of the matched link"}
    )
    url: Optional[str] = field(
        default=None, metadata={"description": "The target URL of the match"}
    )
    score: float = field(
        default=0.0, metadata={"description": "Similarity score between 0 and 1"}
    )
    strategy_used: str = field(
        default="keyword",
        metadata={"description": "The identification of the algorithm used"},
    )
    reason: Optional[str] = field(
        default=None,
        metadata={"description": "Detailed notes on strategy execution or failure"},
    )


@dataclass(frozen=True)
class CitationDecision:
    """
    The final business decision regarding citation injection.
    """

    status: CitationStatus = field(default=CitationStatus.SKIPPED_NO_MATCH)
    matched_label: Optional[str] = field(default=None)
    strategy_used: Optional[str] = field(default=None)
    similarity_score: float = field(default=0.0)
    reason: str = field(
        default="", metadata={"description": "Explanation for the final status choice"}
    )


@dataclass(frozen=True)
class EngineMetrics:
    """
    Performance telemetry for the guardrail execution.
    """

    latency_ms: int = field(default=0)
    llm_calls: int = field(default=0)
