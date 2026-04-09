"""
Global state management for the API layer.
"""

from voiceflip.domain.models import CitationStatus

# Thread-safe counters for health monitoring
HEALTH_STATS = {status.value: 0 for status in CitationStatus}
