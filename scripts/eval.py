import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from voiceflip.domain.guardrail_engine import GuardrailEngine
from voiceflip.matching.strategies.keyword import KeywordMatchingStrategy
from voiceflip.api.schemas import GuardrailRequest


def run_evaluation():
    """
    Evaluates the engine using KeywordStrategy for deterministic local testing.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    golden_set_path = os.path.join(base_dir, "benchmarks/golden_set.json")

    with open(golden_set_path, "r") as f:
        cases = json.load(f)

    strategy = KeywordMatchingStrategy()
    engine = GuardrailEngine(
        strategy=strategy, score_threshold=0.2, ambiguity_margin=0.01
    )

    passed = 0
    total = len(cases)

    print(f"\n{'ID':<32} | {'EXPECTED':<18} | {'ACTUAL':<18} | {'RESULT'}")
    print("-" * 90)

    for case in cases:
        request = GuardrailRequest(**case["input"])
        _, decision, _ = engine.process(request)

        expected_status = case["expected"]["status"]
        actual_status = decision.status.value

        is_pass = expected_status == actual_status
        if is_pass and case["expected"].get("matched_label"):
            is_pass = case["expected"]["matched_label"] == decision.matched_label

        result_text = "PASS" if is_pass else f"FAIL ({decision.reason})"
        if is_pass:
            passed += 1

        print(
            f"{case['id']:<32} | {expected_status:<18} | {actual_status:<18} | {result_text}"
        )

    accuracy = (passed / total) * 100
    print("-" * 90)
    print(f"Final Accuracy: {accuracy:.2f}% ({passed}/{total})")
    print("-" * 90 + "\n")


if __name__ == "__main__":
    run_evaluation()
