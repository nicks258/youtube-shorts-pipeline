"""Unit tests for the pluggable reviewer (Ant Ling + Gemini)."""

import base64
import io
import json
from pathlib import Path
from PIL import Image

import pytest
from verticals.reviewer import (
    _parse_json_verdict,
    resize_frame_to_720_base64,
    reset_review_call_count,
    get_review_call_count,
    _execute_review,
    MAX_CALLS_PER_VIDEO,
)


def test_parse_json_verdict():
    # Valid raw JSON
    raw = '{"pass": true, "problems": [], "worst_sentence": "", "fix_suggestion": ""}'
    res = _parse_json_verdict(raw)
    assert res["pass"] is True
    assert res["problems"] == []

    # Markdown fence JSON
    raw_fence = '```json\n{"pass": false, "problems": ["Missing subject"], "worst_sentence": "Bad", "fix_suggestion": "Fix"}\n```'
    res_fence = _parse_json_verdict(raw_fence)
    assert res_fence["pass"] is False
    assert res_fence["problems"] == ["Missing subject"]
    assert res_fence["worst_sentence"] == "Bad"


def test_resize_frame_to_720_base64(tmp_path):
    # Create test image 1080x1920
    test_img = tmp_path / "test_frame.jpg"
    img = Image.new("RGB", (1080, 1920), color="blue")
    img.save(test_img, "JPEG")

    b64 = resize_frame_to_720_base64(test_img)
    assert isinstance(b64, str)
    raw_bytes = base64.b64decode(b64)
    with Image.open(io.BytesIO(raw_bytes)) as out_img:
        w, h = out_img.size
        assert max(w, h) <= 720
        # Aspect ratio preserved (1080 / 1920 = 0.5625)
        assert abs((w / h) - (1080 / 1920)) < 0.02


def test_spend_guard_hard_cap(monkeypatch):
    reset_review_call_count()
    assert get_review_call_count() == 0

    # Mock provider calls
    monkeypatch.setattr("verticals.reviewer._call_antling_chat", lambda *args, **kwargs: ('{"pass": true, "problems": []}', {"prompt_tokens": 10, "completion_tokens": 10}))
    monkeypatch.setattr("verticals.reviewer._get_antling_key", lambda: "fake-key")

    for i in range(MAX_CALLS_PER_VIDEO):
        v = _execute_review("script", "test prompt")
        assert not v.get("skipped", False)

    assert get_review_call_count() == MAX_CALLS_PER_VIDEO

    # 5th call should be capped and skipped
    capped = _execute_review("script", "test prompt")
    assert capped.get("skipped") is True
    assert capped.get("pass") is True


def test_review_script_decisions_pass(monkeypatch):
    from verticals.reviewer import review_script, reset_review_call_count
    from google.genai._gaos.types.decisions import NoulAnswer, ScoreAnswer
    from types import SimpleNamespace

    reset_review_call_count()

    mock_resp = SimpleNamespace(
        answers={
            "subject_named_in_first_two": NoulAnswer(noul=0.95, confidence=0.95, type="noul"),
            "every_proper_noun_and_jargon_explained": NoulAnswer(noul=0.90, confidence=0.90, type="noul"),
            "hook_names_subject": NoulAnswer(noul=0.92, confidence=0.92, type="noul"),
            "closing_question_specific": NoulAnswer(noul=0.88, confidence=0.88, type="noul"),
            "numbers_in_sources": NoulAnswer(noul=0.99, confidence=0.99, type="noul"),
            "clarity_score": ScoreAnswer(
                score=2.8,
                confidence=0.85,
                probabilities={"0": 0.01, "1": 0.02, "2": 0.12, "3": 0.85},
                legend={"0": "lost", "1": "confused", "2": "clear", "3": "fully clear"},
                type="score",
            ),
        },
        usage=SimpleNamespace(total_input_tokens=450, total_output_tokens=0),
    )

    mock_client = SimpleNamespace(
        decisions=SimpleNamespace(create=lambda **kwargs: mock_resp)
    )
    monkeypatch.setattr("verticals.reviewer._get_genai_client", lambda: mock_client)

    verdict = review_script("Good script. All rules followed. Specific ending?", "Topic A")
    assert verdict["pass"] is True
    assert verdict["soft_fail"] is False
    assert verdict["clarity_score"] >= 2.5
    assert len(verdict["problems"]) == 0
    assert verdict["provider"] == "google-decisions"


def test_review_script_decisions_fail_and_soft_fail(monkeypatch):
    from verticals.reviewer import review_script, reset_review_call_count
    from google.genai._gaos.types.decisions import NoulAnswer, ScoreAnswer
    from types import SimpleNamespace

    reset_review_call_count()

    # Low noul (< 0.5) with low confidence (< 0.6 -> soft fail)
    # Plus clarity score < 2.5 and combined level 0+1 probability > 0.35
    mock_resp = SimpleNamespace(
        answers={
            "subject_named_in_first_two": NoulAnswer(noul=0.10, confidence=0.55, type="noul"),
            "every_proper_noun_and_jargon_explained": NoulAnswer(noul=0.80, confidence=0.80, type="noul"),
            "hook_names_subject": NoulAnswer(noul=0.05, confidence=0.95, type="noul"),
            "closing_question_specific": NoulAnswer(noul=0.02, confidence=0.98, type="noul"),
            "numbers_in_sources": NoulAnswer(noul=0.90, confidence=0.90, type="noul"),
            "clarity_score": ScoreAnswer(
                score=1.4,
                confidence=0.45,
                probabilities={"0": 0.45, "1": 0.35, "2": 0.15, "3": 0.05},
                legend={},
                type="score",
            ),
        },
        usage=SimpleNamespace(total_input_tokens=500, total_output_tokens=0),
    )

    mock_client = SimpleNamespace(
        decisions=SimpleNamespace(create=lambda **kwargs: mock_resp)
    )
    monkeypatch.setattr("verticals.reviewer._get_genai_client", lambda: mock_client)

    verdict = review_script("Vague hook. What do you think? Comment below.", "Topic B")
    assert verdict["pass"] is False
    assert verdict["soft_fail"] is True  # soft fail because noul check had confidence < 0.6
    assert verdict["clarity_score"] < 2.5
    assert len(verdict["problems"]) >= 3


def test_review_script_clarity_probability_fail(monkeypatch):
    """Test that clarity fails if combined level 0+1 probability > 0.35 even if score seems borderline."""
    from verticals.reviewer import review_script, reset_review_call_count
    from google.genai._gaos.types.decisions import NoulAnswer, ScoreAnswer
    from types import SimpleNamespace

    reset_review_call_count()

    # Score is 2.38 (below 2.5) and levels 0+1 is 0.40 (> 0.35)
    mock_resp = SimpleNamespace(
        answers={
            "subject_named_in_first_two": NoulAnswer(noul=0.95, confidence=0.95, type="noul"),
            "every_proper_noun_and_jargon_explained": NoulAnswer(noul=0.90, confidence=0.90, type="noul"),
            "hook_names_subject": NoulAnswer(noul=0.92, confidence=0.92, type="noul"),
            "closing_question_specific": NoulAnswer(noul=0.88, confidence=0.88, type="noul"),
            "numbers_in_sources": NoulAnswer(noul=0.99, confidence=0.99, type="noul"),
            "clarity_score": ScoreAnswer(
                score=2.38,
                confidence=0.42,
                probabilities={"0": 0.15, "1": 0.25, "2": 0.40, "3": 0.20},
                legend={},
                type="score",
            ),
        },
        usage=SimpleNamespace(total_input_tokens=450, total_output_tokens=0),
    )

    mock_client = SimpleNamespace(
        decisions=SimpleNamespace(create=lambda **kwargs: mock_resp)
    )
    monkeypatch.setattr("verticals.reviewer._get_genai_client", lambda: mock_client)

    verdict = review_script("Script text", "Topic C")
    assert verdict["pass"] is False
    assert any("2.5" in p for p in verdict["problems"])
    assert any("confusion risk" in p for p in verdict["problems"])
    assert verdict["soft_fail"] is False  # score confidence does NOT cause soft fail!


def test_review_script_decisions_fallback_to_ling(monkeypatch):
    from verticals.reviewer import review_script, reset_review_call_count

    reset_review_call_count()

    # Decisions API throws an error
    def mock_bad_client():
        raise RuntimeError("Decisions API unavailable / rate limited")

    monkeypatch.setattr("verticals.reviewer._get_genai_client", mock_bad_client)
    monkeypatch.setattr("verticals.reviewer._call_antling_chat", lambda *args, **kwargs: ('{"pass": true, "problems": []}', {"prompt_tokens": 10, "completion_tokens": 10}))
    monkeypatch.setattr("verticals.reviewer._get_antling_key", lambda: "fake-key")

    verdict = review_script("Some script", "Some topic")
    assert verdict["pass"] is True
    assert not verdict.get("skipped", False)


def test_image_relevance_gate_thresholds(monkeypatch):
    from verticals.real_photos import verify_relevance_gate_decisions
    from google.genai._gaos.types.decisions import NoulAnswer
    from types import SimpleNamespace

    # 1. Accept (>= 0.8)
    mock_resp_acc = SimpleNamespace(
        answers={"image_relevance": NoulAnswer(noul=0.88, confidence=0.90, type="noul")},
        usage=SimpleNamespace(total_input_tokens=395, total_output_tokens=0),
    )
    monkeypatch.setattr("verticals.reviewer._get_genai_client", lambda: SimpleNamespace(
        decisions=SimpleNamespace(create=lambda **kwargs: mock_resp_acc)
    ))
    status, prob, conf = verify_relevance_gate_decisions("fakeb64", "Entity", "Desc")
    assert status == "ACCEPT"
    assert prob >= 0.8

    # 2. Reject (< 0.5)
    mock_resp_rej = SimpleNamespace(
        answers={"image_relevance": NoulAnswer(noul=0.15, confidence=0.92, type="noul")},
        usage=SimpleNamespace(total_input_tokens=395, total_output_tokens=0),
    )
    monkeypatch.setattr("verticals.reviewer._get_genai_client", lambda: SimpleNamespace(
        decisions=SimpleNamespace(create=lambda **kwargs: mock_resp_rej)
    ))
    status, prob, conf = verify_relevance_gate_decisions("fakeb64", "Entity", "Desc")
    assert status == "REJECT"
    assert prob < 0.5

    # 3. Borderline (0.5 to 0.8)
    mock_resp_bord = SimpleNamespace(
        answers={"image_relevance": NoulAnswer(noul=0.68, confidence=0.75, type="noul")},
        usage=SimpleNamespace(total_input_tokens=395, total_output_tokens=0),
    )
    monkeypatch.setattr("verticals.reviewer._get_genai_client", lambda: SimpleNamespace(
        decisions=SimpleNamespace(create=lambda **kwargs: mock_resp_bord)
    ))
    status, prob, conf = verify_relevance_gate_decisions("fakeb64", "Entity", "Desc")
    assert status == "BORDERLINE"
    assert 0.5 <= prob < 0.8


def test_topic_ranking_decisions(monkeypatch):
    from verticals.topics.engine import TopicEngine
    from verticals.topics.base import TopicCandidate
    from google.genai._gaos.types.decisions import ScoreAnswer
    from types import SimpleNamespace

    cands = [
        TopicCandidate(title="Curiosity Short A", source="manual", trending_score=0.9),
        TopicCandidate(title="Boring City Council Note", source="manual", trending_score=0.8),
    ]

    mock_resp = SimpleNamespace(
        answers={
            "cand_0": ScoreAnswer(score=2.8, confidence=0.9, probabilities={}, legend={}, type="score"),
            "cand_1": ScoreAnswer(score=0.3, confidence=0.9, probabilities={}, legend={}, type="score"),
        }
    )
    monkeypatch.setattr("verticals.reviewer._get_genai_client", lambda: SimpleNamespace(
        decisions=SimpleNamespace(create=lambda **kwargs: mock_resp)
    ))

    engine = TopicEngine(niche="general")
    scored = engine.score_topics_decisions(cands)
    assert scored[0].metadata["curiosity_score"] == 2.8
    assert scored[1].metadata["curiosity_score"] == 0.3

