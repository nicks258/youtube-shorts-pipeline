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
