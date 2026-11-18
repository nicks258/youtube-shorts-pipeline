"""Tests for YouTube upload functionality and privacy settings."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from verticals.upload import upload_to_youtube


@pytest.fixture
def mock_youtube_deps():
    with patch("verticals.upload.get_youtube_token_path") as mock_token_path, \
         patch("google.oauth2.credentials.Credentials.from_authorized_user_file") as mock_creds_factory, \
         patch("googleapiclient.discovery.build") as mock_build, \
         patch("googleapiclient.http.MediaFileUpload") as mock_media:

        mock_token_path.return_value = Path("/tmp/mock_token.json")
        creds = MagicMock()
        creds.expired = False
        mock_creds_factory.return_value = creds

        youtube_mock = MagicMock()
        videos_mock = MagicMock()
        insert_mock = MagicMock()
        req_mock = MagicMock()

        # Simulate chunk upload completion
        req_mock.next_chunk.return_value = (None, {"id": "test_video_123"})
        insert_mock.return_value = req_mock
        videos_mock.insert = insert_mock
        youtube_mock.videos.return_value = videos_mock
        mock_build.return_value = youtube_mock

        yield {
            "build": mock_build,
            "youtube": youtube_mock,
            "insert": insert_mock,
        }


def test_upload_defaults_to_private(mock_youtube_deps, monkeypatch):
    monkeypatch.delenv("PRIVACY_STATUS", raising=False)
    draft = {
        "news": "Test Topic",
        "youtube_title": "Test Title",
        "youtube_description": "Test Desc",
        "youtube_tags": "tag1, tag2",
    }
    video_path = Path("/tmp/video.mp4")

    url = upload_to_youtube(video_path, draft)

    assert url == "https://youtu.be/test_video_123"
    call_kwargs = mock_youtube_deps["insert"].call_args.kwargs
    body = call_kwargs["body"]
    assert body["status"]["privacyStatus"] == "private"
    assert body["status"]["selfDeclaredMadeForKids"] is False


def test_upload_respects_env_var(mock_youtube_deps, monkeypatch):
    monkeypatch.setenv("PRIVACY_STATUS", "unlisted")
    draft = {
        "news": "Test Topic",
    }
    video_path = Path("/tmp/video.mp4")

    upload_to_youtube(video_path, draft)

    call_kwargs = mock_youtube_deps["insert"].call_args.kwargs
    body = call_kwargs["body"]
    assert body["status"]["privacyStatus"] == "unlisted"


def test_upload_respects_draft_privacy_status(mock_youtube_deps, monkeypatch):
    monkeypatch.setenv("PRIVACY_STATUS", "public")
    draft = {
        "news": "Test Topic",
        "privacy_status": "private",
    }
    video_path = Path("/tmp/video.mp4")

    upload_to_youtube(video_path, draft)

    call_kwargs = mock_youtube_deps["insert"].call_args.kwargs
    body = call_kwargs["body"]
    assert body["status"]["privacyStatus"] == "private"
