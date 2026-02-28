"""Tests for SDK transport retry logic."""

import logging
import queue
import time
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import httpx

from invokelens_sdk.transport import (
    EventTransport,
    MAX_RETRIES,
    INITIAL_BACKOFF_SECONDS,
    BACKOFF_MULTIPLIER,
)


def _make_transport(**kwargs):
    """Create a transport with background worker stopped for synchronous testing."""
    defaults = {
        "endpoint_url": "http://test-endpoint",
        "api_key": "test-key",
        "batch_size": 10,
        "flush_interval_seconds": 0.1,
        "max_queue_size": 100,
    }
    defaults.update(kwargs)
    t = EventTransport(**defaults)
    # Stop the background worker so we can call _flush_http directly
    t._shutdown.set()
    t._worker.join(timeout=2)
    return t


class TestFlushHttpSuccess:
    @patch("httpx.post")
    def test_success_on_200(self, mock_post):
        """Successful 200 response should not retry."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        t = _make_transport()
        t._flush_http(["event1", "event2"])

        assert mock_post.call_count == 1

    @patch("httpx.post")
    def test_success_on_201(self, mock_post):
        """Any 2xx response is treated as success."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_post.return_value = mock_response

        t = _make_transport()
        t._flush_http(["event1"])

        assert mock_post.call_count == 1


class TestFlushHttpClientError:
    @patch("httpx.post")
    def test_no_retry_on_400(self, mock_post):
        """400 client error should NOT retry."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad request"
        mock_post.return_value = mock_response

        t = _make_transport()
        t._flush_http(["event1"])

        assert mock_post.call_count == 1

    @patch("httpx.post")
    def test_no_retry_on_401(self, mock_post):
        """401 unauthorized should NOT retry."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_post.return_value = mock_response

        t = _make_transport()
        t._flush_http(["event1"])

        assert mock_post.call_count == 1

    @patch("httpx.post")
    def test_no_retry_on_422(self, mock_post):
        """422 validation error should NOT retry."""
        mock_response = MagicMock()
        mock_response.status_code = 422
        mock_response.text = "Validation error"
        mock_post.return_value = mock_response

        t = _make_transport()
        t._flush_http(["event1"])

        assert mock_post.call_count == 1


class TestFlushHttpServerError:
    @patch("time.sleep")
    @patch("httpx.post")
    def test_retries_on_500(self, mock_post, mock_sleep):
        """500 server error should retry MAX_RETRIES times."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        t = _make_transport()
        t._flush_http(["event1"])

        assert mock_post.call_count == MAX_RETRIES + 1
        assert mock_sleep.call_count == MAX_RETRIES

    @patch("time.sleep")
    @patch("httpx.post")
    def test_retries_on_503(self, mock_post, mock_sleep):
        """503 service unavailable should retry."""
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_post.return_value = mock_response

        t = _make_transport()
        t._flush_http(["event1"])

        assert mock_post.call_count == MAX_RETRIES + 1

    @patch("time.sleep")
    @patch("httpx.post")
    def test_succeeds_after_retry(self, mock_post, mock_sleep):
        """If retry succeeds, stop retrying."""
        fail_response = MagicMock()
        fail_response.status_code = 500

        success_response = MagicMock()
        success_response.status_code = 200

        mock_post.side_effect = [fail_response, success_response]

        t = _make_transport()
        t._flush_http(["event1"])

        assert mock_post.call_count == 2
        assert mock_sleep.call_count == 1


class TestFlushHttpNetworkError:
    @patch("time.sleep")
    @patch("httpx.post")
    def test_retries_on_timeout(self, mock_post, mock_sleep):
        """Timeout exceptions should trigger retries."""
        mock_post.side_effect = httpx.TimeoutException("Connection timed out")

        t = _make_transport()
        t._flush_http(["event1"])

        assert mock_post.call_count == MAX_RETRIES + 1

    @patch("time.sleep")
    @patch("httpx.post")
    def test_retries_on_connection_error(self, mock_post, mock_sleep):
        """Connection errors should trigger retries."""
        mock_post.side_effect = httpx.ConnectError("Connection refused")

        t = _make_transport()
        t._flush_http(["event1"])

        assert mock_post.call_count == MAX_RETRIES + 1


class TestBackoff:
    @patch("time.sleep")
    @patch("httpx.post")
    def test_exponential_backoff_timing(self, mock_post, mock_sleep):
        """Verify backoff increases: 1s, 2s, 4s."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        t = _make_transport()
        t._flush_http(["event1"])

        expected_backoffs = [
            INITIAL_BACKOFF_SECONDS,
            INITIAL_BACKOFF_SECONDS * BACKOFF_MULTIPLIER,
            INITIAL_BACKOFF_SECONDS * BACKOFF_MULTIPLIER ** 2,
        ]
        actual_backoffs = [call.args[0] for call in mock_sleep.call_args_list]
        assert actual_backoffs == expected_backoffs


class TestQueueFull:
    def test_send_drops_when_queue_full(self, caplog):
        """When queue is full, send() should drop event and log warning."""
        t = _make_transport(max_queue_size=1)

        # Fill the queue manually
        t._queue.put("item1")

        # Create a mock event
        mock_event = MagicMock()
        mock_event.model_dump_json.return_value = '{"test": true}'

        with caplog.at_level(logging.WARNING, logger="invokelens_sdk.transport"):
            t.send(mock_event)

        assert "queue full" in caplog.text.lower()


class TestLogging:
    @patch("time.sleep")
    @patch("httpx.post")
    def test_logs_error_after_all_retries(self, mock_post, mock_sleep, caplog):
        """After all retries exhausted, should log an error."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        t = _make_transport()
        with caplog.at_level(logging.ERROR, logger="invokelens_sdk.transport"):
            t._flush_http(["event1", "event2"])

        assert "failed after" in caplog.text.lower()
        assert "2 events" in caplog.text

    @patch("httpx.post")
    def test_logs_warning_on_client_error(self, mock_post, caplog):
        """4xx errors should log a warning."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad request body"
        mock_post.return_value = mock_response

        t = _make_transport()
        with caplog.at_level(logging.WARNING, logger="invokelens_sdk.transport"):
            t._flush_http(["event1"])

        assert "rejected" in caplog.text.lower()
        assert "400" in caplog.text
