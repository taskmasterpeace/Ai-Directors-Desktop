"""Tests for R2 upload integration."""

from __future__ import annotations


def test_r2_client_is_configured_when_credentials_present() -> None:
    from services.r2_client.r2_client_impl import R2ClientImpl

    client = R2ClientImpl(
        access_key_id="test",
        secret_access_key="test",
        endpoint="https://example.com",
        bucket="test-bucket",
        public_url="https://pub.example.com",
    )
    assert client.is_configured() is True


def test_r2_client_not_configured_when_empty() -> None:
    from services.r2_client.r2_client_impl import R2ClientImpl

    client = R2ClientImpl(
        access_key_id="",
        secret_access_key="",
        endpoint="",
        bucket="",
        public_url="",
    )
    assert client.is_configured() is False
