"""Tests for model guide recommendation logic."""

from __future__ import annotations

from services.model_scanner.model_guide_data import MODEL_FORMATS, DISTILLED_LORA_INFO, recommend_format


class TestRecommendFormat:
    def test_48gb_recommends_bf16(self) -> None:
        assert recommend_format(48) == "bf16"

    def test_32gb_recommends_bf16(self) -> None:
        assert recommend_format(32) == "bf16"

    def test_24gb_recommends_fp8(self) -> None:
        assert recommend_format(24) == "fp8"

    def test_20gb_recommends_fp8(self) -> None:
        assert recommend_format(20) == "fp8"

    def test_16gb_recommends_gguf_q5k(self) -> None:
        assert recommend_format(16) == "gguf_q5k"

    def test_12gb_recommends_gguf_q4k(self) -> None:
        assert recommend_format(12) == "gguf_q4k"

    def test_10gb_recommends_gguf_q4k(self) -> None:
        assert recommend_format(10) == "gguf_q4k"

    def test_8gb_recommends_api_only(self) -> None:
        assert recommend_format(8) == "api_only"

    def test_none_vram_defaults_to_bf16(self) -> None:
        assert recommend_format(None) == "bf16"


class TestModelFormatsData:
    def test_all_formats_have_required_fields(self) -> None:
        for fmt in MODEL_FORMATS:
            assert fmt.id
            assert fmt.name
            assert fmt.min_vram_gb > 0
            assert fmt.download_url.startswith("https://")

    def test_distilled_lora_info_has_url(self) -> None:
        assert DISTILLED_LORA_INFO.download_url.startswith("https://")
        assert DISTILLED_LORA_INFO.size_gb > 0
