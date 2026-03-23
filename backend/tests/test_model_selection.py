"""Integration tests for video model scan, select, and guide endpoints."""

from __future__ import annotations

import struct
from pathlib import Path

from app_handler import AppHandler


def _write_minimal_gguf(path: Path) -> None:
    with open(path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))
        f.write(struct.pack("<Q", 0))
        f.write(struct.pack("<Q", 0))


class TestVideoModelScan:
    def test_scan_returns_empty_for_default_path(self, client) -> None:
        resp = client.get("/api/models/video/scan")
        assert resp.status_code == 200
        data = resp.json()
        assert data["models"] == []
        assert data["distilled_lora_found"] is False


class TestVideoModelSelect:
    def test_select_nonexistent_model_returns_400(self, client) -> None:
        resp = client.post("/api/models/video/select", json={"model": "nonexistent.gguf"})
        assert resp.status_code == 400

    def test_select_valid_model_updates_settings(self, client, test_state: AppHandler) -> None:
        models_dir = test_state.config.models_dir
        gguf_path = models_dir / "test-model.gguf"
        _write_minimal_gguf(gguf_path)

        client.post("/api/settings", json={"customVideoModelPath": str(models_dir)})

        resp = client.post("/api/models/video/select", json={"model": "test-model.gguf"})
        assert resp.status_code == 200

        assert test_state.state.app_settings.selected_video_model == "test-model.gguf"


class TestVideoModelGuide:
    def test_guide_returns_formats_and_recommendation(self, client) -> None:
        resp = client.get("/api/models/video/guide")
        assert resp.status_code == 200
        data = resp.json()
        assert "formats" in data
        assert "recommended_format" in data
        assert "distilled_lora" in data
        assert len(data["formats"]) > 0
