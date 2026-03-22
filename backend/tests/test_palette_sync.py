"""Integration tests for Palette LoRA sync."""

from __future__ import annotations

from starlette.testclient import TestClient
from tests.fakes.services import FakeResponse


class TestSyncLorasNoConnection:
    def test_sync_loras_without_api_key_returns_not_connected(
        self, client: TestClient, test_state,
    ) -> None:
        # Ensure no API key is set
        test_state.state.app_settings.palette_api_key = ""
        resp = client.post("/api/sync/library/sync-loras")
        assert resp.status_code == 200
        data = resp.json()
        assert data["connected"] is False


class TestSyncLorasWithConnection:
    def test_sync_loras_empty_catalog(
        self, client: TestClient, test_state, fake_services,
    ) -> None:
        test_state.state.app_settings.palette_api_key = "test-jwt-token"
        # FakePaletteSyncClient.list_loras returns {"loras": []} by default
        resp = client.post("/api/sync/library/sync-loras")
        assert resp.status_code == 200
        data = resp.json()
        assert data["connected"] is True
        assert data["synced"] == 0
        assert data["skipped"] == 0

    def test_sync_loras_downloads_new_lora(
        self, client: TestClient, test_state, fake_services,
    ) -> None:
        test_state.state.app_settings.palette_api_key = "test-jwt-token"

        # Override list_loras to return a LoRA with a known download URL
        original_list = fake_services.palette_sync_client.list_loras
        def mock_list_loras(*, api_key: str) -> dict:
            return {"loras": [{
                "id": "dcau-k9b",
                "name": "DCAU",
                "type": "style",
                "trigger_word": "DC animation style",
                "thumbnail_url": "",
                "compatible_models": ["flux-2-klein-9b"],
            }]}
        fake_services.palette_sync_client.list_loras = mock_list_loras

        # Queue a fake HTTP response for the LoRA download
        fake_services.http.queue("get", FakeResponse(
            status_code=200,
            content=b"\x00" * 1024,  # Fake safetensors content
        ))

        resp = client.post("/api/sync/library/sync-loras")
        assert resp.status_code == 200
        data = resp.json()
        assert data["connected"] is True
        assert data["synced"] == 1

        # Verify the LoRA was registered in the catalog
        entries = test_state.lora._store.list_all()
        palette_entries = [e for e in entries if e.id.startswith("palette:")]
        assert len(palette_entries) == 1
        assert palette_entries[0].name == "[Palette] DCAU"
        assert palette_entries[0].trigger_phrase == "DC animation style"

        # Restore
        fake_services.palette_sync_client.list_loras = original_list

    def test_sync_loras_skips_already_synced(
        self, client: TestClient, test_state, fake_services,
    ) -> None:
        test_state.state.app_settings.palette_api_key = "test-jwt-token"

        # Pre-register a Palette LoRA in the catalog
        from state.lora_library import LoraEntry
        test_state.lora._store.add(LoraEntry(
            id="palette:dcau-k9b",
            name="[Palette] DCAU",
            file_path="/fake/path.safetensors",
            file_size_bytes=1024,
        ))

        # Override list_loras to return the same LoRA
        original_list = fake_services.palette_sync_client.list_loras
        def mock_list_loras(*, api_key: str) -> dict:
            return {"loras": [{
                "id": "dcau-k9b",
                "name": "DCAU",
                "type": "style",
                "trigger_word": "DC animation style",
                "thumbnail_url": "",
            }]}
        fake_services.palette_sync_client.list_loras = mock_list_loras

        resp = client.post("/api/sync/library/sync-loras")
        assert resp.status_code == 200
        data = resp.json()
        assert data["skipped"] == 1
        assert data["synced"] == 0

        fake_services.palette_sync_client.list_loras = original_list
