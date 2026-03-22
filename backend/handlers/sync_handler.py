"""Handler for Palette sync operations."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from services.http_client.http_client import HTTPClient
from services.palette_sync_client.palette_sync_client import PaletteSyncClient
from state.app_state_types import AppState
from state.lora_library import LoraEntry, LoraLibraryStore

logger = logging.getLogger(__name__)

# Fallback pricing (cents) when the Palette credits endpoint doesn't return it.
# Values sourced from the live Palette /api/desktop/credits/check endpoint.
_DEFAULT_PRICING: dict[str, int] = {
    "video_t2v": 10,
    "video_i2v": 16,
    "video_seedance": 5,
    "image": 6,
    "image_edit": 20,
    "audio": 15,
    "text_enhance": 3,
}

# Known FLUX Klein 9B LoRA weights URLs from Palette's built-in library.
# The Palette API may not include download URLs for all LoRAs, so we map
# known IDs to their weights URLs as a fallback.
_KNOWN_LORA_WEIGHTS: dict[str, tuple[str, float]] = {
    "claymation-k9b": (
        "https://huuezdiitpmafkljkvui.supabase.co/storage/v1/object/public/loras/community/claymation_flux_lora_v1.safetensors",
        1.0,
    ),
    "inflate-k9b": (
        "https://huuezdiitpmafkljkvui.supabase.co/storage/v1/object/public/loras/community/inflate_it.safetensors",
        1.0,
    ),
    "disney-golden-age-k9b": (
        "https://huuezdiitpmafkljkvui.supabase.co/storage/v1/object/public/loras/community/disney_golden_age.safetensors",
        1.0,
    ),
    "nava-k9b": (
        "https://v3.fal.media/files/monkey/oF3DkwBOmrzohIKhCfNie_pytorch_lora_weights.safetensors",
        1.0,
    ),
    "dcau-k9b": (
        "https://huuezdiitpmafkljkvui.supabase.co/storage/v1/object/public/loras/community/jRB4slNlO3KYd18ROU5Up_pytorch_lora_weights_comfy_converted.safetensors",
        1.0,
    ),
    "cinematic-filmstill-k9b": (
        "https://huuezdiitpmafkljkvui.supabase.co/storage/v1/object/public/loras/community/cinematic_filmstill.safetensors",
        1.0,
    ),
    "consistency-k9b": (
        "https://pub-060813fba4064da4815db04b08604ce7.r2.dev/consistency_lora_v3.safetensors",
        0.8,
    ),
}


class SyncHandler:
    def __init__(
        self,
        state: AppState,
        palette_sync_client: PaletteSyncClient,
        http: HTTPClient,
        lora_store: LoraLibraryStore,
        loras_dir: Path,
    ) -> None:
        self._state = state
        self._client = palette_sync_client
        self._http = http
        self._lora_store = lora_store
        self._loras_dir = loras_dir
        self._cached_user: dict[str, Any] | None = None

    def _try_refresh(self) -> dict[str, Any] | None:
        """Attempt to refresh an expired JWT. Returns user info or None."""
        refresh_token = self._state.app_settings.palette_refresh_token
        if not refresh_token:
            return None
        try:
            result = self._client.refresh_access_token(refresh_token=refresh_token)
            self._state.app_settings.palette_api_key = result["access_token"]
            self._state.app_settings.palette_refresh_token = result["refresh_token"]
            self._cached_user = result["user"]
            return result["user"]
        except Exception:
            return None

    def get_status(self) -> dict[str, Any]:
        api_key = self._state.app_settings.palette_api_key
        if not api_key:
            return {"connected": False, "user": None}
        if self._cached_user is not None:
            return {"connected": True, "user": self._cached_user}
        try:
            user = self._client.validate_connection(api_key=api_key)
            self._cached_user = user
            return {"connected": True, "user": user}
        except Exception as exc:
            # JWT might be expired — try refreshing
            user = self._try_refresh()
            if user is not None:
                return {"connected": True, "user": user}
            self._cached_user = None
            return {"connected": False, "user": None, "error": str(exc)}

    def connect(self, token: str) -> dict[str, Any]:
        """Store an auth token and validate it. Returns status."""
        try:
            user = self._client.validate_connection(api_key=token)
        except Exception as exc:
            return {"connected": False, "error": str(exc)}
        self._state.app_settings.palette_api_key = token
        self._cached_user = user
        return {"connected": True, "user": user}

    def login(self, email: str, password: str) -> dict[str, Any]:
        """Sign in with email/password and store the session tokens."""
        try:
            result = self._client.sign_in_with_email(email=email, password=password)
        except Exception as exc:
            return {"connected": False, "error": str(exc)}
        self._state.app_settings.palette_api_key = result["access_token"]
        self._state.app_settings.palette_refresh_token = result["refresh_token"]
        self._cached_user = result["user"]
        return {"connected": True, "user": result["user"]}

    def disconnect(self) -> dict[str, Any]:
        """Clear the stored auth token and cached user."""
        self._state.app_settings.palette_api_key = ""
        self._state.app_settings.palette_refresh_token = ""
        self._cached_user = None
        return {"connected": False, "user": None}

    def get_credits(self) -> dict[str, Any]:
        api_key = self._state.app_settings.palette_api_key
        if not api_key:
            return {"connected": False, "balance_cents": None, "pricing": None}
        try:
            credits = self._client.get_credits(api_key=api_key)
            result: dict[str, Any] = {"connected": True, **credits}
        except Exception:
            result = {"connected": True, "balance_cents": None, "pricing": None}

        # If the credits endpoint didn't return balance_cents, fall back
        # to the check endpoint which reliably includes the balance.
        if result.get("balance_cents") is None:
            try:
                check = self._client.check_credits(
                    api_key=api_key, generation_type="image", count=1,
                )
                result["balance_cents"] = check.get("balance_cents")
            except Exception:
                pass

        # Ensure pricing is present — fall back to known defaults if the
        # credits endpoint didn't provide it.
        if not result.get("pricing"):
            result["pricing"] = _DEFAULT_PRICING

        return result

    def check_credits(self, generation_type: str, count: int = 1) -> dict[str, Any]:
        api_key = self._state.app_settings.palette_api_key
        if not api_key:
            return {"connected": False, "can_afford": True}
        try:
            return {"connected": True, **self._client.check_credits(
                api_key=api_key, generation_type=generation_type, count=count,
            )}
        except Exception as exc:
            logger.warning("Credit check failed: %s", exc)
            # Fail open — don't block generation if credit check is unavailable
            return {"connected": False, "can_afford": True}

    def deduct_credits(
        self, generation_type: str, count: int = 1,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        api_key = self._state.app_settings.palette_api_key
        if not api_key:
            return {"deducted": False}
        try:
            result = self._client.deduct_credits(
                api_key=api_key, generation_type=generation_type,
                count=count, metadata=metadata,
            )
            return {"deducted": True, **result}
        except Exception as exc:
            logger.warning("Credit deduction failed: %s", exc)
            return {"deducted": False, "error": str(exc)}

    def list_gallery(
        self, page: int = 1, per_page: int = 50, asset_type: str = "all",
    ) -> dict[str, Any]:
        api_key = self._state.app_settings.palette_api_key
        if not api_key:
            return {"connected": False, "items": []}
        try:
            return {
                "connected": True,
                **self._client.list_gallery(
                    api_key=api_key, page=page, per_page=per_page, asset_type=asset_type,
                ),
            }
        except Exception as exc:
            logger.warning("Palette gallery list failed: %s", exc)
            return {"connected": False, "items": [], "error": str(exc)}

    def list_characters(self) -> dict[str, Any]:
        api_key = self._state.app_settings.palette_api_key
        if not api_key:
            return {"connected": False, "characters": []}
        try:
            return {"connected": True, **self._client.list_characters(api_key=api_key)}
        except Exception as exc:
            logger.warning("Palette characters list failed: %s", exc)
            return {"connected": False, "characters": [], "error": str(exc)}

    def list_styles(self) -> dict[str, Any]:
        api_key = self._state.app_settings.palette_api_key
        if not api_key:
            return {"connected": False, "styles": []}
        try:
            return {"connected": True, **self._client.list_styles(api_key=api_key)}
        except Exception as exc:
            logger.warning("Palette styles list failed: %s", exc)
            return {"connected": False, "styles": [], "error": str(exc)}

    def list_references(self, category: str | None = None) -> dict[str, Any]:
        api_key = self._state.app_settings.palette_api_key
        if not api_key:
            return {"connected": False, "references": []}
        try:
            return {
                "connected": True,
                **self._client.list_references(api_key=api_key, category=category),
            }
        except Exception as exc:
            logger.warning("Palette references list failed: %s", exc)
            return {"connected": False, "references": [], "error": str(exc)}

    def enhance_prompt(self, prompt: str, level: str = "2x") -> dict[str, Any]:
        api_key = self._state.app_settings.palette_api_key
        if not api_key:
            return {"error": "Not connected to Palette"}
        try:
            return self._client.enhance_prompt(api_key=api_key, prompt=prompt, level=level)
        except Exception as exc:
            logger.warning("Palette prompt enhance failed: %s", exc)
            return {"error": str(exc)}

    def sync_loras(self) -> dict[str, Any]:
        """Fetch LoRA catalog from Palette and download any new LoRAs locally.

        Returns {"synced": N, "skipped": N, "failed": N}.
        """
        api_key = self._state.app_settings.palette_api_key
        if not api_key:
            return {"connected": False, "error": "Not connected to Palette"}

        try:
            data = self._client.list_loras(api_key=api_key)
        except Exception as exc:
            logger.warning("Palette LoRA list failed: %s", exc)
            return {"connected": False, "error": str(exc)}

        palette_loras = data.get("loras", [])
        existing_ids = {e.id for e in self._lora_store.list_all()}

        synced = 0
        skipped = 0
        failed = 0

        for lora in palette_loras:
            lora_id = lora.get("id", "")
            catalog_id = f"palette:{lora_id}"
            if catalog_id in existing_ids:
                skipped += 1
                continue

            # Get download URL — either from API response or hardcoded map
            weights_url = lora.get("weights_url") or lora.get("download_url") or ""
            if not weights_url:
                weights_url = _KNOWN_LORA_WEIGHTS.get(lora_id, ("", 1.0))[0]
            if not weights_url:
                logger.debug("Skipping LoRA %s — no download URL available", lora_id)
                skipped += 1
                continue

            try:
                self._download_and_register_lora(lora, catalog_id, weights_url)
                synced += 1
            except Exception:
                logger.warning("Failed to sync LoRA %s", lora_id, exc_info=True)
                failed += 1

        return {"connected": True, "synced": synced, "skipped": skipped, "failed": failed}

    def _download_and_register_lora(
        self, lora: dict[str, Any], catalog_id: str, weights_url: str,
    ) -> None:
        """Download LoRA weights and register in local catalog."""
        lora_id = lora.get("id", "unknown")
        filename = f"palette_{lora_id}.safetensors"
        dest = self._loras_dir / filename

        if not dest.exists():
            logger.info("Downloading LoRA %s from %s", lora_id, weights_url)
            resp = self._http.get(weights_url, timeout=300)
            if resp.status_code != 200:
                raise RuntimeError(f"Download failed: HTTP {resp.status_code}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(resp.content)

        entry = LoraEntry(
            id=catalog_id,
            name=f"[Palette] {lora.get('name', lora_id)}",
            file_path=str(dest),
            file_size_bytes=dest.stat().st_size,
            thumbnail_url=lora.get("thumbnail_url", ""),
            trigger_phrase=lora.get("trigger_word", ""),
            base_model="flux-klein-9b",
        )
        self._lora_store.add(entry)
