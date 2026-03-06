# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LTX Desktop is an open-source Electron app for AI video generation using LTX models. It supports local generation on Windows NVIDIA GPUs (32GB+ VRAM) and API-only mode for unsupported hardware and macOS.

Three-layer architecture:

```
Renderer (React + TS) --HTTP: localhost:8000--> Backend (FastAPI + Python)
Renderer (React + TS) --IPC: window.electronAPI--> Electron main (TS)
Electron main --> OS integration (files, dialogs, ffmpeg, process mgmt)
Backend --> Local models + GPU | External APIs (when API-backed)
```

- **Frontend** (`frontend/`): React 18 + TypeScript + Tailwind CSS renderer
- **Electron** (`electron/`): Main process managing app lifecycle, IPC, Python backend process, ffmpeg export. Renderer is sandboxed (`contextIsolation: true`, `nodeIntegration: false`).
- **Backend** (`backend/`): Python FastAPI server (port 8000) handling ML model orchestration and generation

## Common Commands

| Command | Purpose |
|---|---|
| `pnpm dev` | Start dev server (Vite + Electron + Python backend) |
| `pnpm dev:debug` | Dev with Electron inspector (port 9229) + Python debugpy |
| `pnpm typecheck` | Run TypeScript (`tsc --noEmit`) and Python (`pyright`) type checks |
| `pnpm typecheck:ts` | TypeScript only |
| `pnpm typecheck:py` | Python pyright only (`cd backend && uv run pyright`) |
| `pnpm backend:test` | Run Python pytest tests (`cd backend && uv sync --frozen --extra test --extra dev && uv run pytest -v --tb=short`) |
| `pnpm build:frontend` | Vite frontend build only |
| `pnpm build:win` / `pnpm build:mac` | Full platform builds (installer) |
| `pnpm build:fast:win` / `pnpm build:fast:mac` | Unpacked build, skip Python bundling |
| `pnpm setup:dev:win` / `pnpm setup:dev:mac` | One-time dev environment setup |

Run a single backend test: `cd backend && uv run pytest tests/test_generation.py -v --tb=short`

Run a single test function: `cd backend && uv run pytest tests/test_generation.py::test_name -v --tb=short`

## CI Checks

PRs must pass: `pnpm typecheck` + `pnpm backend:test` + frontend Vite build.

## Frontend Architecture

- **Path alias**: `@/*` maps to `frontend/*` (configured in `tsconfig.json` and `vite.config.ts`)
- **State management**: React contexts only (`ProjectContext`, `AppSettingsContext`, `KeyboardShortcutsContext`) — no Redux/Zustand
- **Routing**: View-based via `ProjectContext` with views: `home`, `project`, `playground`
- **IPC bridge**: All Electron communication through `window.electronAPI` (defined in `electron/preload.ts`). Key methods: `getBackendUrl`, `readLocalFile`, `checkGpu`, `getAppInfo`, `exportVideo`, `showSaveDialog`, `showItemInFolder`
- **Backend calls**: Frontend calls `http://localhost:8000` directly
- **Styling**: Tailwind with custom semantic color tokens via CSS variables; utilities from `class-variance-authority` + `clsx` + `tailwind-merge`
- **Views**: `Home.tsx`, `GenSpace.tsx`, `Project.tsx`, `Playground.tsx`, `VideoEditor.tsx` (largest frontend file), `editor/` subdirectory
- **No frontend tests** currently exist

## Backend Architecture

Request flow: `_routes/* (thin) -> AppHandler -> handlers/* (logic) -> services/* (side effects) + state/* (mutations)`

Key patterns:
- **Routes** (`_routes/`): Thin plumbing only — parse input, call handler, return typed output. No business logic.
- **AppHandler** (`app_handler.py`): Single composition root owning all sub-handlers, state, and lock. Sub-handlers accessed as `handler.health`, `handler.models`, `handler.downloads`, etc.
- **State** (`state/`): Centralized `AppState` using discriminated union types for state machines (e.g., `GenerationState = GenerationRunning | GenerationComplete | GenerationError | GenerationCancelled`)
- **Services** (`services/`): Protocol interfaces with real implementations and fake test implementations. The test boundary for heavy side effects (GPU, network).
- **Concurrency**: Thread pool with shared `RLock`. Pattern: lock -> read/validate -> unlock -> heavy work -> lock -> write. Never hold lock during heavy compute/IO. Use `handlers.base.with_state_lock` decorator.
- **Exception handling**: Boundary-owned traceback policy. Handlers raise `HTTPError` with `from exc` chaining; `app_factory.py` owns logging. Don't `logger.exception()` then rethrow.
- **Naming**: `*Payload` for DTOs/TypedDicts, `*Like` for structural wrappers, `Fake*` for test implementations

### Backend Composition Roots

- `ltx2_server.py`: Runtime bootstrap (logging, `RuntimeConfig`, `AppHandler`, `uvicorn`)
- `app_factory.py`: FastAPI app factory (routers, DI init, exception handling) — importable from tests
- `state/deps.py`: FastAPI dependency hook (`get_state_service()` returns shared `AppHandler`; tests override via `set_state_service_for_tests()`)

### Backend Testing

- Integration-first using Starlette `TestClient` against real FastAPI app
- **No mocks**: `test_no_mock_usage.py` enforces no `unittest.mock`. Swap services via `ServiceBundle` fakes only.
- Fakes live in `tests/fakes/`; `conftest.py` wires fresh `AppHandler` per test
- Pyright strict mode is also enforced as a test (`test_pyright.py`)

### Adding a Backend Feature

1. Define request/response models in `api_types.py`
2. Add endpoint in `_routes/<domain>.py` delegating to handler
3. Implement logic in `handlers/<domain>_handler.py` with lock-aware state transitions
4. If new heavy side effect needed, add service in `services/` with Protocol + real + fake implementations
5. Add integration test in `tests/` using fake services

## TypeScript Config

- Strict mode with `noUnusedLocals`, `noUnusedParameters`
- Frontend: ES2020 target, React JSX
- Electron main process: ESNext, compiled to `dist-electron/`
- Preload script must be CommonJS (configured in `vite.config.ts` rollup output)

## Python Config

- Python 3.12+ required (`.python-version` pins 3.13), managed with `uv`
- Pyright strict mode (`backend/pyrightconfig.json`) — tests are excluded from pyright
- Dependencies in `backend/pyproject.toml`, lock in `backend/uv.lock`
- PyTorch uses CUDA 12.8 index on Windows/Linux (`tool.uv.sources`)

## Key File Locations

- Backend architecture doc: `backend/architecture.md`
- Default app settings schema: `settings.json`
- Electron builder config: `electron-builder.yml`
- Video editor (largest frontend file): `frontend/views/VideoEditor.tsx`
- Project types: `frontend/types/project.ts`
- IPC API surface: `electron/preload.ts`
- Python backend entry: `backend/ltx2_server.py`
- Build/setup scripts: `scripts/` (platform-specific `.sh` and `.ps1` variants)
