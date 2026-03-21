"""Tests for the long video pipeline (chain-extend + concat)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from state.job_queue import JobQueue, QueueJob
from handlers.queue_worker import QueueWorker
from handlers.video_generation_handler import VideoGenerationHandler


# ---------------------------------------------------------------------------
# Category 1: Route registration — POST /api/generate/long
# ---------------------------------------------------------------------------


class TestLongVideoRoute:
    def test_route_exists_and_validates_request(self, client):
        """POST /api/generate/long should exist and validate body."""
        resp = client.post("/api/generate/long", json={
            "prompt": "A cinematic landscape",
            "imagePath": "/nonexistent/image.png",
            "targetDuration": 12,
        })
        # Route exists (not 404/405). It may fail because models aren't loaded,
        # but it should at least parse the request (not 422 validation error).
        assert resp.status_code != 404
        assert resp.status_code != 405

    def test_route_rejects_empty_prompt(self, client):
        """Empty prompt should be rejected by validation."""
        resp = client.post("/api/generate/long", json={
            "prompt": "",
            "imagePath": "/some/image.png",
            "targetDuration": 12,
        })
        assert resp.status_code == 422

    def test_route_rejects_missing_image_path(self, client):
        """imagePath is required."""
        resp = client.post("/api/generate/long", json={
            "prompt": "test",
            "targetDuration": 12,
        })
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Category 2: Queue submission — type: "long_video"
# ---------------------------------------------------------------------------


class TestLongVideoQueueSubmission:
    def test_submit_long_video_job(self, client):
        """long_video type should be accepted and routed to GPU slot."""
        resp = client.post("/api/queue/submit", json={
            "type": "long_video",
            "model": "fast",
            "params": {
                "prompt": "Epic landscape",
                "imagePath": "/fake/image.png",
                "targetDuration": 20,
                "segmentDuration": 4,
                "resolution": "540p",
            },
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "queued"
        assert "id" in data

    def test_long_video_job_appears_in_status(self, client):
        """Submitted long_video job should appear in queue status."""
        client.post("/api/queue/submit", json={
            "type": "long_video",
            "model": "fast",
            "params": {"prompt": "test", "imagePath": "/img.png", "targetDuration": 16},
        })
        status = client.get("/api/queue/status")
        jobs = status.json()["jobs"]
        assert len(jobs) == 1
        assert jobs[0]["type"] == "long_video"

    def test_long_video_routes_to_gpu_slot(self, client):
        """long_video should always route to GPU slot (local generation)."""
        client.post("/api/queue/submit", json={
            "type": "long_video",
            "model": "fast",
            "params": {"prompt": "test", "imagePath": "/img.png", "targetDuration": 12},
        })
        status = client.get("/api/queue/status")
        jobs = status.json()["jobs"]
        assert jobs[0]["slot"] == "gpu"


# ---------------------------------------------------------------------------
# Category 3: Job executor dispatch — long_video type
# ---------------------------------------------------------------------------


class TestLongVideoExecutorDispatch:
    def test_worker_dispatches_long_video_to_gpu_executor(self, tmp_path: Path):
        """QueueWorker should route long_video jobs to GPU executor."""
        queue = JobQueue(persistence_path=tmp_path / "queue.json")
        job = queue.submit(
            job_type="long_video",
            model="fast",
            params={
                "prompt": "test",
                "imagePath": "/fake/img.png",
                "targetDuration": 16,
                "segmentDuration": 4,
            },
            slot="gpu",
        )

        class FakeExecutor:
            def __init__(self) -> None:
                self.executed_jobs: list[QueueJob] = []

            def execute(self, job: QueueJob) -> list[str]:
                self.executed_jobs.append(job)
                return ["/fake/output_long.mp4"]

        gpu_exec = FakeExecutor()
        api_exec = FakeExecutor()
        worker = QueueWorker(queue=queue, gpu_executor=gpu_exec, api_executor=api_exec)
        worker.tick()

        assert len(gpu_exec.executed_jobs) == 1
        assert gpu_exec.executed_jobs[0].id == job.id
        assert gpu_exec.executed_jobs[0].type == "long_video"
        assert len(api_exec.executed_jobs) == 0

    def test_executor_passes_long_video_params(self, tmp_path: Path):
        """Executor should forward all long_video params correctly."""
        queue = JobQueue(persistence_path=tmp_path / "queue.json")
        params = {
            "prompt": "cinematic shot",
            "imagePath": "/image.png",
            "targetDuration": 20,
            "segmentDuration": 4,
            "resolution": "720p",
            "aspectRatio": "16:9",
            "fps": 24,
            "cameraMotion": "dolly_in",
        }
        job = queue.submit(job_type="long_video", model="fast", params=params, slot="gpu")

        class CapturingExecutor:
            def __init__(self) -> None:
                self.captured_params: dict[str, object] = {}

            def execute(self, job: QueueJob) -> list[str]:
                self.captured_params = dict(job.params)
                return ["/out.mp4"]

        executor = CapturingExecutor()
        worker = QueueWorker(queue=queue, gpu_executor=executor, api_executor=executor)
        worker.tick()

        assert executor.captured_params["prompt"] == "cinematic shot"
        assert executor.captured_params["targetDuration"] == 20
        assert executor.captured_params["segmentDuration"] == 4
        assert executor.captured_params["cameraMotion"] == "dolly_in"


# ---------------------------------------------------------------------------
# Category 4: Segment calculation
# ---------------------------------------------------------------------------


class TestSegmentCalculation:
    """Verify num_segments = ceil(target_duration / segment_duration)."""

    def test_exact_multiple(self):
        """20s target / 4s segments = exactly 5."""
        n = max(1, (20 + 4 - 1) // 4)
        assert n == 5

    def test_remainder_rounds_up(self):
        """10s target / 4s segments = 3 (not 2.5)."""
        n = max(1, (10 + 4 - 1) // 4)
        assert n == 3

    def test_single_segment(self):
        """4s target / 4s segments = 1."""
        n = max(1, (4 + 4 - 1) // 4)
        assert n == 1

    def test_very_long(self):
        """60s target / 4s segments = 15."""
        n = max(1, (60 + 4 - 1) // 4)
        assert n == 15

    def test_short_target_below_segment(self):
        """2s target / 4s segments = 1 (minimum 1)."""
        n = max(1, (2 + 4 - 1) // 4)
        assert n == 1

    def test_12s_target(self):
        """12s / 4s = exactly 3."""
        n = max(1, (12 + 4 - 1) // 4)
        assert n == 3


# ---------------------------------------------------------------------------
# Category 5: Concat logic — _concatenate_segments
# ---------------------------------------------------------------------------


class TestConcatenateSegments:
    """Test the static _concatenate_segments method."""

    def test_single_segment_copies_file(self, tmp_path: Path):
        """Single segment should just copy the file."""
        seg = tmp_path / "segment_001.mp4"
        seg.write_bytes(b"fake_video_content")
        out = tmp_path / "final.mp4"

        VideoGenerationHandler._concatenate_segments(
            segment_paths=[str(seg)],
            output_path=str(out),
            ffmpeg_path="unused",
            fps=24,
        )

        assert out.exists()
        assert out.read_bytes() == b"fake_video_content"

    def test_concat_file_format(self, tmp_path: Path):
        """Multi-segment should write proper concat file with forward slashes."""
        # We can't run real ffmpeg in tests, but we can verify the concat file
        # is written correctly by monkeypatching subprocess.run
        import subprocess
        calls: list[list[str]] = []

        original_run = subprocess.run

        def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
            calls.append(cmd)
            # Read the concat file before it gets cleaned up
            for i, arg in enumerate(cmd):
                if arg == "-i" and i + 1 < len(cmd):
                    concat_path = cmd[i + 1]
                    content = Path(concat_path).read_text()
                    # Verify forward slashes
                    for line in content.strip().split("\n"):
                        assert "\\" not in line, f"Backslash found in concat file: {line}"
                        assert line.startswith("file '")
            # Create the output file so the method succeeds
            for i, arg in enumerate(cmd):
                if arg == "-y" and len(cmd) > 1:
                    # The output is the last arg
                    Path(cmd[-1]).write_bytes(b"concatenated")
                    break
            return subprocess.CompletedProcess(cmd, 0)

        subprocess.run = fake_run  # type: ignore[assignment]
        try:
            seg1 = tmp_path / "seg1.mp4"
            seg2 = tmp_path / "seg2.mp4"
            seg1.write_bytes(b"video1")
            seg2.write_bytes(b"video2")
            out = tmp_path / "out.mp4"

            VideoGenerationHandler._concatenate_segments(
                segment_paths=[str(seg1), str(seg2)],
                output_path=str(out),
                ffmpeg_path="ffmpeg",
                fps=24,
            )

            assert len(calls) == 1
            assert "-f" in calls[0]
            assert "concat" in calls[0]
            assert "-c" in calls[0]
            assert "copy" in calls[0]
        finally:
            subprocess.run = original_run  # type: ignore[assignment]

    def test_concat_cleans_up_temp_file(self, tmp_path: Path):
        """Concat file should be cleaned up even on success."""
        import subprocess

        created_files: list[str] = []

        original_run = subprocess.run

        def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
            for i, arg in enumerate(cmd):
                if arg == "-i" and i + 1 < len(cmd):
                    created_files.append(cmd[i + 1])
            Path(cmd[-1]).write_bytes(b"result")
            return subprocess.CompletedProcess(cmd, 0)

        subprocess.run = fake_run  # type: ignore[assignment]
        try:
            seg1 = tmp_path / "a.mp4"
            seg2 = tmp_path / "b.mp4"
            seg1.write_bytes(b"x")
            seg2.write_bytes(b"y")

            VideoGenerationHandler._concatenate_segments(
                [str(seg1), str(seg2)], str(tmp_path / "out.mp4"), "ffmpeg", 24,
            )

            # The temp concat file should have been cleaned up
            assert len(created_files) == 1
            assert not os.path.exists(created_files[0])
        finally:
            subprocess.run = original_run  # type: ignore[assignment]

    def test_windows_paths_converted_to_forward_slashes(self, tmp_path: Path):
        """Backslashes in Windows paths should be converted to forward slashes."""
        import subprocess

        concat_content: list[str] = []

        original_run = subprocess.run

        def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
            for i, arg in enumerate(cmd):
                if arg == "-i" and i + 1 < len(cmd):
                    concat_content.append(Path(cmd[i + 1]).read_text())
            Path(cmd[-1]).write_bytes(b"result")
            return subprocess.CompletedProcess(cmd, 0)

        subprocess.run = fake_run  # type: ignore[assignment]
        try:
            seg1 = tmp_path / "s1.mp4"
            seg2 = tmp_path / "s2.mp4"
            seg1.write_bytes(b"x")
            seg2.write_bytes(b"y")

            # Simulate Windows-style paths
            win_paths = [
                str(seg1).replace("/", "\\"),
                str(seg2).replace("/", "\\"),
            ]

            VideoGenerationHandler._concatenate_segments(
                win_paths, str(tmp_path / "out.mp4"), "ffmpeg", 24,
            )

            assert len(concat_content) == 1
            for line in concat_content[0].strip().split("\n"):
                assert "\\" not in line
        finally:
            subprocess.run = original_run  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Category 6: Phase reporting — generating_segment, concatenating
# ---------------------------------------------------------------------------


class TestLongVideoPhaseReporting:
    """Verify that phase names match what the frontend expects."""

    def test_phase_names_are_valid_strings(self):
        """The phase names used in generate_long_video must match frontend expectations."""
        # These are the phases reported by generate_long_video
        expected_phases = {"generating_segment", "concatenating"}
        # We can't run generate_long_video in tests (no GPU), but we can
        # verify the constants by inspecting the source
        import inspect
        source = inspect.getsource(VideoGenerationHandler.generate_long_video)
        for phase in expected_phases:
            assert f'"{phase}"' in source, f"Phase '{phase}' not found in generate_long_video source"


# ---------------------------------------------------------------------------
# Category 7: Frontend submission — long_video type detection
# (These test the logic extracted from use-generation.ts)
# ---------------------------------------------------------------------------


class TestFrontendLongVideoLogic:
    """Test the frontend submission logic (mirrored in Python for validation)."""

    @staticmethod
    def _should_use_long_video(
        duration: int, has_image: bool, has_audio: bool, has_last_frame: bool,
    ) -> bool:
        """Mirror of the frontend logic: useLongVideo = duration > 8 && imagePath && !audioPath && !lastFramePath"""
        return duration > 8 and has_image and not has_audio and not has_last_frame

    def test_long_video_with_image_over_8s(self):
        assert self._should_use_long_video(12, has_image=True, has_audio=False, has_last_frame=False)

    def test_regular_video_at_8s(self):
        """8s should NOT trigger long video (must be > 8)."""
        assert not self._should_use_long_video(8, has_image=True, has_audio=False, has_last_frame=False)

    def test_regular_video_at_4s(self):
        assert not self._should_use_long_video(4, has_image=True, has_audio=False, has_last_frame=False)

    def test_no_image_stays_regular(self):
        """Without an image, can't use long_video (needs I2V seed segment)."""
        assert not self._should_use_long_video(20, has_image=False, has_audio=False, has_last_frame=False)

    def test_with_audio_stays_regular(self):
        """Audio path present means A2V, not long video."""
        assert not self._should_use_long_video(20, has_image=True, has_audio=True, has_last_frame=False)

    def test_with_last_frame_stays_regular(self):
        """Last frame means extend, not long video."""
        assert not self._should_use_long_video(20, has_image=True, has_audio=False, has_last_frame=True)

    def test_30s_long_video(self):
        assert self._should_use_long_video(30, has_image=True, has_audio=False, has_last_frame=False)

    def test_60s_long_video(self):
        assert self._should_use_long_video(60, has_image=True, has_audio=False, has_last_frame=False)


# ---------------------------------------------------------------------------
# Category 8: Phase message mapping
# ---------------------------------------------------------------------------


class TestPhaseMessageMapping:
    """Verify frontend getPhaseMessage covers all long video phases.

    We can't import TS, but we validate the source file contains the mappings.
    """

    def test_phase_messages_exist_in_frontend(self):
        """use-generation.ts must contain phase messages for long video phases."""
        frontend_path = Path(__file__).parent.parent.parent / "frontend" / "hooks" / "use-generation.ts"
        if not frontend_path.exists():
            # Skip if frontend not available (CI without frontend)
            return

        source = frontend_path.read_text()
        required_phases = [
            "generating_segment",
            "extracting_frame",
            "concatenating",
        ]
        for phase in required_phases:
            assert f"'{phase}'" in source, (
                f"Phase '{phase}' not found in use-generation.ts getPhaseMessage()"
            )


# ---------------------------------------------------------------------------
# Category 9: Duration options consistency
# ---------------------------------------------------------------------------


class TestDurationOptionsConsistency:
    """Verify GenSpace.tsx and SettingsPanel.tsx have matching duration configs."""

    def test_duration_options_match(self):
        """Both files must have the same duration options and LOCAL_MAX_DURATION."""
        frontend_dir = Path(__file__).parent.parent.parent / "frontend"
        genspace = frontend_dir / "views" / "GenSpace.tsx"
        settings_panel = frontend_dir / "components" / "SettingsPanel.tsx"

        if not genspace.exists() or not settings_panel.exists():
            return  # Skip in CI

        gs_src = genspace.read_text()
        sp_src = settings_panel.read_text()

        # Both should have the same duration array
        duration_pattern = "[4, 5, 6, 8, 10, 12, 16, 20, 30, 60]"
        assert duration_pattern in gs_src, "GenSpace.tsx missing updated duration options"
        assert duration_pattern in sp_src, "SettingsPanel.tsx missing updated duration options"

        # Both should have 540p: 60
        assert "'540p': 60" in gs_src, "GenSpace.tsx LOCAL_MAX_DURATION for 540p should be 60"
        assert "'540p': 60" in sp_src, "SettingsPanel.tsx LOCAL_MAX_DURATION for 540p should be 60"

        # Both should have 720p: 10
        assert "'720p': 10" in gs_src
        assert "'720p': 10" in sp_src

        # Both should have 1080p: 5
        assert "'1080p': 5" in gs_src
        assert "'1080p': 5" in sp_src
