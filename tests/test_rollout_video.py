from __future__ import annotations

import hashlib
import importlib
import os
import stat
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

import causal_vla.rollout_video as rollout_video
from causal_vla.rollout_video import (
    RolloutVideoDependencyError,
    RolloutVideoEncoding,
    RolloutVideoVerificationError,
    raw_rgb_frame_stream_sha256,
    write_rollout_mp4,
)


def _frames(count: int = 3, shape: tuple[int, int, int] = (8, 10, 3)) -> list[NDArray[np.uint8]]:
    return [np.full(shape, index * 17, dtype=np.uint8) for index in range(count)]


class _NpzVideoBackend:
    def __init__(self) -> None:
        self.encode_calls = 0
        self.decode_calls = 0

    def identity(self) -> str:
        return "test-npz-video-v1"

    def encode(
        self,
        path: Path,
        frames: Sequence[NDArray[np.uint8]],
        encoding: RolloutVideoEncoding,
    ) -> None:
        self.encode_calls += 1
        with path.open("wb") as handle:
            np.savez(
                handle,
                frames=np.stack(frames),
                fps=np.asarray([encoding.fps], dtype=np.int64),
            )

    def decode(self, path: Path) -> tuple[NDArray[np.uint8], ...]:
        self.decode_calls += 1
        with np.load(path, allow_pickle=False) as archive:
            array = np.asarray(archive["frames"], dtype=np.uint8)
        return tuple(np.ascontiguousarray(frame) for frame in array)


class _FailingEncodeBackend(_NpzVideoBackend):
    def encode(
        self,
        path: Path,
        frames: Sequence[NDArray[np.uint8]],
        encoding: RolloutVideoEncoding,
    ) -> None:
        path.write_bytes(b"incomplete")
        raise RuntimeError("synthetic encoder failure")


class _WrongCountBackend(_NpzVideoBackend):
    def decode(self, path: Path) -> tuple[NDArray[np.uint8], ...]:
        decoded = super().decode(path)
        return decoded[:-1]


class _WrongShapeBackend(_NpzVideoBackend):
    def decode(self, path: Path) -> tuple[NDArray[np.uint8], ...]:
        decoded = list(super().decode(path))
        decoded[-1] = np.zeros((6, 10, 3), dtype=np.uint8)
        return tuple(decoded)


def test_writer_atomically_publishes_verified_private_video(tmp_path: Path) -> None:
    frames = _frames()
    backend = _NpzVideoBackend()
    output = tmp_path / "rollout.mp4"

    evidence = write_rollout_mp4(frames, output, backend=backend)

    assert backend.encode_calls == backend.decode_calls == 1
    assert output.is_file() and not output.is_symlink()
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert evidence.path == str(output)
    assert evidence.backend == "test-npz-video-v1"
    assert evidence.frame_count == evidence.decoded_frame_count == 3
    assert evidence.frame_shape == (8, 10, 3)
    assert evidence.raw_frame_stream_sha256 == raw_rgb_frame_stream_sha256(frames)
    assert evidence.mp4_sha256 == hashlib.sha256(output.read_bytes()).hexdigest()
    assert evidence.mp4_size_bytes == output.stat().st_size
    assert evidence.to_dict()["frame_shape"] == [8, 10, 3]
    assert not tuple(tmp_path.glob(".*.partial.mp4"))


@pytest.mark.parametrize(
    ("frames", "error", "message"),
    [
        ([], ValueError, "At least one"),
        ([np.zeros((8, 10, 3), dtype=np.float32)], ValueError, "dtype uint8"),
        ([np.zeros((8, 10), dtype=np.uint8)], ValueError, "shape"),
        ([np.zeros((8, 10, 4), dtype=np.uint8)], ValueError, "shape"),
        ([np.zeros((7, 10, 3), dtype=np.uint8)], ValueError, "positive even"),
        ([np.zeros((8, 9, 3), dtype=np.uint8)], ValueError, "positive even"),
        (
            [
                np.zeros((8, 10, 3), dtype=np.uint8),
                np.zeros((8, 12, 3), dtype=np.uint8),
            ],
            ValueError,
            "differs from the first frame",
        ),
        (["not-an-array"], TypeError, "NumPy array"),
    ],
)
def test_writer_rejects_invalid_frame_streams(
    tmp_path: Path,
    frames: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        write_rollout_mp4(frames, tmp_path / "invalid.mp4", backend=_NpzVideoBackend())  # type: ignore[arg-type]
    assert not (tmp_path / "invalid.mp4").exists()


def test_writer_refuses_to_overwrite_any_existing_destination(tmp_path: Path) -> None:
    output = tmp_path / "existing.mp4"
    output.write_bytes(b"user-owned")
    backend = _NpzVideoBackend()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_rollout_mp4(_frames(), output, backend=backend)

    assert output.read_bytes() == b"user-owned"
    assert backend.encode_calls == backend.decode_calls == 0


@pytest.mark.parametrize("backend", [_WrongCountBackend(), _WrongShapeBackend()])
def test_decode_verification_failure_publishes_nothing(
    tmp_path: Path,
    backend: _NpzVideoBackend,
) -> None:
    output = tmp_path / "unverified.mp4"

    with pytest.raises(RolloutVideoVerificationError):
        write_rollout_mp4(_frames(), output, backend=backend)

    assert not output.exists()
    assert not tuple(tmp_path.glob(".*.partial.mp4"))


def test_encoder_failure_removes_only_its_private_temporary_file(tmp_path: Path) -> None:
    output = tmp_path / "failed.mp4"
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("owned by caller", encoding="utf-8")

    with pytest.raises(RuntimeError, match="synthetic encoder failure"):
        write_rollout_mp4(_frames(), output, backend=_FailingEncodeBackend())

    assert not output.exists()
    assert sentinel.read_text(encoding="utf-8") == "owned by caller"
    assert not tuple(tmp_path.glob(".*.partial.mp4"))


def test_default_backend_imports_pyav_only_at_write_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = importlib.import_module

    def unavailable(name: str, package: str | None = None) -> object:
        if name == "av":
            raise ModuleNotFoundError("synthetic missing PyAV")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", unavailable)
    output = tmp_path / "missing-dependency.mp4"

    with pytest.raises(RolloutVideoDependencyError, match="requires PyAV"):
        write_rollout_mp4(_frames(), output)

    assert not output.exists()
    assert not tuple(tmp_path.glob(".*.partial.mp4"))


def test_raw_frame_hash_binds_order_shape_and_content() -> None:
    frames = _frames(2)
    digest = raw_rgb_frame_stream_sha256(frames)

    assert digest == raw_rgb_frame_stream_sha256([frame.copy() for frame in frames])
    assert digest != raw_rgb_frame_stream_sha256(tuple(reversed(frames)))
    changed = [frame.copy() for frame in frames]
    changed[1][0, 0, 0] ^= np.uint8(1)
    assert digest != raw_rgb_frame_stream_sha256(changed)


def test_writer_rejects_noncanonical_symlink_parent(tmp_path: Path) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="nonsymlink directory"):
        write_rollout_mp4(_frames(), linked_parent / "rollout.mp4", backend=_NpzVideoBackend())

    assert not (real_parent / "rollout.mp4").exists()


def test_encoding_contract_rejects_unfrozen_codec_parameters() -> None:
    with pytest.raises(ValueError, match="libx264"):
        RolloutVideoEncoding(codec="mpeg4")
    with pytest.raises(ValueError, match="yuv420p"):
        RolloutVideoEncoding(pixel_format="rgb24")
    with pytest.raises(ValueError, match="fps"):
        RolloutVideoEncoding(fps=0)
    with pytest.raises(ValueError, match="CRF"):
        RolloutVideoEncoding(crf=52)


def test_output_requires_mp4_suffix_and_existing_parent(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"\.mp4 suffix"):
        write_rollout_mp4(_frames(), tmp_path / "rollout.avi", backend=_NpzVideoBackend())
    with pytest.raises(ValueError, match="must already exist"):
        write_rollout_mp4(
            _frames(), tmp_path / "missing" / "rollout.mp4", backend=_NpzVideoBackend()
        )


def test_backend_cannot_replace_reserved_temporary_inode(tmp_path: Path) -> None:
    class ReplacingBackend(_NpzVideoBackend):
        def encode(
            self,
            path: Path,
            frames: Sequence[NDArray[np.uint8]],
            encoding: RolloutVideoEncoding,
        ) -> None:
            path.unlink()
            super().encode(path, frames, encoding)

    output = tmp_path / "replaced.mp4"
    with pytest.raises(rollout_video.RolloutVideoError, match="inode changed"):
        write_rollout_mp4(_frames(), output, backend=ReplacingBackend())
    assert not output.exists()


def test_source_frames_are_copied_before_the_backend_sees_them(tmp_path: Path) -> None:
    class InspectingBackend(_NpzVideoBackend):
        def encode(
            self,
            path: Path,
            frames: Sequence[NDArray[np.uint8]],
            encoding: RolloutVideoEncoding,
        ) -> None:
            assert all(frame.flags.c_contiguous and not frame.flags.writeable for frame in frames)
            super().encode(path, frames, encoding)

    source = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)[:, ::-1]
    before = source.copy()
    write_rollout_mp4([source], tmp_path / "copied.mp4", backend=InspectingBackend())
    assert np.array_equal(source, before)


def test_successful_writer_leaves_no_open_or_linked_staging_inode(tmp_path: Path) -> None:
    output = tmp_path / "complete.mp4"
    evidence = write_rollout_mp4(_frames(), output, backend=_NpzVideoBackend())

    assert output.stat().st_nlink == 1
    assert evidence.file_mode == "0600"
    assert not any(path.name.startswith(f".{output.name}.") for path in tmp_path.iterdir())
    assert os.path.samefile(output, Path(evidence.path))
