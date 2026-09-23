"""Evidence-grade MP4 persistence for simulator rollout frames.

The core package does not require a video dependency at import time.  The
default backend imports PyAV only when :func:`write_rollout_mp4` is called.  A
caller may inject a backend implementing :class:`RolloutVideoBackend`, which
also makes the persistence and verification contract testable without a local
FFmpeg installation.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import stat
import struct
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

_FRAME_HASH_DOMAIN = b"causal-vla-rollout-rgb24-frame-stream-v1\0"
_BUFFER_SIZE = 1 << 20
_PRIVATE_MODE = 0o600
_X264_PRESETS = frozenset(
    {
        "ultrafast",
        "superfast",
        "veryfast",
        "faster",
        "fast",
        "medium",
        "slow",
        "slower",
        "veryslow",
        "placebo",
    }
)


class RolloutVideoError(RuntimeError):
    """Base exception for rollout-video encoding and verification failures."""


class RolloutVideoDependencyError(RolloutVideoError):
    """Raised when the requested video backend is unavailable."""


class RolloutVideoEncodingError(RolloutVideoError):
    """Raised when the backend cannot produce the requested MP4."""


class RolloutVideoVerificationError(RolloutVideoError):
    """Raised when decoding does not reproduce the declared frame structure."""


@dataclass(frozen=True)
class RolloutVideoEncoding:
    """Frozen encoding parameters for a browser-compatible rollout MP4."""

    fps: int = 20
    codec: str = "libx264"
    pixel_format: str = "yuv420p"
    crf: int = 18
    preset: str = "medium"

    def __post_init__(self) -> None:
        if isinstance(self.fps, bool) or not isinstance(self.fps, int) or not 1 <= self.fps <= 240:
            raise ValueError("Video fps must be an integer in [1, 240]")
        if self.codec != "libx264":
            raise ValueError("Evidence-grade rollout video is frozen to the libx264 codec")
        if self.pixel_format != "yuv420p":
            raise ValueError("Evidence-grade rollout video is frozen to yuv420p")
        if isinstance(self.crf, bool) or not isinstance(self.crf, int) or not 0 <= self.crf <= 51:
            raise ValueError("x264 CRF must be an integer in [0, 51]")
        if self.preset not in _X264_PRESETS:
            raise ValueError("Unknown x264 preset")


class RolloutVideoBackend(Protocol):
    """Minimal codec boundary used by the atomic evidence writer."""

    def identity(self) -> str:
        """Return a stable, human-readable backend identity."""

    def encode(
        self,
        path: Path,
        frames: Sequence[NDArray[np.uint8]],
        encoding: RolloutVideoEncoding,
    ) -> None:
        """Encode all RGB frames into ``path`` and close the container."""

    def decode(self, path: Path) -> tuple[NDArray[np.uint8], ...]:
        """Decode every video frame in presentation order as RGB24."""


class PyAVRolloutVideoBackend:
    """Strict PyAV/libx264 backend with no codec or library fallback."""

    def __init__(self) -> None:
        self._av: Any | None = None

    def _module(self) -> Any:
        if self._av is None:
            try:
                self._av = importlib.import_module("av")
            except (ImportError, ModuleNotFoundError) as error:
                raise RolloutVideoDependencyError(
                    "Writing rollout MP4s requires PyAV with libx264 support; "
                    "install the project video runtime or inject a RolloutVideoBackend"
                ) from error
        return self._av

    def identity(self) -> str:
        av = self._module()
        version = getattr(av, "__version__", "unknown")
        return f"pyav-{version}"

    def encode(
        self,
        path: Path,
        frames: Sequence[NDArray[np.uint8]],
        encoding: RolloutVideoEncoding,
    ) -> None:
        av = self._module()
        height, width, _ = frames[0].shape
        try:
            with av.open(str(path), mode="w", format="mp4") as container:
                stream = container.add_stream(encoding.codec, rate=encoding.fps)
                stream.width = width
                stream.height = height
                stream.pix_fmt = encoding.pixel_format
                stream.options = {
                    "crf": str(encoding.crf),
                    "preset": encoding.preset,
                }
                for frame_index, array in enumerate(frames):
                    frame = av.VideoFrame.from_ndarray(array, format="rgb24")
                    frame.pts = frame_index
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
        except Exception as error:
            raise RolloutVideoEncodingError(
                f"PyAV could not encode a {encoding.codec}/{encoding.pixel_format} MP4"
            ) from error

    def decode(self, path: Path) -> tuple[NDArray[np.uint8], ...]:
        av = self._module()
        decoded: list[NDArray[np.uint8]] = []
        try:
            with av.open(str(path), mode="r") as container:
                video_streams = tuple(container.streams.video)
                if len(video_streams) != 1:
                    raise RolloutVideoVerificationError(
                        "Encoded artifact must contain exactly one video stream"
                    )
                if tuple(container.streams.audio):
                    raise RolloutVideoVerificationError("Encoded artifact must not contain audio")
                for frame in container.decode(video_streams[0]):
                    array = frame.to_ndarray(format="rgb24")
                    decoded.append(
                        cast(NDArray[np.uint8], np.ascontiguousarray(array, dtype=np.uint8))
                    )
        except RolloutVideoVerificationError:
            raise
        except Exception as error:
            raise RolloutVideoVerificationError("PyAV could not decode the encoded MP4") from error
        return tuple(decoded)


@dataclass(frozen=True)
class RolloutVideoEvidence:
    """Hashes and structural facts verified before an MP4 becomes visible."""

    schema_version: int
    artifact_kind: str
    path: str
    backend: str
    container_format: str
    codec: str
    pixel_format: str
    fps: int
    crf: int
    preset: str
    frame_count: int
    decoded_frame_count: int
    frame_shape: tuple[int, int, int]
    raw_frame_stream_sha256: str
    mp4_sha256: str
    mp4_size_bytes: int
    file_mode: str

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.artifact_kind != "rollout_video_evidence":
            raise ValueError("Unknown rollout-video evidence schema")
        if self.container_format != "mp4" or self.file_mode != "0600":
            raise ValueError("Rollout-video container or file mode is invalid")
        if self.frame_count <= 0 or self.decoded_frame_count != self.frame_count:
            raise ValueError("Rollout-video frame counts are invalid")
        if len(self.frame_shape) != 3 or self.frame_shape[2] != 3:
            raise ValueError("Rollout-video evidence must describe HWC RGB frames")
        if self.mp4_size_bytes <= 0:
            raise ValueError("Rollout-video evidence must bind a nonempty MP4")
        for label, digest in (
            ("raw frame stream", self.raw_frame_stream_sha256),
            ("MP4", self.mp4_sha256),
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"Rollout-video {label} digest is not lowercase SHA-256")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible evidence record."""

        payload = cast(dict[str, object], asdict(self))
        payload["frame_shape"] = list(self.frame_shape)
        return payload


@dataclass(frozen=True)
class _FileDigest:
    sha256: str
    size_bytes: int
    device: int
    inode: int
    mode: int


def _materialize_frames(
    frames: Iterable[NDArray[np.uint8]],
) -> tuple[NDArray[np.uint8], ...]:
    materialized: list[NDArray[np.uint8]] = []
    expected_shape: tuple[int, int, int] | None = None
    for index, frame in enumerate(frames):
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"Rollout frame {index} must be a NumPy array")
        if frame.dtype != np.dtype(np.uint8):
            raise ValueError(f"Rollout frame {index} must have dtype uint8")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Rollout frame {index} must have shape (height, width, 3)")
        height, width, channels = cast(tuple[int, int, int], frame.shape)
        if height <= 0 or width <= 0 or height % 2 or width % 2:
            raise ValueError(
                f"Rollout frame {index} must have positive even height and width for yuv420p"
            )
        shape = (height, width, channels)
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError(
                f"Rollout frame {index} shape {shape} differs from the first frame {expected_shape}"
            )
        frozen = np.ascontiguousarray(frame).copy()
        frozen.flags.writeable = False
        materialized.append(frozen)
    if not materialized:
        raise ValueError("At least one rollout frame is required")
    return tuple(materialized)


def _raw_frame_stream_sha256(frames: Sequence[NDArray[np.uint8]]) -> str:
    height, width, channels = frames[0].shape
    digest = hashlib.sha256()
    digest.update(_FRAME_HASH_DOMAIN)
    digest.update(struct.pack(">QQQQ", len(frames), height, width, channels))
    for index, frame in enumerate(frames):
        digest.update(struct.pack(">QQ", index, frame.nbytes))
        digest.update(frame.tobytes(order="C"))
    return digest.hexdigest()


def raw_rgb_frame_stream_sha256(frames: Iterable[NDArray[np.uint8]]) -> str:
    """Validate and hash an ordered RGB24 frame stream independent of MP4 bytes."""

    return _raw_frame_stream_sha256(_materialize_frames(frames))


def _canonical_output_path(value: str | os.PathLike[str]) -> Path:
    output = Path(os.path.abspath(os.fspath(value)))
    if output.suffix.casefold() != ".mp4":
        raise ValueError("Rollout video output must use the .mp4 suffix")
    parent = output.parent
    try:
        parent_stat = parent.lstat()
    except OSError as error:
        raise ValueError("Rollout video parent directory must already exist") from error
    if stat.S_ISLNK(parent_stat.st_mode) or not stat.S_ISDIR(parent_stat.st_mode):
        raise ValueError("Rollout video parent must be a nonsymlink directory")
    if parent.resolve(strict=True) != parent:
        raise ValueError("Rollout video parent path must be canonical and contain no symlink")
    if os.path.lexists(output):
        raise FileExistsError(f"Refusing to overwrite rollout video {output}")
    return output


def _read_private_regular_file(
    path: Path,
    *,
    expected_identity: tuple[int, int] | None = None,
    set_mode: bool = False,
) -> _FileDigest:
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise RolloutVideoError("Rollout video must be one singly linked regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        identity = (opened.st_dev, opened.st_ino)
        if expected_identity is not None and identity != expected_identity:
            raise RolloutVideoError("Rollout video inode changed during publication")
        if identity != (before.st_dev, before.st_ino) or opened.st_size != before.st_size:
            raise RolloutVideoError("Rollout video changed while it was opened")
        if opened.st_size <= 0:
            raise RolloutVideoEncodingError("Video backend produced an empty MP4")
        if set_mode:
            os.fchmod(descriptor, _PRIVATE_MODE)
            os.fsync(descriptor)
            opened = os.fstat(descriptor)
        if stat.S_IMODE(opened.st_mode) != _PRIVATE_MODE:
            raise RolloutVideoError("Rollout video mode must be 0600")
        digest = hashlib.sha256()
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(_BUFFER_SIZE, remaining))
            if not chunk:
                raise RolloutVideoError("Rollout video ended before its declared size")
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RolloutVideoError("Rollout video grew while it was hashed")
        after_read = os.fstat(descriptor)
        if (after_read.st_dev, after_read.st_ino, after_read.st_size) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
        ) or after_read.st_nlink != 1:
            raise RolloutVideoError("Rollout video changed while it was hashed")
        result = _FileDigest(
            sha256=digest.hexdigest(),
            size_bytes=opened.st_size,
            device=opened.st_dev,
            inode=opened.st_ino,
            mode=stat.S_IMODE(opened.st_mode),
        )
    finally:
        os.close(descriptor)
    after = path.lstat()
    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) and not set_mode:
        raise RolloutVideoError("Rollout video changed after it was hashed")
    if (after.st_dev, after.st_ino, after.st_size) != (
        result.device,
        result.inode,
        result.size_bytes,
    ):
        raise RolloutVideoError("Rollout video path changed after it was hashed")
    return result


def _verify_decoded_frames(
    decoded: Sequence[NDArray[np.uint8]],
    *,
    expected_count: int,
    expected_shape: tuple[int, int, int],
) -> None:
    if len(decoded) != expected_count:
        raise RolloutVideoVerificationError(
            f"Decoded {len(decoded)} frames, expected {expected_count}"
        )
    for index, frame in enumerate(decoded):
        if not isinstance(frame, np.ndarray):
            raise RolloutVideoVerificationError(f"Decoded frame {index} is not a NumPy array")
        if frame.dtype != np.dtype(np.uint8) or tuple(frame.shape) != expected_shape:
            raise RolloutVideoVerificationError(
                f"Decoded frame {index} has dtype/shape {frame.dtype}/{tuple(frame.shape)}, "
                f"expected uint8/{expected_shape}"
            )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_owned_path(path: Path, identity: tuple[int, int]) -> None:
    if not os.path.lexists(path):
        return
    observed = path.lstat()
    if (observed.st_dev, observed.st_ino) == identity and stat.S_ISREG(observed.st_mode):
        path.unlink()


def write_rollout_mp4(
    frames: Iterable[NDArray[np.uint8]],
    output_path: str | os.PathLike[str],
    *,
    encoding: RolloutVideoEncoding | None = None,
    backend: RolloutVideoBackend | None = None,
) -> RolloutVideoEvidence:
    """Validate, encode, decode-check, and atomically publish one fresh MP4.

    The destination is created only after the complete private temporary file
    has decoded to the expected frame count and shape.  Publication uses an
    exclusive hard link, so an existing path is never replaced.  Any failure
    removes only files created by this call and is raised to the caller.
    """

    frame_sequence = _materialize_frames(frames)
    frame_shape = cast(tuple[int, int, int], tuple(frame_sequence[0].shape))
    raw_digest = _raw_frame_stream_sha256(frame_sequence)
    output = _canonical_output_path(output_path)
    video_encoding = RolloutVideoEncoding() if encoding is None else encoding
    codec_backend: RolloutVideoBackend = PyAVRolloutVideoBackend() if backend is None else backend
    backend_identity = codec_backend.identity()
    if not backend_identity:
        raise ValueError("Rollout video backend identity must be nonempty")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".partial.mp4", dir=output.parent
    )
    temporary = Path(temporary_name)
    os.fchmod(descriptor, _PRIVATE_MODE)
    reserved = os.fstat(descriptor)
    reserved_identity = (reserved.st_dev, reserved.st_ino)
    published = False
    try:
        codec_backend.encode(temporary, frame_sequence, video_encoding)
        if _raw_frame_stream_sha256(frame_sequence) != raw_digest:
            raise RolloutVideoError("Video backend mutated the frozen source frames")
        encoded = _read_private_regular_file(
            temporary,
            expected_identity=reserved_identity,
            set_mode=True,
        )
        decoded = codec_backend.decode(temporary)
        _verify_decoded_frames(
            decoded,
            expected_count=len(frame_sequence),
            expected_shape=frame_shape,
        )
        after_decode = _read_private_regular_file(
            temporary,
            expected_identity=reserved_identity,
        )
        if after_decode != encoded:
            raise RolloutVideoError("Encoded MP4 changed while it was decoded and verified")
        if _raw_frame_stream_sha256(frame_sequence) != raw_digest:
            raise RolloutVideoError("Video verification mutated the frozen source frames")

        os.link(temporary, output, follow_symlinks=False)
        published = True
        temporary.unlink()
        _fsync_directory(output.parent)
        final = _read_private_regular_file(output, expected_identity=reserved_identity)
        if final != encoded:
            raise RolloutVideoError("Published MP4 differs from the verified private artifact")
        return RolloutVideoEvidence(
            schema_version=1,
            artifact_kind="rollout_video_evidence",
            path=str(output),
            backend=backend_identity,
            container_format="mp4",
            codec=video_encoding.codec,
            pixel_format=video_encoding.pixel_format,
            fps=video_encoding.fps,
            crf=video_encoding.crf,
            preset=video_encoding.preset,
            frame_count=len(frame_sequence),
            decoded_frame_count=len(decoded),
            frame_shape=frame_shape,
            raw_frame_stream_sha256=raw_digest,
            mp4_sha256=final.sha256,
            mp4_size_bytes=final.size_bytes,
            file_mode="0600",
        )
    except BaseException:
        _remove_owned_path(output, reserved_identity)
        _remove_owned_path(temporary, reserved_identity)
        if published:
            _fsync_directory(output.parent)
        raise
    finally:
        # Keep the reserved inode alive for the complete transaction.  If a
        # backend unlinks and recreates the temporary path, POSIX cannot reuse
        # this inode while the descriptor remains open, so the identity check
        # above is reliable on both Linux and macOS.
        os.close(descriptor)


__all__ = [
    "PyAVRolloutVideoBackend",
    "RolloutVideoBackend",
    "RolloutVideoDependencyError",
    "RolloutVideoEncoding",
    "RolloutVideoEncodingError",
    "RolloutVideoError",
    "RolloutVideoEvidence",
    "RolloutVideoVerificationError",
    "raw_rgb_frame_stream_sha256",
    "write_rollout_mp4",
]
