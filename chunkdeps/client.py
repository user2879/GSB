"""High-level local client for deterministic chunk uploads."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from .errors import E_CHUNK, BusinessError
from .protocol import ProtocolClient
from .util import b64encode, chunk_count, read_file_chunk, validate_chunk_size, validate_total_size


class WorkbenchClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765, timeout: float = 60.0) -> None:
        self.protocol = ProtocolClient(host, port, timeout=timeout)
        self.host = host
        self.port = port

    def request(self, op: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.protocol.request(op, params)

    def ping(self) -> dict[str, Any]:
        return self.request("ping")

    def create_session(
        self,
        *,
        object_key: str,
        source: str | Path,
        chunk_size: int,
        tags: dict[str, str] | None = None,
        digest_algorithm: str = "sha256",
        expected_base_version: int | None = None,
    ) -> dict[str, Any]:
        path = Path(source)
        if not path.is_file():
            raise BusinessError(E_CHUNK, "source file does not exist", {"path": str(path)})
        validate_chunk_size(chunk_size)
        total_size = validate_total_size(path.stat().st_size)
        return self.request(
            "create_session",
            {
                "object_key": object_key,
                "chunk_size": chunk_size,
                "total_size": total_size,
                "tags": tags or {},
                "digest_algorithm": digest_algorithm,
                "expected_base_version": expected_base_version,
            },
        )

    def upload_session(
        self,
        *,
        session_id: str,
        source: str | Path,
        concurrency: int = 4,
        reupload_indices: list[int] | None = None,
        inject: str | None = None,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        if concurrency <= 0:
            raise BusinessError(E_CHUNK, "concurrency must be a positive integer")
        resumed = self.request("resume_session", {"session_id": session_id})
        chunk_size = int(resumed["chunk_size"])
        total_size = int(resumed["total_size"])
        algorithm = str(resumed["digest_algorithm"])
        path = Path(source)
        if not path.is_file():
            raise BusinessError(E_CHUNK, "source file does not exist", {"path": str(path)})
        if path.stat().st_size != total_size:
            raise BusinessError(
                E_CHUNK,
                "source file size differs from the session declaration",
                {"path": str(path), "file_size": path.stat().st_size, "session_size": total_size},
            )
        if resumed["state"] == "committed":
            raise BusinessError(
                E_CHUNK,
                "committed session cannot be uploaded again",
                {"session_id": session_id},
            )
        expected_count = chunk_count(total_size, chunk_size)
        if expected_count != int(resumed["chunk_count"]):
            raise BusinessError(E_CHUNK, "local split shape differs from server session")

        if reupload_indices is not None:
            indices = sorted(set(int(index) for index in reupload_indices))
            for index in indices:
                if index < 0 or index >= expected_count:
                    raise BusinessError(
                        E_CHUNK,
                        "reupload chunk index is outside the session range",
                        {"chunk_index": index, "chunk_count": expected_count},
                    )
            skipped = []
        else:
            indices = list(resumed["missing_indices"])
            skipped = list(resumed["skipped_indices"])

        results: list[dict[str, Any]] = []
        errors: list[tuple[int, BusinessError]] = []

        def upload_one(index: int) -> dict[str, Any]:
            payload, digest = read_file_chunk(
                path, index, chunk_size, total_size, algorithm
            )
            params: dict[str, Any] = {
                "session_id": session_id,
                "chunk_index": index,
                "data_b64": b64encode(payload),
                "digest": digest,
            }
            if inject:
                params["inject"] = inject
            result = self.request("upload_chunk", params)
            result["chunk_index"] = index
            if progress is not None:
                progress(result)
            return result

        if indices:
            with ThreadPoolExecutor(max_workers=min(concurrency, len(indices))) as pool:
                future_map = {pool.submit(upload_one, index): index for index in indices}
                for future in as_completed(future_map):
                    index = future_map[future]
                    try:
                        results.append(future.result())
                    except BusinessError as exc:
                        errors.append((index, exc))
            if errors:
                errors.sort(key=lambda item: item[0])
                first_index, first_error = errors[0]
                first_error.details.setdefault("chunk_index", first_index)
                first_error.details.setdefault("failed_chunks", [index for index, _ in errors])
                raise first_error
        results.sort(key=lambda item: item["chunk_index"])
        final_session = self.request("get_session", {"session_id": session_id})
        return {
            "session_id": session_id,
            "uploaded": results,
            "uploaded_indices": [item["chunk_index"] for item in results],
            "skipped_indices": skipped,
            "session": final_session,
            "schedule": [item.get("schedule") for item in results],
        }

    def upload_file(
        self,
        *,
        object_key: str,
        source: str | Path,
        chunk_size: int,
        tags: dict[str, str] | None = None,
        digest_algorithm: str = "sha256",
        concurrency: int = 4,
        expected_base_version: int | None = None,
        commit: bool = True,
        inject: str | None = None,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        created = self.create_session(
            object_key=object_key,
            source=source,
            chunk_size=chunk_size,
            tags=tags,
            digest_algorithm=digest_algorithm,
            expected_base_version=expected_base_version,
        )
        session_id = created["session_id"]
        uploaded = self.upload_session(
            session_id=session_id,
            source=source,
            concurrency=concurrency,
            inject=inject,
            progress=progress,
        )
        result: dict[str, Any] = {
            "created_session": created,
            "upload": uploaded,
        }
        if commit:
            result["commit"] = self.request("commit_session", {"session_id": session_id})
        return result

