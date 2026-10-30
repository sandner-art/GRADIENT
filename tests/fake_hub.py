# tests/fake_hub.py
"""A minimal fake Hugging Face Hub (model info, tree, resolve) for offline download tests.

Implements just enough of the HTTP API used by huggingface_hub.snapshot_download:
  GET  /api/models/{owner}/{repo}[/revision/{rev}]     -> repo info (sha, siblings)
  GET  /api/models/{owner}/{repo}/tree/{rev}            -> file list (recursive)
  HEAD /{owner}/{repo}/resolve/{rev}/{path}             -> X-Repo-Commit, ETag, Content-Length
  GET  /{owner}/{repo}/resolve/{rev}/{path}             -> file bytes (Range supported)
Plus GET /files/{name} as a plain "direct URL" server with Range support.
"""

import asyncio
import hashlib
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route


@dataclass
class FakeRepo:
    files: dict[str, bytes]
    gated: bool = False
    slow: bool = False  # stream slowly (for cancel tests)
    sha: str = field(default="")

    def __post_init__(self) -> None:
        if not self.sha:
            h = hashlib.sha1()
            for name in sorted(self.files):
                h.update(name.encode() + self.files[name])
            self.sha = h.hexdigest()


class FakeHub:
    def __init__(self) -> None:
        self.repos: dict[str, FakeRepo] = {}
        self.direct_files: dict[str, bytes] = {}
        self.requests: list[str] = []
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    # ---- content helpers ----
    def add_repo(self, repo_id: str, files: dict[str, bytes], **kw) -> FakeRepo:
        repo = FakeRepo(files=files, **kw)
        self.repos[repo_id] = repo
        return repo

    def add_folder(self, repo_id: str, folder: Path, **kw) -> FakeRepo:
        files = {
            str(p.relative_to(folder)).replace("\\", "/"): p.read_bytes()
            for p in folder.rglob("*")
            if p.is_file()
        }
        return self.add_repo(repo_id, files, **kw)

    # ---- handlers ----
    def _repo(self, request: Request) -> tuple[str, FakeRepo | None]:
        repo_id = f"{request.path_params['owner']}/{request.path_params['repo']}"
        return repo_id, self.repos.get(repo_id)

    @staticmethod
    def _not_found() -> Response:
        return JSONResponse(
            {"error": "Repository not found"},
            status_code=404,
            headers={"X-Error-Code": "RepoNotFound"},
        )

    @staticmethod
    def _gated() -> Response:
        return JSONResponse(
            {"error": "Access to model is restricted"},
            status_code=403,
            headers={"X-Error-Code": "GatedRepo", "X-Error-Message": "gated"},
        )

    async def info(self, request: Request) -> Response:
        self.requests.append(f"info {request.url.path}")
        repo_id, repo = self._repo(request)
        if repo is None:
            return self._not_found()
        return JSONResponse(
            {
                "id": repo_id,
                "modelId": repo_id,
                "sha": repo.sha,
                "private": False,
                "gated": "auto" if repo.gated else False,
                "siblings": [{"rfilename": name} for name in sorted(repo.files)],
                "tags": [],
            }
        )

    async def tree(self, request: Request) -> Response:
        self.requests.append(f"tree {request.url.path}")
        _, repo = self._repo(request)
        if repo is None:
            return self._not_found()
        if repo.gated:
            return self._gated()
        entries = [
            {"type": "file", "path": name, "size": len(data), "oid": hashlib.sha1(data).hexdigest()}
            for name, data in sorted(repo.files.items())
        ]
        return JSONResponse(entries)

    def _serve_bytes(
        self, request: Request, data: bytes, headers: dict, slow: bool = False
    ) -> Response:
        start = 0
        status = 200
        if rng := request.headers.get("range"):
            start = int(rng.split("=")[1].split("-")[0] or 0)
            status = 206
            headers["Content-Range"] = f"bytes {start}-{len(data) - 1}/{len(data)}"
        body = data[start:]
        headers["Content-Length"] = str(len(body))
        headers["Accept-Ranges"] = "bytes"
        if request.method == "HEAD":
            headers["Content-Length"] = str(len(data))
            return Response(status_code=200, headers=headers)
        if not slow:
            return Response(
                body, status_code=status, headers=headers, media_type="application/octet-stream"
            )

        async def gen():
            step = 256 * 1024
            for i in range(0, len(body), step):
                yield body[i : i + step]
                await asyncio.sleep(0.05)

        return StreamingResponse(
            gen(), status_code=status, headers=headers, media_type="application/octet-stream"
        )

    async def resolve(self, request: Request) -> Response:
        self.requests.append(f"{request.method} {request.url.path}")
        _, repo = self._repo(request)
        if repo is None:
            return self._not_found()
        if repo.gated:
            return self._gated()
        path = request.path_params["path"]
        data = repo.files.get(path)
        if data is None:
            return JSONResponse(
                {"error": "Entry not found"},
                status_code=404,
                headers={"X-Error-Code": "EntryNotFound"},
            )
        headers = {"X-Repo-Commit": repo.sha, "ETag": f'"{hashlib.sha1(data).hexdigest()}"'}
        return self._serve_bytes(request, data, headers, slow=repo.slow)

    async def direct(self, request: Request) -> Response:
        self.requests.append(f"{request.method} {request.url.path}")
        name = request.path_params["name"]
        data = self.direct_files.get(name)
        if data is None:
            return Response(status_code=404)
        headers = {"Content-Disposition": f'attachment; filename="{name}"'}
        return self._serve_bytes(request, data, headers)

    # ---- lifecycle ----
    def app(self) -> Starlette:
        return Starlette(
            routes=[
                Route("/api/models/{owner}/{repo}/tree/{rev}", self.tree),
                Route("/api/models/{owner}/{repo}/revision/{rev}", self.info),
                Route("/api/models/{owner}/{repo}", self.info),
                Route(
                    "/{owner}/{repo}/resolve/{rev}/{path:path}",
                    self.resolve,
                    methods=["GET", "HEAD"],
                ),
                Route("/files/{name}", self.direct, methods=["GET", "HEAD"]),
            ]
        )

    def start(self) -> "FakeHub":
        config = uvicorn.Config(
            self.app(), host="127.0.0.1", port=self.port, log_level="warning", lifespan="off"
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        deadline = time.time() + 10
        while not self._server.started and time.time() < deadline:
            time.sleep(0.05)
        return self

    def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
        if self._thread:
            self._thread.join(timeout=5)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
