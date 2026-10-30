# src/gradient/cli.py
"""Command line interface.

gradient                         start the app (API + web UI) at http://127.0.0.1:8000
gradient serve --port 9000 --open
gradient info                    device, versions
gradient models catalog [--type diffusers]
gradient models list [--type lora]
gradient models download sdxl-base
gradient models download stabilityai/sdxl-turbo
gradient models download bartowski/Qwen_Qwen3-4B-GGUF --type llm
gradient models download "https://civitai.com/models/123?modelVersionId=456" --type lora
gradient models delete diffusers sdxl-turbo
"""

import argparse
import asyncio
import sys
import threading
import webbrowser


def _human(n: float | None) -> str:
    if not n:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TB"


# --------------------------------------------------------------------------- serve
def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from gradient.core.config import PROJECT_ROOT, get_setting
    from gradient.core.log_setup import setup_logging

    setup_logging()
    host = args.host or get_setting("app.host", "127.0.0.1")
    port = int(args.port or get_setting("app.port", 8000))
    reload = args.reload or bool(get_setting("app.reload", False))
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}"
    print(f"Starting GRADIENT at {url} ... (the first start can take a minute while PyTorch loads)")
    open_browser = bool(args.open or get_setting("app.open_browser", False))
    threading.Thread(target=_announce_when_ready, args=(url, open_browser), daemon=True).start()
    uvicorn.run(
        "gradient.api.main:app",
        host=host,
        port=port,
        reload=reload,
        reload_dirs=[str(PROJECT_ROOT / "src")] if reload else None,
        log_config=None,
    )
    return 0


def _announce_when_ready(url: str, open_browser: bool, timeout: float = 600) -> None:
    """Wait until the API answers, then say so (and open the browser if asked)."""
    import time
    import urllib.request

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never proxy localhost
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with opener.open(f"{url}/api/v1/status", timeout=2):
                break
        except Exception:
            time.sleep(1)
    else:
        return
    print(
        f"\nGRADIENT is ready: {url}  (API docs: {url}/docs). Press Ctrl+C to stop.\n", flush=True
    )
    if open_browser:
        webbrowser.open(url)


def cmd_info(_: argparse.Namespace) -> int:
    from importlib.metadata import PackageNotFoundError, version

    from gradient.core.config import get_model_search_dirs
    from gradient.core.device import device_summary

    for key, value in device_summary().items():
        print(f"{key:>20}: {value}")
    for pkg in (
        "diffusers",
        "transformers",
        "huggingface-hub",
        "accelerate",
        "peft",
        "fastapi",
        "llama-cpp-python",
    ):
        try:
            v = version(pkg)
        except PackageNotFoundError:
            v = "not installed"
        print(f"{pkg:>20}: {v}")
    print(f"{'model dirs':>20}: {', '.join(str(d) for d in get_model_search_dirs())}")
    return 0


# --------------------------------------------------------------------------- models
def cmd_models_catalog(args: argparse.Namespace) -> int:
    from gradient.model_manager import manifest
    from gradient.model_manager.catalog import load_catalog

    installed = {m.get("catalog_id") for m in manifest.all_entries().values()}
    rows = [
        e
        for e in load_catalog()
        if (not args.type or e.type == args.type) and ("test" not in e.tags or args.all)
    ]
    if not rows:
        print("No catalog entries.")
        return 0
    print(f"{'ID':<28} {'TYPE':<11} {'SIZE':>8} {'VRAM':>6}  NAME")
    for e in rows:
        flags = (" [gated]" if e.gated else "") + (" [installed]" if e.id in installed else "")
        size = f"{e.size_gb:.1f}G" if e.size_gb else "?"
        vram = f"{e.min_vram_gb:g}G" if e.min_vram_gb else "-"
        print(f"{e.id:<28} {e.type:<11} {size:>8} {vram:>6}  {e.name}{flags}")
    print("\nDownload with: gradient models download <ID>")
    return 0


def cmd_models_list(args: argparse.Namespace) -> int:
    from gradient.model_manager.inventory import list_local_models

    types = (
        [args.type]
        if args.type
        else ["diffusers", "checkpoints", "vae", "lora", "controlnet", "embeddings", "llm"]
    )
    total = 0
    for t in types:
        models = list_local_models(t)  # type: ignore[arg-type]
        if not models:
            continue
        print(f"\n[{t}]")
        for m in models:
            total += 1
            origin = (m.source or {}).get("repo_id") or (m.source or {}).get("url") or ""
            print(f"  {m.name:<50} {_human(m.size_bytes):>10}  {m.format:<11} {origin}")
    if not total:
        print("No local models yet. Try: gradient models catalog")
    return 0


def _build_request(args: argparse.Namespace):
    from gradient.model_manager.catalog import get_catalog_entry
    from gradient.model_manager.schemas import DownloadRequest

    ref: str = args.ref
    common = {
        "model_type": args.type,
        "files": args.file or None,
        "target_name": args.name,
        "token": args.token,
        "revision": args.revision,
    }
    if get_catalog_entry(ref):
        return DownloadRequest(catalog_id=ref, target_name=args.name, token=args.token)
    if ref.startswith("https://") and "huggingface.co" not in ref and "hf.co" not in ref:
        return DownloadRequest(url=ref, **common)
    return DownloadRequest(repo_id=ref, **common)


async def _download(args: argparse.Namespace) -> int:
    from gradient.model_manager.downloader import download_manager

    try:
        job = await download_manager.submit(_build_request(args))
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"Downloading {job.name} -> {job.target_path}")
    task = asyncio.ensure_future(download_manager.wait(job.id))
    try:
        while not task.done():
            await asyncio.sleep(0.5)
            pct = f"{job.progress * 100:5.1f}%" if job.progress is not None else "  ... "
            line = f"\r  {job.status:<11} {pct}  {_human(job.downloaded_bytes)} / {_human(job.total_bytes)}  {_human(job.speed_bps)}/s   "
            sys.stdout.write(line)
            sys.stdout.flush()
    except (KeyboardInterrupt, asyncio.CancelledError):
        await download_manager.cancel(job.id)
    print()
    if job.status == "completed":
        print(f"Done: {job.target_path}")
        return 0
    print(f"{job.status}: {job.error or ''}", file=sys.stderr)
    return 1


def cmd_models_download(args: argparse.Namespace) -> int:
    from gradient.core.log_setup import setup_logging

    setup_logging()
    try:
        return asyncio.run(_download(args))
    except KeyboardInterrupt:
        print("\nCancelled (partial files kept; run the same command again to resume).")
        return 130


def cmd_models_delete(args: argparse.Namespace) -> int:
    from gradient.model_manager.inventory import delete_local_model

    try:
        path = delete_local_model(args.type, args.name)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"Deleted {path}")
    return 0


# --------------------------------------------------------------------------- parser
MODEL_TYPES = ["diffusers", "checkpoints", "vae", "lora", "controlnet", "embeddings", "llm"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gradient", description="GRADIENT generative research workbench"
    )
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="Start the API server and web UI (default)")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.add_argument(
        "--reload", action="store_true", help="Auto-reload on code changes (development)"
    )
    serve.add_argument("--open", action="store_true", help="Open the browser")
    serve.set_defaults(func=cmd_serve)

    info = sub.add_parser("info", help="Show device and library versions")
    info.set_defaults(func=cmd_info)

    models = sub.add_parser("models", help="Manage models (catalog, download, list, delete)")
    msub = models.add_subparsers(dest="models_command", required=True)

    cat = msub.add_parser("catalog", help="List the curated downloadable models")
    cat.add_argument("--type", choices=MODEL_TYPES)
    cat.add_argument("--all", action="store_true", help="Include tiny test models")
    cat.set_defaults(func=cmd_models_catalog)

    lst = msub.add_parser("list", help="List installed models")
    lst.add_argument("--type", choices=MODEL_TYPES)
    lst.set_defaults(func=cmd_models_list)

    dl = msub.add_parser(
        "download", help="Download a catalog model, Hugging Face repo/file, or URL"
    )
    dl.add_argument("ref", help="catalog ID, 'owner/repo', huggingface.co URL or https URL")
    dl.add_argument("--type", choices=MODEL_TYPES, help="Model type (inferred when possible)")
    dl.add_argument("--file", action="append", help="Specific file from the repo (repeatable)")
    dl.add_argument("--name", help="Store under this folder/file name")
    dl.add_argument("--revision")
    dl.add_argument("--token", help="Access token for gated repos (or set HF_TOKEN)")
    dl.set_defaults(func=cmd_models_download)

    rm = msub.add_parser("delete", help="Delete an installed model")
    rm.add_argument("type", choices=MODEL_TYPES)
    rm.add_argument("name")
    rm.set_defaults(func=cmd_models_delete)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        args = parser.parse_args(["serve", *(argv or [])])
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
