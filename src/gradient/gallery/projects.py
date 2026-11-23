# src/gradient/gallery/projects.py
"""Project folders: hand gallery images to other tools and take their work back.

A project folder is flat, so any tool can open it directly (ArtAgents reads captions from one
folder, Kurator watches folders):

    0012_t2i_48db8b25_seed5.png        a copy of the image (PNG metadata included)
    0012_t2i_48db8b25_seed5.png.xmp    XMP sidecar: keywords, rating, prompt (Kurator, digiKam,
                                       Lightroom, darktable and Bridge read it)
    0012_t2i_48db8b25_seed5.txt        caption, written by ArtAgents (or the prompt, if asked)
    gradient-project.json              which file is which gallery image, and what GRADIENT wrote

Keywords in the sidecar: your tags, "Projects/<project>" (Kurator files it under Projects) and,
optionally, "GRADIENT/Model/<model>" and "GRADIENT/Type/<kind>".

A sync reads the folder back and compares it with what GRADIENT wrote:
- keywords another program added become tags with source "xmp"; keywords removed there remove
  those tags again (your own GRADIENT tags are never removed by a sync);
- a changed rating (xmp:Rating) becomes the image's rating;
- a caption file that GRADIENT didn't write becomes a caption under the label you choose (one
  label per captioner, so several captioners can be compared).
Files renamed by another tool are found again by their content.
"""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
import logging
import os
import re
import shutil
import uuid as uuidlib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape, quoteattr

from sqlalchemy import delete, func, select
from sqlalchemy.orm import selectinload

from gradient.core.config import get_data_dir, get_setting, resolve_project_path
from gradient.database.models import (
    ExperimentRun,
    GeneratedOutput,
    OutputCaption,
    OutputTag,
    Project,
    ProjectItem,
)
from gradient.database.session import get_db_session

from . import store
from .metadata import _version

logger = logging.getLogger(__name__)

MANIFEST = "gradient-project.json"
FORMAT = "gradient-project"
NS = "https://github.com/sandner-art/gradient/ns/1.0/"
MAX_EXPORT = 10_000
MAX_TEXT = 1024 * 1024
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
XMP_SOURCE = "xmp"
RUN_TYPE_KEYWORDS = {
    "text_to_image": "Text-to-image",
    "image_to_image": "Image-to-image",
    "xyz_grid": "XYZ grid",
    "lab_taylor": "Lab Taylor",
    "lab_attention": "Lab attention",
    "lab_concept_attention": "Lab ConceptAttention",
    "lab_token_ablation": "Lab ablation",
}

_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
_XMP_NS = {
    "rdf": _RDF,
    "dc": "http://purl.org/dc/elements/1.1/",
    "lr": "http://ns.adobe.com/lightroom/1.0/",
    "digiKam": "http://www.digikam.org/ns/1.0/",
    "xmp": "http://ns.adobe.com/xap/1.0/",
    "xmpRights": "http://ns.adobe.com/xap/1.0/rights/",
}
# Elements another program may have added to our sidecar (Kurator adds these); kept on rewrite.
_CARRY = ("dc:source", "dc:creator", "dc:rights", "xmpRights:WebStatement")

_locks: dict[int, asyncio.Lock] = {}


def _lock() -> asyncio.Lock:
    """One project operation at a time (per event loop)."""
    return _locks.setdefault(id(asyncio.get_running_loop()), asyncio.Lock())


class ProjectError(ValueError):
    pass


def projects_root() -> Path:
    """Where new project folders go: paths.projects, or <data>/projects."""
    configured = get_setting("paths.projects", None)
    if configured:
        return resolve_project_path(configured)
    return get_data_dir(create=False) / "projects"


def slug(name: str) -> str:
    s = re.sub(r"[^\w\- ]+", "", name, flags=re.UNICODE).strip().replace(" ", "-")
    return s[:60] or "project"


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def _iso(dt: datetime.datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.UTC)
    return dt.isoformat()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ------------------------------------------------------------------------------------ XMP
def sidecar_candidates(image: Path) -> list[Path]:
    """Sidecars of `image` in the order tools write them: photo.png.xmp, photo.xmp."""
    return [
        image.with_name(image.name + ".xmp"),
        image.with_suffix(".xmp"),
        image.with_name(image.name + ".XMP"),
        image.with_suffix(".XMP"),
    ]


def _read_small(path: Path) -> str | None:
    try:
        if path.stat().st_size > MAX_TEXT:
            return None
        return path.read_bytes().decode("utf-8", "ignore")
    except OSError:
        return None


def is_ours(text: str | None) -> bool:
    return bool(text) and NS in text


def _rdf_bag(tag: str, kind: str, items: list[str]) -> str:
    if not items:
        return ""
    lis = "".join(f"\n     <rdf:li>{escape(i)}</rdf:li>" for i in items)
    return f"\n   <{tag}>\n    <rdf:{kind}>{lis}\n    </rdf:{kind}>\n   </{tag}>"


def keyword_lists(keywords: list[str]) -> tuple[list[str], list[str]]:
    """(hierarchical "A/B" keywords, flat words incl. every level), as Lightroom and Kurator do."""
    hier = sorted(dict.fromkeys(k for k in keywords if k), key=str.lower)
    flat = sorted({p for k in hier for p in k.split("/") if p}, key=str.lower)
    return hier, flat


def build_sidecar(
    keywords: list[str],
    *,
    rating: int | None,
    description: str | None,
    created: str | None,
    attrs: dict[str, Any],
    carry: list[str] | None = None,
) -> str:
    """An XMP sidecar in the layout Lightroom, digiKam and Kurator write."""
    hier, flat = keyword_lists(keywords)
    head = [f'xmlns:{p}="{u}"' for p, u in _XMP_NS.items() if p != "rdf"]
    head.append(f'xmlns:gradient="{NS}"')
    head.append(f"xmp:CreatorTool={quoteattr('GRADIENT ' + _version())}")
    if created:
        head.append(f"xmp:CreateDate={quoteattr(created)}")
    if rating:
        head.append(f'xmp:Rating="{int(rating)}"')
    for key, value in attrs.items():
        if value is not None and value != "":
            head.append(f"gradient:{key}={quoteattr(str(value))}")
    body = (
        _rdf_bag("dc:subject", "Bag", flat)
        + _rdf_bag("lr:hierarchicalSubject", "Bag", [k.replace("/", "|") for k in hier])
        + _rdf_bag("digiKam:TagsList", "Seq", hier)
    )
    if description:
        body += (
            "\n   <dc:description>\n    <rdf:Alt>\n"
            f'     <rdf:li xml:lang="x-default">{escape(description)}</rdf:li>\n'
            "    </rdf:Alt>\n   </dc:description>"
        )
    for block in carry or []:
        body += "\n   " + block.strip()
    sep = "\n    "
    return (
        '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        f'<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="GRADIENT {escape(_version())}">\n'
        f' <rdf:RDF xmlns:rdf="{_RDF}">\n'
        f'  <rdf:Description rdf:about=""{sep}{sep.join(head)}>{body}\n'
        "  </rdf:Description>\n </rdf:RDF>\n</x:xmpmeta>\n"
        '<?xpacket end="w"?>\n'
    )


def carried_blocks(text: str) -> list[str]:
    """Source/author/licence elements another program added to our sidecar (kept on rewrite)."""
    out = []
    for prop in _CARRY:
        for pat in (rf"<{prop}\b[^>]*/>", rf"<{prop}\b[^>]*>.*?</{prop}>"):
            out += re.findall(pat, text, re.S)
    return out


def read_sidecar(text: str) -> dict[str, Any] | None:
    """Keywords ("A/B"), rating and description from an XMP packet; None if it isn't valid XML."""
    try:
        root = ET.fromstring(re.sub(r"<\?xpacket[^>]*\?>", "", text).strip())
    except ET.ParseError:
        return None
    ns = _XMP_NS
    hier: list[str] = []
    flat: list[str] = []
    rating: int | None = None
    description: str | None = None
    for desc in root.iter(f"{{{_RDF}}}Description"):
        for prop, sep in (
            (f"{{{ns['lr']}}}hierarchicalSubject", "|"),
            (f"{{{ns['digiKam']}}}TagsList", "/"),
        ):
            for el in desc.iter(prop):
                for li in el.iter(f"{{{_RDF}}}li"):
                    value = (li.text or "").strip()
                    if value:
                        hier.append("/".join(p.strip() for p in value.split(sep) if p.strip()))
        for el in desc.iter(f"{{{ns['dc']}}}subject"):
            flat += [(li.text or "").strip() for li in el.iter(f"{{{_RDF}}}li")]
        raw = desc.get(f"{{{ns['xmp']}}}Rating")
        if raw is None:
            el = desc.find(f"{{{ns['xmp']}}}Rating")
            raw = el.text if el is not None else None
        if raw is not None:
            try:
                rating = round(float(raw))
            except ValueError:
                pass
        for el in desc.iter(f"{{{ns['dc']}}}description"):
            li = el.find(f".//{{{_RDF}}}li")
            if li is not None and li.text:
                description = li.text.strip()
    covered = {p.lower() for k in hier for p in k.split("/")}
    keywords = list(dict.fromkeys(hier + [f for f in flat if f and f.lower() not in covered]))
    return {"keywords": keywords, "rating": rating, "description": description}


# ------------------------------------------------------------------------------- manifest
def read_manifest(folder: Path) -> dict[str, Any] | None:
    path = folder / MANIFEST
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ProjectError(f"Can't read {path}: {e}") from e
    if data.get("format") != FORMAT:
        raise ProjectError(f"{path} is not a GRADIENT project file")
    data.setdefault("items", [])
    return data


def write_manifest(folder: Path, data: dict[str, Any]) -> None:
    tmp = folder / (MANIFEST + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, folder / MANIFEST)


def generated_keywords(item: dict[str, Any], project_name: str, with_model: bool) -> list[str]:
    out = [f"Projects/{_kw(project_name)}"]
    if with_model:
        if item.get("model_id"):
            out.append(f"GRADIENT/Model/{_kw(item['model_id'])}")
        if item.get("run_type"):
            kind = RUN_TYPE_KEYWORDS.get(item["run_type"], item["run_type"])
            out.append(f"GRADIENT/Type/{_kw(kind)}")
    return out


def _kw(value: str) -> str:
    """A keyword level: no separators inside it."""
    return " ".join(str(value).replace("/", "-").replace("|", "-").split())


def _is_generated(keyword: str, project_name: str) -> bool:
    low = keyword.lower()
    return low.startswith("gradient/") or low == f"projects/{_kw(project_name)}".lower()


# --------------------------------------------------------------------------------- export
def resolve_folder(path: str | None, name: str) -> Path:
    folder = Path(path).expanduser() if path else projects_root() / slug(name)
    if not folder.is_absolute():
        folder = projects_root() / folder
    return folder


async def export(
    *,
    ids: list[int] | None = None,
    filters: dict[str, Any] | None = None,
    project: str | None = None,
    name: str | None = None,
    path: str | None = None,
    sidecars: bool = True,
    model_keywords: bool = True,
    captions: str = "none",
    link: bool = False,
) -> dict[str, Any]:
    """Copy images into a project folder (new or existing) and write sidecars and the manifest."""
    if captions not in ("none", "prompt"):
        raise ProjectError("captions must be 'none' or 'prompt'")
    async with _lock():
        if not await _export_items(ids, filters):
            raise ProjectError("No images to export")
        proj = await _project_for_export(project, name, path)
        folder = Path(proj["path"])
        report: dict[str, Any] = {}
        if await asyncio.to_thread((folder / MANIFEST).is_file):
            # take the other tools' work in first, so rewriting a sidecar loses nothing
            report["sync"] = await _sync_locked(proj["uuid"], caption_source="artagents")
        items = await _export_items(ids, filters)
        options = {
            "sidecars": sidecars,
            "model_keywords": model_keywords,
            "captions": captions,
            "link": link,
        }
        result = await asyncio.to_thread(_export_files, folder, proj, items, options)
        async with get_db_session() as db:
            row = await db.scalar(select(Project).where(Project.uuid == proj["uuid"]))
            have = {
                i.output_id: i
                for i in (
                    await db.scalars(select(ProjectItem).where(ProjectItem.project_id == row.id))
                ).all()
            }
            for output_id, file_name in result["files"].items():
                if output_id in have:
                    have[output_id].file_name = file_name
                else:
                    db.add(ProjectItem(project_id=row.id, output_id=output_id, file_name=file_name))
            row.updated = _now()
            row.options = options
        report.update({k: v for k, v in result.items() if k != "files"})
        report["project"] = await get_project(proj["uuid"])
        return report


async def _project_for_export(project: str | None, name: str | None, path: str | None) -> dict:
    if project:
        return await get_project(project)
    if not name or not name.strip():
        raise ProjectError("Give the project a name")
    name = " ".join(name.split())[:80]
    folder = resolve_folder(path, name)
    manifest = await asyncio.to_thread(_check_folder, folder)
    async with get_db_session() as db:
        row = await db.scalar(select(Project).where(Project.path == str(folder)))
        if row is None and manifest:  # an existing project folder we don't know yet
            row = await db.scalar(select(Project).where(Project.uuid == manifest["uuid"]))
            if row is not None:
                row.path = str(folder)
        if row is None:
            row = Project(
                uuid=(manifest or {}).get("uuid") or str(uuidlib.uuid4()),
                name=(manifest or {}).get("name") or name,
                path=str(folder),
            )
            db.add(row)
        await db.flush()
        return {"uuid": row.uuid, "name": row.name, "path": row.path}


def _check_folder(folder: Path) -> dict[str, Any] | None:
    if folder.exists() and not folder.is_dir():
        raise ProjectError(f"{folder} is a file")
    manifest = read_manifest(folder) if folder.is_dir() else None
    if folder.is_dir() and manifest is None and any(folder.iterdir()):
        raise ProjectError(
            f"{folder} is not empty and isn't a GRADIENT project. Pick an empty or new folder."
        )
    return manifest


async def _export_items(ids: list[int] | None, filters: dict[str, Any] | None) -> list[dict]:
    if ids is not None:
        stmt = select(GeneratedOutput).where(GeneratedOutput.id.in_(ids))
    else:
        stmt = store.filtered_outputs(**(filters or {}))
    stmt = stmt.options(selectinload(GeneratedOutput.tags), selectinload(GeneratedOutput.run))
    async with get_db_session() as db:
        rows = (await db.scalars(stmt.order_by(GeneratedOutput.id).limit(MAX_EXPORT + 1))).all()
        if len(rows) > MAX_EXPORT:
            raise ProjectError(f"At most {MAX_EXPORT} images per export")
        out = []
        for o in rows:
            run: ExperimentRun | None = o.run
            out.append(
                {
                    "output_id": o.id,
                    "source": o.file_path,
                    "prompt": o.prompt,
                    "model_id": o.model_id,
                    "seed": o.seed,
                    "run_uuid": run.run_uuid if run else None,
                    "run_type": run.run_type if run else None,
                    "created": _iso(o.creation_time),
                    "rating": o.user_rating,
                    "favorite": bool(o.favorite),
                    "tags": list(dict.fromkeys(t.tag for t in o.tags)),
                }
            )
        return out


def _export_files(
    folder: Path, proj: dict, items: list[dict], options: dict[str, Any]
) -> dict[str, Any]:
    folder.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(folder) or {
        "format": FORMAT,
        "version": 1,
        "uuid": proj["uuid"],
        "name": proj["name"],
        "created": _iso(_now()),
        "items": [],
    }
    by_id = {it["output_id"]: it for it in manifest["items"]}
    res: dict[str, Any] = {
        "added": 0,
        "updated": 0,
        "missing_source": [],
        "kept_foreign_sidecars": [],
        "kept_captions": 0,
        "errors": [],
        "files": {},
    }
    for item in items:
        src = Path(item["source"])
        entry = by_id.get(item["output_id"])
        try:
            if not src.is_file():
                res["missing_source"].append(item["output_id"])
                continue
            file_name = entry["file"] if entry else f"{item['output_id']:04d}_{src.name}"
            target = folder / file_name
            sha = _sha256(src)
            if not target.is_file() or (entry and entry.get("sha256") != sha):
                _place(src, target, options["link"])
            if entry is None:
                entry = {"output_id": item["output_id"], "file": file_name}
                manifest["items"].append(entry)
                by_id[item["output_id"]] = entry
                res["added"] += 1
            else:
                res["updated"] += 1
            entry.update(
                sha256=sha,
                run_uuid=item["run_uuid"],
                seed=item["seed"],
                model_id=item["model_id"],
                run_type=item["run_type"],
                prompt=item["prompt"],
            )
            if options["sidecars"]:
                _write_sidecar(target, item, entry, proj["name"], options, res)
            if options["captions"] == "prompt" and item.get("prompt"):
                _write_caption(target, item["prompt"], entry, res)
            res["files"][item["output_id"]] = file_name
        except OSError as e:
            res["errors"].append(f"{src.name}: {e}")
    manifest["name"] = proj["name"]
    manifest["updated"] = _iso(_now())
    manifest["app"] = f"GRADIENT {_version()}"
    write_manifest(folder, manifest)
    return res


def _place(src: Path, target: Path, link: bool) -> None:
    tmp = target.with_name(target.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    if link:
        try:
            os.link(src, tmp)
        except OSError:
            shutil.copy2(src, tmp)
    else:
        shutil.copy2(src, tmp)
    os.replace(tmp, target)


def _write_sidecar(
    target: Path, item: dict, entry: dict, project_name: str, options: dict, res: dict
) -> None:
    existing = [c for c in sidecar_candidates(target) if c.is_file()]
    carry: list[str] = []
    for c in existing:
        text = _read_small(c)
        if not is_ours(text):  # another program's sidecar: leave it alone
            res["kept_foreign_sidecars"].append(c.name)
            return
        carry = carried_blocks(text or "")
    keywords = item["tags"] + generated_keywords(item, project_name, options["model_keywords"])
    content = build_sidecar(
        keywords,
        rating=item["rating"],
        description=item.get("prompt"),
        created=item.get("created"),
        attrs={
            "OutputId": item["output_id"],
            "RunUuid": item.get("run_uuid"),
            "Model": item.get("model_id"),
            "Seed": item.get("seed"),
        },
        carry=carry,
    )
    path = target.with_name(target.name + ".xmp")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(content, encoding="utf-8", newline="\n")
    os.replace(tmp, path)
    entry["sidecar"] = path.name
    entry["seen_keywords"] = sorted(set(item["tags"]), key=str.lower)
    entry["rating"] = item["rating"]


def _write_caption(target: Path, prompt: str, entry: dict, res: dict) -> None:
    path = target.with_suffix(".txt")
    if path.is_file():
        current = _read_small(path)
        if current is not None and current.strip() != (entry.get("caption") or "").strip():
            res["kept_captions"] += 1  # someone else's caption (ArtAgents): keep it
            return
    path.write_text(prompt, encoding="utf-8")
    entry["caption"] = prompt
    entry["caption_seen"] = _text_sha(prompt)


# ----------------------------------------------------------------------------------- sync
async def sync(uuid: str, caption_source: str = "artagents", captions_as_tags: bool = False):
    async with _lock():
        return await _sync_locked(uuid, caption_source, captions_as_tags)


def _clean_source(label: str) -> str:
    label = " ".join(str(label or "").split())[:60]
    if not label or label in ("manual", XMP_SOURCE):
        raise ProjectError("Pick a caption label other than 'manual' or 'xmp'")
    return label


async def _sync_locked(
    uuid: str, caption_source: str = "artagents", captions_as_tags: bool = False
) -> dict[str, Any]:
    caption_source = _clean_source(caption_source)
    proj = await get_project(uuid)
    folder = Path(proj["path"])
    if not proj["exists"]:
        raise ProjectError(f"The folder {folder} doesn't exist (moved or deleted?)")
    manifest, observed, unknown = await asyncio.to_thread(_scan, folder)
    report: dict[str, Any] = {
        "files": len(observed),
        "renamed": [],
        "missing": [],
        "not_in_gallery": 0,
        "tags_added": 0,
        "tags_removed": 0,
        "ratings_changed": 0,
        "captions": 0,
        "unreadable_sidecars": [],
        "unknown_files": unknown[:50],
        "unknown_count": len(unknown),
    }
    entries = {e["output_id"]: e for e in manifest["items"]}
    async with get_db_session() as db:
        project_row = await db.scalar(select(Project).where(Project.uuid == uuid))
        outputs = {
            o.id: o
            for o in (
                await db.scalars(
                    select(GeneratedOutput)
                    .where(GeneratedOutput.id.in_(list(entries)))
                    .options(selectinload(GeneratedOutput.tags))
                )
            ).all()
        }
        items = {
            i.output_id: i
            for i in (
                await db.scalars(
                    select(ProjectItem).where(ProjectItem.project_id == project_row.id)
                )
            ).all()
        }
        for obs in observed:
            entry = entries[obs["output_id"]]
            if obs["missing"]:
                report["missing"].append(entry["file"])
                continue
            if obs["file"] != entry["file"]:
                report["renamed"].append({"from": entry["file"], "to": obs["file"]})
                entry["file"] = obs["file"]
                if obs["output_id"] in items:
                    items[obs["output_id"]].file_name = obs["file"]
            o = outputs.get(obs["output_id"])
            if o is None:
                report["not_in_gallery"] += 1
                continue
            if obs["output_id"] not in items:
                db.add(
                    ProjectItem(project_id=project_row.id, output_id=o.id, file_name=obs["file"])
                )
            if obs["sidecar_error"]:
                report["unreadable_sidecars"].append(obs["sidecar_error"])
            elif obs["sidecar"] is not None:
                _apply_sidecar(o, entry, obs["sidecar"], proj["name"], report)
            if obs["caption"] is not None:
                await _apply_caption(
                    db, o, entry, obs["caption"], caption_source, captions_as_tags, report
                )
        project_row.last_sync = _now()
    await asyncio.to_thread(write_manifest, folder, manifest)
    report["project"] = await get_project(uuid)
    return report


def _scan(folder: Path) -> tuple[dict, list[dict], list[str]]:
    manifest = read_manifest(folder)
    if manifest is None:
        raise ProjectError(f"No {MANIFEST} in {folder}")
    images = {p.name: p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()}
    claimed = {e["file"] for e in manifest["items"] if e.get("file") in images}
    by_hash: dict[str, str] | None = None
    observed = []
    for entry in manifest["items"]:
        name = entry.get("file")
        if name not in images:
            if by_hash is None:  # renamed by another tool? find it by content
                by_hash = {_sha256(p): n for n, p in images.items() if n not in claimed}
            name = by_hash.pop(entry.get("sha256", ""), None)
            if name:
                claimed.add(name)
        if not name:
            observed.append({"output_id": entry["output_id"], "missing": True})
            continue
        image = images[name]
        sidecar, error = None, None
        for c in sidecar_candidates(image):
            if c.is_file():
                text = _read_small(c)
                sidecar = read_sidecar(text) if text is not None else None
                if sidecar is None:
                    error = c.name
                break
        caption_file = image.with_suffix(".txt")
        caption = _read_small(caption_file) if caption_file.is_file() else None
        observed.append(
            {
                "output_id": entry["output_id"],
                "missing": False,
                "file": name,
                "sidecar": sidecar,
                "sidecar_error": error,
                "caption": caption,
            }
        )
    unknown = sorted(n for n in images if n not in claimed)
    return manifest, observed, unknown


def _apply_sidecar(
    o: GeneratedOutput, entry: dict, sidecar: dict, project_name: str, report: dict
) -> None:
    seen = set(entry.get("seen_keywords") or [])
    # Flat words that were only levels of a hierarchical keyword ("Creatures" of
    # "Creatures/Dragons") can outlive it: programs keep old dc:subject words when they merge.
    levels = {
        p.lower()
        for k in [*seen, *generated_keywords(entry, project_name, True)]
        if "/" in k
        for p in k.split("/")
    }
    current = [
        k
        for k in store._clean_tags(sidecar["keywords"])
        if not _is_generated(k, project_name)
        and not ("/" not in k and k not in seen and k.lower() in levels)
    ]
    have = {t.tag for t in o.tags}
    for k in current:
        if k not in seen and k not in have:
            o.tags.append(OutputTag(tag=k, source=XMP_SOURCE))
            have.add(k)
            report["tags_added"] += 1
    gone = seen - set(current)
    if gone:
        before = len(o.tags)
        o.tags = [t for t in o.tags if not (t.source == XMP_SOURCE and t.tag in gone)]
        report["tags_removed"] += before - len(o.tags)
    entry["seen_keywords"] = sorted(set(current), key=str.lower)
    rating = sidecar.get("rating")
    if rating is not None and rating != entry.get("rating"):
        rating = max(0, min(5, rating))
        if (o.user_rating or 0) != rating:
            o.user_rating = rating or None
            report["ratings_changed"] += 1
        entry["rating"] = rating


async def _apply_caption(
    db, o: GeneratedOutput, entry: dict, text: str, source: str, as_tags: bool, report: dict
) -> None:
    text = text.strip()
    if not text or text == (entry.get("caption") or "").strip():
        return  # empty, or the prompt GRADIENT wrote itself
    sha = _text_sha(text)
    if sha == entry.get("caption_seen"):
        return
    row = await db.scalar(
        select(OutputCaption).where(OutputCaption.output_id == o.id, OutputCaption.source == source)
    )
    if row is None:
        db.add(OutputCaption(output_id=o.id, source=source, text=text))
    else:
        row.text = text
        row.updated = _now()
    report["captions"] += 1
    entry["caption_seen"] = sha
    if as_tags:
        have = {(t.tag, t.source) for t in o.tags}
        for tag in store._clean_tags(text.replace("\n", ",").split(",")):
            if len(tag.split()) <= 6 and (tag, source) not in have:
                o.tags.append(OutputTag(tag=tag, source=source))
                have.add((tag, source))
                report["tags_added"] += 1


# ------------------------------------------------------------------------------ registry
def _project_view(p: Project, count: int) -> dict[str, Any]:
    return {
        "uuid": p.uuid,
        "name": p.name,
        "path": p.path,
        "created": _iso(p.created),
        "updated": _iso(p.updated),
        "last_sync": _iso(p.last_sync),
        "count": count,
        "options": p.options or {},
    }


async def list_projects() -> list[dict[str, Any]]:
    async with get_db_session() as db:
        rows = (await db.scalars(select(Project).order_by(Project.updated.desc()))).all()
        counts = dict(
            (
                await db.execute(
                    select(ProjectItem.project_id, func.count()).group_by(ProjectItem.project_id)
                )
            ).all()
        )
        views = [_project_view(p, counts.get(p.id, 0)) for p in rows]
    exists = await asyncio.to_thread(lambda: [Path(v["path"]).is_dir() for v in views])
    for v, e in zip(views, exists, strict=True):
        v["exists"] = e
    return views


async def get_project(uuid: str) -> dict[str, Any]:
    async with get_db_session() as db:
        p = await db.scalar(select(Project).where(Project.uuid == uuid))
        if p is None:
            raise KeyError(f"No project {uuid}")
        count = await db.scalar(select(func.count()).where(ProjectItem.project_id == p.id))
        view = _project_view(p, count or 0)
    view["exists"] = await asyncio.to_thread(Path(view["path"]).is_dir)
    return view


async def open_folder(path: str, caption_source: str = "artagents") -> dict[str, Any]:
    """Add an existing project folder (moved, or made on another machine) and sync it."""
    folder = resolve_folder(path, "")
    manifest = await asyncio.to_thread(read_manifest, folder)
    if manifest is None:
        raise ProjectError(f"No {MANIFEST} in {folder}")
    async with get_db_session() as db:
        row = await db.scalar(select(Project).where(Project.uuid == manifest["uuid"]))
        if row is None:
            row = await db.scalar(select(Project).where(Project.path == str(folder)))
        if row is None:
            db.add(
                Project(
                    uuid=manifest["uuid"],
                    name=manifest.get("name") or folder.name,
                    path=str(folder),
                )
            )
        else:
            row.path = str(folder)
    return await sync(manifest["uuid"], caption_source=caption_source)


async def forget(uuid: str) -> None:
    """Remove a project from the list (the folder and its files stay)."""
    async with get_db_session() as db:
        p = await db.scalar(select(Project).where(Project.uuid == uuid))
        if p is None:
            raise KeyError(f"No project {uuid}")
        await db.execute(delete(ProjectItem).where(ProjectItem.project_id == p.id))
        await db.delete(p)
