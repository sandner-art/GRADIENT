# src/gradient/ai/diffusion/styles.py
"""Prompt styles (A1111 `styles.csv`), wildcards and `<lora:name:weight>` tags.

A style has a name, a prompt and a negative prompt. The style's prompt may hold `{prompt}`,
where your prompt goes; without it the style follows your prompt after a comma (A1111). Put
style files into the styles folder (`<data>/styles`, setting `paths.styles`): A1111/Forge
`styles.csv` (columns name, prompt, negative_prompt) or JSON (a list of {name, prompt,
negative_prompt}, as the ComfyUI stylers use). The part of a name before ":" is its group
("CX: Dynamic" is in group CX).

`style_position` places the styles' text:
* template - A1111: into `{prompt}`, else after the prompt.
* before / after - the style's text (without `{prompt}`) before or after your prompt.
* chunk - after your prompt, in its own 75-token chunk (BREAK), so on SD 1.x/2.x and SDXL
  the style is encoded apart from your words.
`style_start` > 0 adds the styles only from that fraction of the steps (prompt editing), so
the composition comes from your prompt alone (SD 1.x/2.x and SDXL).

With A1111 prompt syntax (not `plain`), `{a|b|c}` picks one option (by the seed, as the
dynamic prompts extension does) and `<lora:name:0.8>` loads that LoRA instead of being
read as words.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gradient.core.config import get_data_dir, get_setting, resolve_project_path

logger = logging.getLogger(__name__)

POSITIONS = ("template", "before", "after", "chunk")
STYLE_SUFFIXES = (".csv", ".json")
_WILDCARD = re.compile(r"\{([^{}]*\|[^{}]*)\}")
_TAG = re.compile(r"<(lora|lyco|hypernet):([^:>]+)(?::([^:>]*))?(?::[^>]*)?>", re.IGNORECASE)
_PLACEHOLDER = re.compile(r"[\s,.]*\{prompt\}[\s,.]*")


@dataclass(frozen=True)
class Style:
    name: str
    prompt: str
    negative_prompt: str
    file: str
    group: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "group": self.group,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "file": self.file,
        }


# --------------------------------------------------------------------------- files
def styles_dir(create: bool = True) -> Path:
    configured = get_setting("paths.styles", None)
    path = resolve_project_path(configured) if configured else get_data_dir() / "styles"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def _group(name: str, file: str) -> str:
    head, sep, _ = name.partition(":")
    return head.strip() if sep and head.strip() else Path(file).stem


def _rows(text: str, filename: str) -> list[dict[str, Any]]:
    if filename.lower().endswith(".json"):
        data = json.loads(text)
        if isinstance(data, dict):  # {name: {prompt, negative_prompt}}
            data = [{"name": k, **v} for k, v in data.items() if isinstance(v, dict)]
        return [row for row in data if isinstance(row, dict)]
    reader = csv.DictReader(io.StringIO(text))
    return [{(k or "").strip().lower(): v for k, v in row.items()} for row in reader]


def parse_text(text: str, filename: str) -> list[Style]:
    """Styles in a file's text (rows without a name, or without any text, are skipped)."""
    out = []
    for row in _rows(text.lstrip("\ufeff"), filename):
        name = str(row.get("name") or "").strip()
        prompt = str(row.get("prompt") or "").strip()
        negative = str(row.get("negative_prompt") or row.get("negative") or "").strip()
        if name and (prompt or negative):
            out.append(Style(name, prompt, negative, filename, _group(name, filename)))
    return out


def parse_file(path: Path) -> list[Style]:
    return parse_text(path.read_text(encoding="utf-8-sig"), path.name)


_cache: tuple[tuple, list[Style], dict[str, str]] | None = None


def load_styles() -> tuple[list[Style], dict[str, str]]:
    """All styles (a later file's name clash gets " (file)" added) and {file: error}."""
    global _cache
    folder = styles_dir()
    files = sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in STYLE_SUFFIXES
    )
    key = tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in files)
    if _cache is not None and _cache[0] == key:
        return _cache[1], _cache[2]
    styles: list[Style] = []
    errors: dict[str, str] = {}
    seen: set[str] = set()
    for path in files:
        try:
            parsed = parse_file(path)
        except Exception as e:
            errors[path.name] = str(e)
            logger.warning("Styles file %s could not be read: %s", path.name, e)
            continue
        for style in parsed:
            name = style.name if style.name not in seen else f"{style.name} ({path.stem})"
            if name in seen:
                continue
            seen.add(name)
            styles.append(Style(name, style.prompt, style.negative_prompt, path.name, style.group))
    _cache = (key, styles, errors)
    return styles, errors


def find_styles(names: list[str]) -> tuple[list[Style], list[str]]:
    """The styles named (in that order) and the names not found."""
    by_name = {s.name: s for s in load_styles()[0]}
    found = [by_name[n] for n in names if n in by_name]
    return found, [n for n in names if n not in by_name]


def save_styles_file(filename: str, content: str) -> tuple[Path, int]:
    """Check an uploaded styles file and save it into the styles folder (same name: replaced)."""
    name = Path(filename.replace("\\", "/")).name
    if not name or name.startswith(".") or Path(name).suffix.lower() not in STYLE_SUFFIXES:
        raise ValueError("A styles file must be .csv (A1111) or .json")
    try:
        count = len(parse_text(content, name))
    except Exception as e:
        raise ValueError(f"Not a styles file: {e}") from e
    if not count:
        raise ValueError("No styles found (expected the columns name, prompt, negative_prompt)")
    target = styles_dir() / name
    target.write_text(content, encoding="utf-8")
    return target, count


# --------------------------------------------------------------------------- applying
def _join(first: str, second: str) -> str:
    first, second = first.strip().rstrip(","), second.strip().lstrip(",")
    if not first:
        return second.strip()
    if not second:
        return first
    return f"{first}, {second.strip()}"


def style_text(style_prompt: str) -> str:
    """The style's own words: `{prompt}` taken out, the punctuation around it tidied."""
    return _PLACEHOLDER.sub(", ", style_prompt).strip(" ,.")


def merge(prompt: str, style_prompt: str, position: str = "template", chunks: bool = True) -> str:
    style_prompt = style_prompt.strip()
    if not style_prompt:
        return prompt
    if position == "template":
        if "{prompt}" in style_prompt:
            return style_prompt.replace("{prompt}", prompt.strip())
        return _join(prompt, style_prompt)
    words = style_text(style_prompt)
    if position == "before":
        return _join(words, prompt)
    if position == "chunk" and chunks:
        return f"{prompt.strip()} BREAK {words}" if prompt.strip() else words
    return _join(prompt, words)


def resolve_wildcards(text: str, rng: random.Random) -> str:
    """`{a|b|c}` -> one option (innermost first)."""
    for _ in range(100):
        m = _WILDCARD.search(text)
        if not m:
            break
        options = m.group(1).split("|")
        text = text[: m.start()] + rng.choice(options).strip() + text[m.end() :]
    return text


def extract_tags(text: str) -> tuple[str, list[tuple[str, str, float]]]:
    """Remove `<lora:name:w>` (also lyco, hypernet) tags: (text, [(kind, name, weight)])."""
    tags: list[tuple[str, str, float]] = []

    def take(m: re.Match) -> str:
        try:
            weight = float(m.group(3)) if m.group(3) not in (None, "") else 1.0
        except ValueError:
            weight = 1.0
        tags.append((m.group(1).lower(), m.group(2).strip(), weight))
        return ""

    out = _TAG.sub(take, text)
    out = re.sub(r"\s*,(\s*,)+", ",", out)
    out = re.sub(r" {2,}", " ", out).strip(" ,")
    return out, tags


def resolve_lora(name: str, installed: list[str]) -> str | None:
    """An installed LoRA by file name or by name without extension (A1111 style)."""
    if name in installed:
        return name
    lowered = name.lower()
    for candidate in installed:
        if Path(candidate).stem.lower() == lowered:
            return candidate
    return None


@dataclass
class Prepared:
    prompt: str
    negative_prompt: str | None
    loras: list[dict[str, Any]] = field(default_factory=list)  # from tags, resolved
    params: dict[str, Any] = field(default_factory=dict)


def prepare(
    request, seed: int, clip_syntax: bool = True, installed_loras: list[str] | None = None
) -> Prepared:
    """The prompts as rendered: styles applied, wildcards picked, LoRA tags taken out.

    `clip_syntax`: the model reads A1111 syntax (SD 1.x/2.x, SDXL, not `plain`), so the
    `chunk` position (BREAK) and `style_start` (prompt editing) can be used.
    """
    prompt = request.prompt or ""
    negative = request.negative_prompt or ""
    params: dict[str, Any] = {}
    names = list(getattr(request, "styles", None) or [])
    position = getattr(request, "style_position", "template") or "template"
    if names:
        found, missing = find_styles(names)
        if missing:
            params["styles_missing"] = missing
        unstyled = prompt
        for style in found:
            prompt = merge(prompt, style.prompt, position, chunks=clip_syntax)
            negative = merge(negative, style.negative_prompt, "template")
        start = float(getattr(request, "style_start", 0.0) or 0.0)
        if found and 0 < start < 1 and prompt != unstyled:
            if clip_syntax:
                prompt = f"[{unstyled}:{prompt}:{start:g}]"
            else:
                params["style_start_applied"] = False  # needs SD 1.x/2.x or SDXL
    plain = getattr(request, "prompt_syntax", "a1111") == "plain"
    loras: list[dict[str, Any]] = []
    if not plain:
        rng = random.Random(seed)
        prompt, negative = resolve_wildcards(prompt, rng), resolve_wildcards(negative, rng)
        prompt, tags = extract_tags(prompt)
        negative, negative_tags = extract_tags(negative)
        tags += negative_tags
        if tags:
            if installed_loras is None:
                from gradient.model_manager.inventory import list_model_names

                installed_loras = list_model_names("lora")
            unresolved = []
            for kind, name, weight in tags:
                found_name = resolve_lora(name, installed_loras) if kind != "hypernet" else None
                if found_name is None:
                    unresolved.append(name)
                elif all(lo["name"] != found_name for lo in loras):
                    loras.append({"name": found_name, "weight": weight})
            params["lora_tags"] = [f"{k}:{n}:{w:g}" for k, n, w in tags]
            if unresolved:
                params["lora_tags_missing"] = unresolved
    if prompt != (request.prompt or ""):
        params["styled_prompt"] = prompt
    if negative != (request.negative_prompt or ""):
        params["styled_negative_prompt"] = negative
    return Prepared(prompt, negative, loras, params)
