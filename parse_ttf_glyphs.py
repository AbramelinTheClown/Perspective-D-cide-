"""
perspective_dcide.parse_ttf_glyphs – Generate sigil_contract.json from TTF glyph fonts.

This utility scans all .ttf files in a directory (defaults to ./glyphs next to this
module) and emits a consolidated JSON mapping of glyph metadata that serves as
a foundation for the Symbolic App System.

Each record in the JSON contains the raw glyph code-point plus placeholders for
higher-level symbolic properties which will be enriched in a later pass.

Usage (from project root):

    python -m perspective_dcide.parse_ttf_glyphs --fonts ./perspective_dcide/glyphs --out ./perspective_dcide/sigil_contract.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

try:
    from fontTools.ttLib import TTFont  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit("fontTools package missing. Run `pip install fonttools`." ) from exc


# ──────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ──────────────────────────────────────────────────────────────

def _extract_glyphs(font_path: Path) -> List[Dict[str, Any]]:
    """Return minimal glyph metadata for *font_path*."""
    tt = TTFont(str(font_path))
    cmap = tt["cmap"].getBestCmap()  # {codepoint:int -> glyphName:str}
    glyphs: List[Dict[str, Any]] = []

    for codepoint, glyph_name in cmap.items():
        glyph_id = f"U+{codepoint:04X}"
        glyphs.append(
            {
                "glyph_id": glyph_id,
                "font": font_path.name,
                "glyph_name": glyph_name,
                # Placeholders to be filled once the symbolic registry is in place.
                "symbolic_id": "",
                "collapse_function": "",
                "suit": "",
                "path": "",
            }
        )
    return glyphs


def _build_contract(fonts_dir: Path) -> List[Dict[str, Any]]:
    """Iterate over *.ttf in *fonts_dir* and aggregate glyph metadata."""
    contract: List[Dict[str, Any]] = []
    for font_path in sorted(fonts_dir.glob("*.ttf")):
        contract.extend(_extract_glyphs(font_path))
    return contract


# ──────────────────────────────────────────────────────────────
# CLI ENTRYPOINT
# ──────────────────────────────────────────────────────────────

def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Generate sigil_contract.json from glyph fonts.")
    parser.add_argument(
        "--fonts",
        type=Path,
        default=Path(__file__).parent / "glyphs",
        help="Directory containing .ttf glyph fonts.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "sigil_contract.json",
        help="Path of the output JSON file.",
    )

    args = parser.parse_args()
    fonts_dir: Path = args.fonts.expanduser().resolve()
    if not fonts_dir.exists():
        raise SystemExit(f"Fonts directory not found: {fonts_dir}")

    contract = _build_contract(fonts_dir)
    args.out.write_text(json.dumps(contract, indent=2))
    print(f"[parse_ttf_glyphs] Wrote {len(contract)} glyph records → {args.out}")


if __name__ == "__main__":
    main() 