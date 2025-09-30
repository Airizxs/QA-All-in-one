from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from bs4 import BeautifulSoup


def _extract_dimensions(img) -> Tuple[Optional[int], Optional[int]]:
    width_attr = img.get("width")
    height_attr = img.get("height")
    width = int(width_attr) if width_attr and str(width_attr).isdigit() else None
    height = int(height_attr) if height_attr and str(height_attr).isdigit() else None

    if width is None or height is None:
        style = img.get("style") or ""
        pairs = {
            part.split(":", 1)[0].strip().lower(): part.split(":", 1)[1].strip().lower()
            for part in style.split(";") if ":" in part
        }
        try:
            if width is None and "width" in pairs and pairs["width"].endswith("px"):
                width = int(pairs["width"].rstrip("px"))
            if height is None and "height" in pairs and pairs["height"].endswith("px"):
                height = int(pairs["height"].rstrip("px"))
        except ValueError:
            pass

    return width, height


def check_hero_image(soup: BeautifulSoup) -> Dict[str, Any]:
    hero_img = soup.find("img", class_=lambda cls: cls and "hero" in " ".join(cls).lower())

    if hero_img is None:
        hero_img = soup.find("img", id=lambda ident: ident and "hero" in ident.lower())

    if hero_img is None:
        hero_img = soup.find("img")

    if hero_img is None:
        return {
            "status": "warning",
            "message": "No images found on the page; hero image cannot be verified.",
        }

    width, height = _extract_dimensions(hero_img)
    status = "pass"
    notes = []

    if width is None or height is None:
        status = "warning"
        notes.append("Hero dimensions not specified; add width/height attributes or inline styles.")
    else:
        notes.append(f"Hero image is {width}x{height}px.")

    src = hero_img.get("src") or hero_img.get("data-src") or "(no src)"

    return {
        "status": status,
        "message": " ".join(notes) if notes else "Hero image detected.",
        "src": src,
        "width": width,
        "height": height,
    }
