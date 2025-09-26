import os
import re
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from ..utils.fetch import build_session


_STYLE_SIZE_RE = re.compile(r"(width|height)\s*:\s*(\d+)px", re.I)


def _extract_display_size(img) -> Optional[str]:
    w = img.get('width')
    h = img.get('height')
    # Numeric attrs only
    try:
        w_i = int(w) if w and str(w).isdigit() else None
        h_i = int(h) if h and str(h).isdigit() else None
    except Exception:
        w_i = h_i = None
    if w_i and h_i:
        return f"{w_i}x{h_i}"
    # Try style attributes
    style = img.get('style') or ''
    sizes = dict((m.group(1).lower(), int(m.group(2))) for m in _STYLE_SIZE_RE.finditer(style))
    if 'width' in sizes and 'height' in sizes:
        return f"{sizes['width']}x{sizes['height']}"
    # Try srcset with w-descriptors; report the largest width found
    srcset = img.get('srcset') or ''
    widths = []
    for part in str(srcset).split(','):
        seg = part.strip().split()
        if len(seg) >= 2 and seg[1].endswith('w'):
            try:
                widths.append(int(seg[1][:-1]))
            except Exception:
                pass
    if widths:
        return f"{max(widths)}w"
    return None


def _head_size(session: requests.Session, url: str, timeout: int = 10) -> Dict[str, Any]:
    try:
        r = session.head(url, timeout=timeout, allow_redirects=True)
        # Some servers don’t support HEAD; fallback to GET with stream
        if r.status_code >= 400 or 'content-length' not in r.headers:
            r = session.get(url, timeout=timeout, allow_redirects=True, stream=True)
        ctype = r.headers.get('content-type') or ''
        clen = r.headers.get('content-length')
        size = int(clen) if clen and clen.isdigit() else None
        return {'content_type': ctype.split(';')[0], 'bytes': size}
    except requests.RequestException:
        return {'content_type': None, 'bytes': None}


def check_images(soup: BeautifulSoup, base_url: Optional[str] = None, *, fetch_sizes: bool = False, max_fetch: int = 20, timeout: int = 10) -> Dict:
    images = soup.find_all('img')
    missing: List[str] = []
    missing_no_title: List[str] = []
    missing_with_title: List[str] = []
    poor: List[str] = []
    size_info: List[Dict[str, Any]] = []
    default_name_re = re.compile(r"^(img[_-]?\d+|dsc[_-]?\d+|image[_-]?\d+|photo[_-]?\d+)$", re.I)
    # Build a session if we will fetch sizes
    session = build_session() if fetch_sizes else None
    for img in images:
        alt = img.get('alt')
        alt_text = (str(alt).strip() if isinstance(alt, str) else None)
        if not alt_text:
            src_val = img.get('src') or img.get('data-src') or img.get('srcset') or ''
            src_str = src_val if isinstance(src_val, str) else str(src_val)
            missing.append(src_str)
            title_attr = img.get('title')
            has_title = bool(title_attr and str(title_attr).strip())
            if has_title:
                missing_with_title.append(src_str)
            else:
                missing_no_title.append(src_str)
        else:
            # Heuristics for poor alt quality
            # - Very short (<3 words)
            # - Alt equals filename (without extension) or looks like a default camera name
            src = img.get('src') or ''
            base = os.path.splitext(os.path.basename(src.split('?')[0]))[0]
            if len(alt_text.split()) < 3:
                poor.append(src or alt_text)
            elif base and (alt_text.lower() == base.lower() or default_name_re.match(base or '')):
                poor.append(src or alt_text)

        # Collect size information (limited)
        if fetch_sizes and session and base_url and len(size_info) < max_fetch:
            raw_src = img.get('src') or (img.get('data-src') if isinstance(img.get('data-src'), str) else None)
            if raw_src:
                abs_src = urljoin(base_url, raw_src)
                disp = _extract_display_size(img)
                meta = _head_size(session, abs_src, timeout=timeout)
                size_info.append({
                    'src': abs_src,
                    'display': disp,
                    'bytes': meta.get('bytes'),
                    'type': meta.get('content_type'),
                })

    status = "pass" if images and not missing and not poor else ("warning" if images else "fail")
    oversized_threshold = 500 * 1024  # 500KB default heuristic
    oversized_count = sum(1 for it in size_info if (it.get('bytes') or 0) > oversized_threshold)
    return {
        "total_images": len(images),
        "missing_alt": missing[:50],  # cap to avoid huge outputs
        "missing_alt_no_title": missing_no_title[:50],
        "missing_alt_with_title": missing_with_title[:50],
        "poor_alt": poor[:50],
        "status": status,
        "sizes": size_info,
        "oversized_count": oversized_count,
        "oversized_threshold_bytes": oversized_threshold,
        "message": (
            (
                f"All {len(images)} images have descriptive alt text." if images and not missing and not poor
                else ("No images found." if not images else (
                    (f"{len(missing)} image(s) missing alt. "
                     f"{len(missing_no_title)} without alt or title. "
                     f"{len(missing_with_title)} missing alt but has title." ) if missing else f"{len(poor)} image(s) with weak alt."
                ))
            )
        )
    }
