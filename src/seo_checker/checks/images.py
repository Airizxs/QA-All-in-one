import re
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from ..utils.fetch import build_session


_STYLE_SIZE_RE = re.compile(r"(width|height)\s*:\s*(\d+)px", re.I)
_OPACITY_ZERO_RE = re.compile(r"opacity\s*:\s*0(?:\.0+)?", re.I)
_HIDDEN_CLASS_HINTS = {
    'hidden',
    'is-hidden',
    'sr-only',
    'visually-hidden',
    'screen-reader-only',
    'aria-hidden',
}


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


def _looks_hidden(style: str) -> bool:
    style_lower = style.lower()
    if 'display:none' in style_lower or 'visibility:hidden' in style_lower:
        return True
    if _OPACITY_ZERO_RE.search(style_lower):
        return True
    return False


def _has_hidden_class(img) -> bool:
    classes = img.get('class')
    if not classes:
        return False
    tokens: List[str] = []
    if isinstance(classes, (list, tuple, set)):
        tokens = [str(cls).lower() for cls in classes if cls]
    elif isinstance(classes, str):
        tokens = [token.lower() for token in classes.split() if token]
    return any(token in _HIDDEN_CLASS_HINTS for token in tokens)


def _is_hidden_image(img) -> bool:
    if img.has_attr('hidden'):
        return True
    aria_hidden = img.get('aria-hidden')
    if isinstance(aria_hidden, str) and aria_hidden.strip().lower() == 'true':
        return True
    style_attr = img.get('style') or ''
    if isinstance(style_attr, str) and _looks_hidden(style_attr):
        return True
    if _has_hidden_class(img):
        return True
    for dimension in ('width', 'height'):
        value = img.get(dimension)
        try:
            if value is not None and str(value).isdigit() and int(value) == 0:
                return True
        except Exception:
            continue
    return False


def check_images(
    soup: BeautifulSoup,
    base_url: Optional[str] = None,
    *,
    fetch_sizes: bool = False,
    max_fetch: int = 20,
    timeout: int = 10,
) -> Dict:
    images = soup.find_all('img')
    total_found = len(images)
    visible_count = 0
    hidden_skipped = 0
    missing: List[str] = []
    missing_no_title: List[str] = []
    missing_with_title: List[str] = []
    size_info: List[Dict[str, Any]] = []

    session = build_session() if fetch_sizes else None

    for img in images:
        if _is_hidden_image(img):
            hidden_skipped += 1
            continue

        visible_count += 1
        alt = img.get('alt')
        alt_text = (str(alt).strip() if isinstance(alt, str) else None)
        if not alt_text:
            src_val = img.get('src') or img.get('data-src') or img.get('srcset') or ''
            src_str = src_val if isinstance(src_val, str) else str(src_val)
            title_attr = img.get('title')
            has_title = bool(title_attr and str(title_attr).strip())
            if has_title:
                missing.append(src_str)
                missing_with_title.append(src_str)
            else:
                missing_no_title.append(src_str)

        if fetch_sizes and session and base_url and len(size_info) < max_fetch:
            raw_src = img.get('src') or (img.get('data-src') if isinstance(img.get('data-src'), str) else None)
            if raw_src:
                abs_src = urljoin(base_url, raw_src)
                disp = _extract_display_size(img)
                meta = _head_size(session, abs_src, timeout=timeout)
                size_info.append(
                    {
                        'src': abs_src,
                        'display': disp,
                        'bytes': meta.get('bytes'),
                        'type': meta.get('content_type'),
                    }
                )

    if visible_count == 0:
        if total_found == 0:
            status = 'fail'
            message = 'No <img> elements found.'
        else:
            status = 'pass'
            message = f'Skipped {hidden_skipped} hidden or non-visible image(s); nothing visible to audit.'
    else:
        if not missing:
            status = 'pass'
            if len(missing_no_title) > 0:
                message = f'All {visible_count} visible images include alt text or lack a title attribute.'
            else:
                message = f'All {visible_count} visible images include alt text.'
        else:
            status = 'warning'
            message = f'{len(missing)} image(s) have a title attribute but missing alt text.'

    oversized_threshold = 500 * 1024  # 500KB default heuristic
    oversized_count = sum(1 for it in size_info if (it.get('bytes') or 0) > oversized_threshold)

    return {
        'total_found': total_found,
        'total_images': visible_count,
        'missing_alt': missing[:50],
        'missing_alt_no_title': missing_no_title[:50],
        'missing_alt_with_title': missing_with_title[:50],
        'poor_alt': [],
        'status': status,
        'skipped_hidden': hidden_skipped,
        'sizes': size_info,
        'oversized_count': oversized_count,
        'oversized_threshold_bytes': oversized_threshold,
        'message': message,
    }
