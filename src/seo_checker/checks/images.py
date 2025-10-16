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
_LOGO_KEYWORDS = {
    'logo',
    'favicon',
    'icon',
    'brandmark',
    'wordmark',
    'lockup',
    'mark',
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


def _is_logo_like(img, alt_text: Optional[str], title_text: Optional[str], src_text: Optional[str]) -> bool:
    candidates: List[str] = []
    if alt_text:
        candidates.append(alt_text.lower())
    if title_text:
        candidates.append(title_text.lower())
    if src_text:
        candidates.append(src_text.lower())
    classes = img.get('class')
    if isinstance(classes, (list, tuple, set)):
        candidates.extend(str(cls).lower() for cls in classes if cls)
    elif isinstance(classes, str):
        candidates.extend(token.lower() for token in classes.split() if token)
    if not candidates:
        return False
    blob = " ".join(candidates)
    return any(keyword in blob for keyword in _LOGO_KEYWORDS)


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
    image_titles: List[str] = []
    content_images_info: List[Dict[str, Optional[str]]] = []
    content_missing_alt: List[str] = []
    content_missing_title: List[str] = []
    content_with_alt_title: List[Dict[str, Optional[str]]] = []
    decorative_skipped = 0

    session = build_session() if fetch_sizes else None

    for img in images:
        if _is_hidden_image(img):
            hidden_skipped += 1
            continue

        visible_count += 1
        alt = img.get('alt')
        alt_text = (str(alt).strip() if isinstance(alt, str) else None)
        title_attr = img.get('title')
        title_text = (str(title_attr).strip() if isinstance(title_attr, str) else None)

        src_val = img.get('src') or img.get('data-src') or img.get('srcset') or ''
        src_str = src_val if isinstance(src_val, str) else str(src_val)
        src_reference = src_str.strip()
        if isinstance(src_reference, str) and ',' in src_reference:
            src_reference = src_reference.split(',')[0].strip()
        if not src_reference:
            src_reference = alt_text or title_text or ''
        if not src_reference:
            src_reference = '[image]'

        is_logo = _is_logo_like(img, alt_text, title_text, src_str if isinstance(src_str, str) else None)

        if not is_logo:
            content_info = {
                'alt': alt_text,
                'title': title_text,
                'src': src_reference,
            }
            content_images_info.append(content_info)
            if alt_text and title_text:
                content_with_alt_title.append(content_info)
            if title_text:
                image_titles.append(title_text)

            if not alt_text:
                missing.append(src_reference)
                if title_text:
                    missing_with_title.append(src_reference)
                else:
                    missing_no_title.append(src_reference)
                content_missing_alt.append(src_reference)
            if not title_text:
                content_missing_title.append(src_reference)
        else:
            decorative_skipped += 1

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
        content_count = len(content_images_info)
        if content_count == 0:
            status = 'warning'
            message = 'Only logo or decorative images detected; add descriptive photos with alt/title.'
        elif not content_missing_alt and not content_missing_title:
            status = 'pass'
            message = f'All {content_count} content images include alt and title text.'
        else:
            status = 'warning'
            parts: List[str] = []
            if content_missing_alt:
                parts.append(f'{len(content_missing_alt)} content image(s) missing alt text')
            if content_missing_title:
                parts.append(f'{len(content_missing_title)} content image(s) missing title text')
            message = '; '.join(parts) + '.'

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
        'content_images': content_images_info[:50],
        'content_images_with_alt_title': content_with_alt_title[:50],
        'content_missing_alt': content_missing_alt[:50],
        'content_missing_title': content_missing_title[:50],
        'content_image_count': len(content_images_info),
        'decorative_images_filtered': decorative_skipped,
        'message': message,
        'image_titles': image_titles[:50],
    }
