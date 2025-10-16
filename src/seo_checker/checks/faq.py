from typing import Dict, List
from bs4 import BeautifulSoup


_FAQ_TEXT_HINTS = (
    'faq',
    'faqs',
    'frequently asked question',
    'frequently asked questions',
)


def _has_faq_hint(text: str) -> bool:
    lower = text.lower()
    return any(hint in lower for hint in _FAQ_TEXT_HINTS)


def check_faq(soup: BeautifulSoup) -> Dict:
    """Heuristic FAQ check: looks for FAQ sections and enforces H3 usage when present."""
    faq_sections = []
    # Sections with class or id containing 'faq'
    for el in soup.find_all(True, attrs={"class": True}):
        classes = " ".join(el.get("class") or [])
        if _has_faq_hint(classes):
            faq_sections.append(el)
    for el in soup.find_all(True, attrs={"id": True}):
        if _has_faq_hint(el.get('id') or ''):
            faq_sections.append(el)

    faq_sections = list(dict.fromkeys(faq_sections))

    h3_count = 0
    h5_count = 0
    h5_texts: List[str] = []
    other_headings: Dict[str, int] = {}
    other_examples: List[str] = []

    heading_targets = list(faq_sections)

    if not heading_targets:
        heading_candidates = soup.find_all(['h2', 'h3', 'h4', 'h5', 'h6'])
        for heading in heading_candidates:
            text = heading.get_text(strip=True)
            if text and _has_faq_hint(text):
                heading_targets.append(heading.parent or heading)

    faq_detected = bool(heading_targets)

    def _collect_counts(node: BeautifulSoup):
        nonlocal h3_count, h5_count
        skipped_heading = False
        for heading in node.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            name = heading.name.lower()
            text = heading.get_text(strip=True)
            if not skipped_heading and name in ('h1', 'h2') and text and _has_faq_hint(text):
                skipped_heading = True
                continue
            if name == 'h3':
                h3_count += 1
                continue
            if name == 'h5':
                h5_count += 1
                if text and len(h5_texts) < 10:
                    h5_texts.append(text)
            other_headings[name] = other_headings.get(name, 0) + 1
            if text and len(other_examples) < 10:
                other_examples.append(f"{name.upper()}: {text}")

    if faq_sections:
        for sec in faq_sections:
            _collect_counts(sec)
    elif heading_targets:
        for target in heading_targets:
            _collect_counts(target)
    else:
        # No FAQ indicators found anywhere; treat as no FAQ
        faq_detected = False

    non_h3_counts = {tag: count for tag, count in other_headings.items() if count > 0}
    non_h3_total = sum(non_h3_counts.values())
    question_total = h3_count + non_h3_total

    if not faq_detected:
        status = 'pass'
        message = 'No FAQ section detected.'
        non_h3_counts = {}
        other_examples = []
    elif question_total == 0:
        status = 'pass'
        message = 'FAQ section detected but no question headings found.'
        non_h3_counts = {}
        other_examples = []
    elif non_h3_total > 0:
        status = 'fail'
        misused = ", ".join(f"{tag.upper()}({count})" for tag, count in sorted(non_h3_counts.items()))
        message = f'FAQ headings should use H3; found {misused}.'
    else:
        status = 'pass'
        message = 'H3 FAQ headings found.'
        non_h3_counts = {}
        other_examples = []

    return {
        'faq_detected': faq_detected,
        'h3_count': h3_count,
        'h5_count': h5_count,
        'h5_examples': h5_texts,
        'non_h3_counts': non_h3_counts,
        'non_h3_examples': other_examples,
        'status': status,
        'message': message,
    }
