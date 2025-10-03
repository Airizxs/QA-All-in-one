from typing import Optional
from bs4 import BeautifulSoup

def check_title_and_meta(soup: BeautifulSoup, keyword: Optional[str] = None) -> dict:
    """
    Checks for the presence and content of the title and meta description.

    Args:
        soup (BeautifulSoup): The parsed HTML content of the page.

    Returns:
        dict: A dictionary with the results of the check.
    """
    results = {
        'title': {'found': False, 'content': None, 'status': 'missing'},
        'meta_description': {'found': False, 'content': None, 'status': 'missing'},
        'author': {'found': False, 'content': None, 'status': 'fail'}
    }

    # Check for the title tag
    title_tag = soup.find('title')
    if title_tag:
        title_text = title_tag.get_text().strip()
        results['title']['found'] = True
        results['title']['content'] = title_text
        results['title']['status'] = 'ok'
        # Updated recommended range 50–60 characters
        if len(title_text) < 50 or len(title_text) > 60:
            results['title']['status'] = 'warning'
            results['title']['message'] = 'Title length should be in the 50–60 character range.'
        # If a keyword is provided, check for presence
        if keyword:
            if keyword.lower() not in title_text.lower():
                # Do not fail, but warn to keep scoring nuanced
                results['title']['status'] = 'warning'
                results['title']['message'] = (
                    results['title'].get('message', '') + (" " if results['title'].get('message') else '') +
                    f'Keyword "{keyword}" not found in title.'
                )

    # Check for the meta description tag
    meta_desc_tag = soup.find('meta', attrs={'name': 'description'})
    if meta_desc_tag and 'content' in meta_desc_tag.attrs:
        meta_desc_content = meta_desc_tag['content'].strip()
        results['meta_description']['found'] = True
        results['meta_description']['content'] = meta_desc_content
        results['meta_description']['status'] = 'ok'
        # Updated recommended range 120–155 characters
        if len(meta_desc_content) < 120 or len(meta_desc_content) > 155:
            results['meta_description']['status'] = 'warning'
            results['meta_description']['message'] = 'Meta description length should be in the 120–155 character range.'
        # Optional keyword presence
        if keyword and keyword.lower() not in meta_desc_content.lower():
            results['meta_description']['status'] = 'warning'
            results['meta_description']['message'] = (
                results['meta_description'].get('message', '') + (" " if results['meta_description'].get('message') else '') +
                f'Keyword "{keyword}" not found in meta description.'
            )

    # Check for the author meta tag
    # Prefer a standard <meta name="author" content="...">
    author_tag = soup.find('meta', attrs={'name': 'author'})
    author_content = None
    if author_tag and 'content' in author_tag.attrs:
        author_content = author_tag['content'].strip()
    else:
        # Fallbacks: common alternatives seen in the wild
        # Open Graph: <meta property="article:author" content="...">
        og_author = soup.find('meta', attrs={'property': 'article:author'})
        if og_author and 'content' in og_author.attrs:
            author_content = og_author['content'].strip()
        else:
            # Sometimes sites use <meta name="byl" content="By Jane Doe">
            byl = soup.find('meta', attrs={'name': 'byl'})
            if byl and 'content' in byl.attrs:
                author_content = byl['content'].strip()

    if author_content:
        results['author']['found'] = True
        results['author']['content'] = author_content
        results['author']['status'] = 'ok'
    else:
        # Explicit fail as requested when no author is found
        results['author']['status'] = 'fail'

    
    return results
