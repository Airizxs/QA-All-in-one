from bs4 import BeautifulSoup

def check_headings(soup: BeautifulSoup) -> dict:
    """
    Checks for the presence and hierarchy of heading tags.

    Args:
        soup (BeautifulSoup): The parsed HTML content of the page.

    Returns:
        dict: A dictionary with the results of the heading check.
    """
    results = {
        'h1_status': 'missing',
        'h1_content': None,
        'h_hierarchy': 'ok',
        'h_tags_found': []
    }

    # Find the H1 tag
    h1_tag = soup.find('h1')
    if h1_tag:
        results['h1_status'] = 'found'
        results['h1_content'] = h1_tag.get_text().strip()
    else:
        results['h1_status'] = 'missing'
        results['h_hierarchy'] = 'error' # If H1 is missing, hierarchy is broken

    # Check for heading hierarchy
    headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    heading_levels = [int(h.name[1]) for h in headings]
    
    results['h_tags_found'] = heading_levels

    # Check if the levels are in a valid sequence (e.g., h1 -> h3 is an error)
    if heading_levels:
        current_level = 0
        for level in heading_levels:
            if level > current_level + 1 and current_level != 0:
                results['h_hierarchy'] = 'warning'
                results['h_hierarchy_message'] = 'Heading levels are not in a strict hierarchical order. (e.g., jumping from h2 to h4).'
                break
            current_level = level

    return results
