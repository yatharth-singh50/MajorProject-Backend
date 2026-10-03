"""
Evidence source policy -- a preference list of reputable, fact-check-
relevant domains used to rank web-search results before handing them to a
model for stance analysis.

Per the project spec: never invent citations. Every entry that ends up in
`matchedClaims` must carry the actual title/source/URL a search returned --
this module only RANKS real results, it never fabricates or substitutes
one. Extend TRUSTED_DOMAINS freely; an untrusted-domain result is still
real evidence, just lower priority, not excluded.
"""

from urllib.parse import urlparse

TRUSTED_DOMAINS = [
    # International wire services / broadcasters
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "bbc.co.uk",
    "aljazeera.com",
    # Indian national press
    "thehindu.com",
    "indianexpress.com",
    "timesofindia.indiatimes.com",
    "ndtv.com",
    "hindustantimes.com",
    # Indian government / official sources
    "pib.gov.in",
    "rbi.org.in",
    "gov.in",
    "nic.in",
    # International institutions
    "who.int",
    "nasa.gov",
    "un.org",
    # Dedicated fact-checkers
    "factcheck.afp.com",
    "boomlive.in",
    "altnews.in",
    "factchecker.in",
    "politifact.com",
    "snopes.com",
]


def domain_of(url: str) -> str:
    """Returns the registrable-ish domain (netloc minus a leading www.)."""
    try:
        netloc = urlparse(url).netloc.lower()
    except ValueError:
        return ""
    return netloc[4:] if netloc.startswith("www.") else netloc


def is_trusted(url: str) -> bool:
    domain = domain_of(url)
    if not domain:
        return False
    return any(domain == d or domain.endswith(f".{d}") for d in TRUSTED_DOMAINS)


def rank_results(results: list[dict]) -> list[dict]:
    """Stable-sorts so trusted-domain sources come first. Nothing is
    dropped -- an untrusted source is still real evidence, just lower
    priority than, say, Reuters or PIB."""
    return sorted(results, key=lambda r: 0 if is_trusted(r.get("url", "")) else 1)
