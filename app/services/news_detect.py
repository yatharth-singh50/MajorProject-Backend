"""Decides whether a post should be treated as news (and therefore
fact-checked). A post is news if the author ticked the "News" tag, OR it
carries an unmistakable news marker -- #breaking, #news, "breaking news"
(plus the Hindi equivalent). Everything else is an ordinary post and is
deliberately NOT sent through the pipeline: fact-checking every selfie
caption and opinion would be slow, noisy, and wrong."""

import re

_NEWS_RE = re.compile(
    r"(#\s?breaking(\s?news)?\b|#\s?news(alert)?\b|\bbreaking\s+news\b|ब्रेकिंग\s*न्यूज़?|#\s?समाचार)",
    re.IGNORECASE,
)


def has_news_keywords(text: str) -> bool:
    return bool(_NEWS_RE.search(text or ""))
