"""Tells the models what "now" is.

LLMs have a training cutoff and no clock: left alone they treat their own
last-known date as the present, and will happily "fact-check" a story about
something that happened last week as if it can't have. Every reasoning
prompt gets this line prepended so the model knows today's date and that
the web results it's shown are newer than anything it memorized."""

from datetime import datetime, timezone

from app.core.config import get_settings

settings = get_settings()


def _now() -> datetime:
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo(settings.APP_TIMEZONE))
    except Exception:  # noqa: BLE001 - missing tzdata / bad zone name -> UTC
        return datetime.now(timezone.utc)


def now_context() -> str:
    now = _now()
    stamp = now.strftime("%A, %d %B %Y, %H:%M %Z").strip()
    return (
        f"CURRENT DATE AND TIME: {stamp}.\n"
        "Today's date is the one above, NOT the date your training data ended. Web search results "
        "you are shown are live and may describe events newer than anything you know -- never "
        "dismiss a claim just because you don't recognise it, and never rely on memory over the "
        "results provided."
    )


def today_label() -> str:
    """e.g. 'October 2026' -- for steering search queries toward recent coverage."""
    return _now().strftime("%B %Y")
