"""
app_factory.py
==============
Compatibility wrapper around the canonical `app.py` entry point.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def create_app():
    """Return the canonical Flask app from `app.py`."""
    from app import app

    return app


if __name__ == "__main__":
    from app import app
    from config import PORT

    app.run(port=PORT)
