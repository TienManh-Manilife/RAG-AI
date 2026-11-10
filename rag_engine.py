"""Backward-compatible facade for the modular rag_engine package.

Existing app.py imports can remain unchanged:
    import rag_engine
    from rag_engine import retrieve_context
"""
from rag_engine import *  # noqa: F401,F403
