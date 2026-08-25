"""Metadata stores (M2): PostgreSQL (relational) + MongoDB (document).

SQL holds structured experiment/run/metric records; NoSQL holds flexible per-image
artifact annotations and model-version documents. Both are import-light and only
touched when you opt in (a DB URL / client), so the core pipeline never requires them.
"""
