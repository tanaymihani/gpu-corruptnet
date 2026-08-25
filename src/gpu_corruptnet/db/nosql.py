"""Document (MongoDB) store for flexible per-image artifact annotations.

NoSQL fits here because annotations are schema-flexible: variable-length artifact
lists, optional bounding regions, per-model version docs. Use ``.connect()`` for a real
MongoDB, or ``.in_memory()`` (mongomock) for dev/tests — same API.

An annotation document looks like:
    {"image_key": "s3://.../frame_0001.png", "split": "train",
     "artifacts": ["screen_tearing", "discoloration"], "severity": 3,
     "regions": [{"artifact": "discoloration", "bbox": [x, y, w, h]}]}
"""

from __future__ import annotations

from typing import Any


class AnnotationStore:
    def __init__(self, client: Any, db_name: str = "gpu_corruptnet") -> None:
        self.client = client
        self.db = client[db_name]
        # Index the most common query field (artifact type).
        self.db.annotations.create_index("artifacts")

    @classmethod
    def in_memory(cls, db_name: str = "gpu_corruptnet") -> AnnotationStore:
        import mongomock

        return cls(mongomock.MongoClient(), db_name)

    @classmethod
    def connect(
        cls, uri: str = "mongodb://localhost:27017", db_name: str = "gpu_corruptnet"
    ) -> AnnotationStore:
        from pymongo import MongoClient

        return cls(MongoClient(uri), db_name)

    def add_annotation(self, doc: dict) -> str:
        return str(self.db.annotations.insert_one(doc).inserted_id)

    def add_many(self, docs: list[dict]) -> int:
        if not docs:
            return 0
        return len(self.db.annotations.insert_many(docs).inserted_ids)

    def find_by_artifact(self, artifact: str) -> list[dict]:
        return list(self.db.annotations.find({"artifacts": artifact}))

    def count(self) -> int:
        return self.db.annotations.count_documents({})

    def artifact_counts(self) -> dict[str, int]:
        """How many annotated frames contain each artifact type (aggregation)."""
        pipeline = [
            {"$unwind": "$artifacts"},
            {"$group": {"_id": "$artifacts", "count": {"$sum": 1}}},
        ]
        return {d["_id"]: d["count"] for d in self.db.annotations.aggregate(pipeline)}

    def register_model(self, doc: dict) -> str:
        return str(self.db.models.insert_one(doc).inserted_id)
