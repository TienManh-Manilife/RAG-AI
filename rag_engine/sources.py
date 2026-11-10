"""Inspect and manage indexed document sources."""
from .database import get_collection, clear_collection

def get_collection_stats() -> dict[str, int]:
    collection = get_collection()
    return {"chunks": collection.count(), "sources": len(list_indexed_sources())}

def list_indexed_sources() -> list[dict]:
    collection = get_collection()
    if collection.count() == 0:
        return []
    data = collection.get(include=["metadatas"])
    grouped = {}
    for meta in data.get("metadatas") or []:
        if not meta:
            continue
        source_id = meta.get("source_id", "")
        item = grouped.setdefault(source_id, {
            "source_id": source_id, "source": meta.get("source", "Unknown"),
            "chunks": 0, "pages": set(),
        })
        item["chunks"] += 1
        item["pages"].add(meta.get("page"))
    return [
        {
            "source_id": item["source_id"], "source": item["source"],
            "chunks": item["chunks"], "pages": len(item["pages"]),
        }
        for item in sorted(grouped.values(), key=lambda x: x["source"].lower())
    ]

def delete_source(source_id: str) -> None:
    get_collection().delete(where={"source_id": source_id})
