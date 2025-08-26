from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional, Tuple

import orjson

from .tristore import TriStore


IMMUTABLE_IGNORED_KEYS = {"created_at", "updated_at"}


def _strip_volatile(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if k not in IMMUTABLE_IGNORED_KEYS}


def _canonical_bytes(flow: Dict[str, Any]) -> bytes:
    clean = _strip_volatile(flow)
    return orjson.dumps(clean, option=orjson.OPT_SORT_KEYS)


def compute_flow_hash(flow: Dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(flow)).hexdigest()


def publish_flow(
    flow: Dict[str, Any],
    store: TriStore,
    *,
    name: Optional[str] = None,
    semver: Optional[str] = None,
) -> Dict[str, Any]:
    # Inject name/semver if provided, else require present in flow
    if name:
        flow["name"] = name
    if semver:
        flow["semver"] = semver
    if "name" not in flow or "semver" not in flow:
        raise ValueError("Flow must include 'name' and 'semver'.")

    # Compute and attach hash (not included in hash itself)
    flow_copy = dict(flow)
    flow_hash = compute_flow_hash(flow_copy)
    flow_copy["flow_hash"] = flow_hash

    # Check immutability: if exists, must be identical (ignoring volatile keys and flow_hash)
    exists = False
    try:
        stored = store.get_flow(flow_copy["name"], flow_copy["semver"]).copy()
        exists = True
    except KeyError:
        stored = None

    if exists:
        stored_clean = _strip_volatile({k: v for k, v in stored.items() if k != "flow_hash"})
        local_clean = _strip_volatile({k: v for k, v in flow.items() if k != "flow_hash"})
        if stored_clean != local_clean:
            raise ValueError(
                f"Flow {flow['name']}@{flow['semver']} already exists with a different spec. "
                f"Semver is immutable; bump the version."
            )
        # Idempotent publish: ensure graph nodes/edges exist but don't replace spec
        store.create_flow(stored)
        steps_count = len(stored.get("steps", []))
        trans_count = len(stored.get("transitions", []))
        return {
            "flow_id": f"{flow['name']}@{flow['semver']}",
            "flow_hash": stored.get("flow_hash") or compute_flow_hash(stored),
            "status": "exists",
            "steps": steps_count,
            "transitions": trans_count,
        }

    # New publish
    store.create_flow(flow_copy)
    steps_count = len(flow_copy.get("steps", []))
    trans_count = len(flow_copy.get("transitions", []))
    return {
        "flow_id": f"{flow_copy['name']}@{flow_copy['semver']}",
        "flow_hash": flow_hash,
        "status": "created",
        "steps": steps_count,
        "transitions": trans_count,
    }

