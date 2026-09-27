"""Immutable amendment chain for accepted events.

An amendment is a *full replacement* of an already accepted event's
mutable body (``occurred_at``, ``event_type``, ``schema_version`` and
``payload``); ``device_id`` and ``event_id`` can never change. Every
amendment is addressed by a client-chosen ``amendment_id`` and must carry
the ``base_revision`` the client built on. Revisions are numbered from
1: the original ingestion record is revision 1 and stays in the
``events`` table untouched -- amendment rows only record the replacement
bodies, so every derived result (sessions, state-machine projections,
alarm progress, stream bindings) and the transparency Merkle receipt keep
referring to the original ingestion record forever.

The first accepted amendment freezes the original body into the
``event_amendments`` table as the revision-1 row (a full copy with no
foreign key to ``events``), so the chain -- original plus every
replacement, ordered by revision -- stays readable even after retention
cleanup has deleted the event body.

Concurrency is optimistic and enforced inside the caller's
``BEGIN IMMEDIATE`` transaction: the ``base_revision`` must equal the
chain's current revision, so two branches racing from the same base
serialize on the single writer lock and exactly one succeeds; the loser
sees the advanced revision and receives :class:`AmendmentConflict`
(HTTP 409) with nothing stored. The same ``amendment_id`` with the same
full content is an idempotent replay; the same id with changed content
(or a changed base) is a permanent ``409``.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .ingestion import ContractResolver, validate_event_content
from .sessions import from_storage, to_storage

# Columns read for one stored revision; revision 1 (the frozen original)
# has NULL amendment_id/base_revision.
REVISION_COLUMNS = (
    "revision, amendment_id, base_revision, device_id, event_id, occurred_at, "
    "event_type, schema_version, payload, created_at"
)


class AmendmentError(Exception):
    """Base class for amendment-domain failures."""


class UnknownEvent(AmendmentError):
    """The event_id has no ingestion record and no frozen chain (HTTP 404)."""


class AmendmentConflict(AmendmentError):
    """The amendment cannot be accepted (HTTP 409 semantics).

    ``detail`` is the exact response detail payload -- a plain string or a
    structured object -- that the route re-raises as a ``409`` after
    rolling back the transaction.
    """

    def __init__(self, detail: Any):
        super().__init__(detail if isinstance(detail, str) else detail.get("message"))
        self.detail = detail


@dataclass(frozen=True)
class Revision:
    """One entry of the amendment chain in revision order."""

    revision: int
    amendment_id: str | None  # NULL for revision 1 (the original ingestion)
    base_revision: int | None  # NULL for revision 1
    occurred_at: datetime
    event_type: str | None
    schema_version: str | None
    payload: Any  # parsed JSON, None for payload-less revisions
    created_at: datetime
    frozen_at: datetime | None  # set when the original body was first frozen


@dataclass(frozen=True)
class AmendmentChain:
    """The complete chain: revision 1 plus every replacement."""

    device_id: str
    event_id: str
    current_revision: int
    frozen: bool
    frozen_at: datetime | None
    revisions: tuple[Revision, ...]

    @property
    def original(self) -> Revision:
        return self.revisions[0]

    @property
    def current(self) -> Revision:
        return self.revisions[-1]


@dataclass(frozen=True)
class AmendmentOutcome:
    """Result of one amendment write, ready to map onto the response."""

    replay: bool
    chain: AmendmentChain
    revision: int  # the revision created/replayed by this request


# ----------------------------------------------------------------- helpers


def _revision_from_row(row: sqlite3.Row, frozen_at: datetime | None) -> Revision:
    revision_no = int(row["revision"])
    return Revision(
        revision=revision_no,
        amendment_id=row["amendment_id"],
        base_revision=(
            int(row["base_revision"]) if row["base_revision"] is not None else None
        ),
        occurred_at=from_storage(row["occurred_at"]),
        event_type=row["event_type"],
        schema_version=row["schema_version"],
        payload=json.loads(row["payload"]) if row["payload"] is not None else None,
        created_at=from_storage(row["created_at"]),
        # Revision 1's created_at is recorded when its body is first
        # frozen (NULL while the event has no amendment yet).
        frozen_at=frozen_at if revision_no == 1 else None,
    )


def load_chain(conn: sqlite3.Connection, event_id: str) -> AmendmentChain | None:
    """Load the full amendment chain for an event.

    With at least one amendment the revision-1 frozen copy and every
    replacement live in ``event_amendments`` (surviving retention
    cleanup); before the first amendment the single revision is built
    from the live ``events`` row. Returns ``None`` when neither exists.
    """
    rows = conn.execute(
        f"SELECT {REVISION_COLUMNS} FROM event_amendments WHERE event_id = ? "
        "ORDER BY revision",
        (event_id,),
    ).fetchall()
    if rows:
        frozen_row = rows[0]
        frozen_at = from_storage(frozen_row["created_at"])
        revisions = tuple(_revision_from_row(row, frozen_at) for row in rows)
        return AmendmentChain(
            device_id=rows[-1]["device_id"],
            event_id=event_id,
            current_revision=len(rows),
            frozen=True,
            frozen_at=frozen_at,
            revisions=revisions,
        )

    row = conn.execute(
        "SELECT device_id, occurred_at, event_type, schema_version, payload "
        "FROM events WHERE event_id = ?",
        (event_id,),
    ).fetchone()
    if row is None:
        return None
    occurred = from_storage(row["occurred_at"])
    single = Revision(
        revision=1,
        amendment_id=None,
        base_revision=None,
        occurred_at=occurred,
        event_type=row["event_type"],
        schema_version=row["schema_version"],
        payload=json.loads(row["payload"]) if row["payload"] is not None else None,
        created_at=occurred,
        frozen_at=None,
    )
    return AmendmentChain(
        device_id=row["device_id"],
        event_id=event_id,
        current_revision=1,
        frozen=False,
        frozen_at=None,
        revisions=(single,),
    )


def submit_amendment(
    conn: sqlite3.Connection,
    *,
    contract_resolver: ContractResolver,
    event_id: str,
    amendment_id: str,
    base_revision: int,
    occurred_at: datetime,
    event_type: str | None = None,
    schema_version: str | None = None,
    payload: Any = None,
    payload_provided: bool = False,
    now: datetime | None = None,
) -> AmendmentOutcome:
    """Validate and append one amendment inside the caller's write transaction.

    The caller must have begun a ``BEGIN IMMEDIATE`` transaction on
    ``conn`` and commits (or rolls back) afterwards: the optimistic
    base-revision check, the first-time freeze of the original body and
    the replacement insert are covered by that single transaction, so a
    stale branch never leaves a row behind.

    Raises:
        ContractError: the contract triplet is incomplete/unknown or the
            payload fails validation (mapped to 422 by the route).
        UnknownEvent: the event has no ingestion record (404).
        AmendmentConflict: the amendment_id exists with different
            content/base, or the base_revision is stale (409); nothing is
            persisted.
    """
    if base_revision < 1:
        raise AmendmentConflict(
            {
                "message": "base_revision must be >= 1",
                "reason": "invalid_base_revision",
                "base_revision": base_revision,
            }
        )

    # Same validation rules as first-time ingestion: all-or-nothing
    # contract triplet, registered schema, JSON-Schema payload.
    _occurred, stored, event_type, schema_version, payload, stored_payload = (
        validate_event_content(
            conn,
            contract_resolver=contract_resolver,
            occurred_at=occurred_at,
            event_type=event_type,
            schema_version=schema_version,
            payload=payload,
            payload_provided=payload_provided,
        )
    )
    if now is None:
        now = datetime.now(timezone.utc)

    existing = conn.execute(
        f"SELECT {REVISION_COLUMNS} FROM event_amendments WHERE event_id = ? "
        "AND amendment_id = ?",
        (event_id, amendment_id),
    ).fetchone()
    if existing is not None:
        same_content = (
            int(existing["base_revision"]) == base_revision
            and existing["occurred_at"] == stored
            and (existing["event_type"] or None) == event_type
            and (existing["schema_version"] or None) == schema_version
            and (existing["payload"] or None) == stored_payload
        )
        if same_content:
            # Idempotent replay of the exact same amendment request.
            chain = load_chain(conn, event_id)
            assert chain is not None
            return AmendmentOutcome(
                replay=True,
                chain=chain,
                revision=int(existing["revision"]),
            )
        raise AmendmentConflict(
            {
                "message": (
                    f"amendment_id {amendment_id!r} already exists for event "
                    f"{event_id!r} with different content"
                ),
                "reason": "amendment_id_conflict",
                "amendment_id": amendment_id,
            }
        )

    stored_rows = conn.execute(
        f"SELECT {REVISION_COLUMNS} FROM event_amendments WHERE event_id = ? "
        "ORDER BY revision",
        (event_id,),
    ).fetchall()
    if stored_rows:
        device_id = stored_rows[-1]["device_id"]
        current_revision = len(stored_rows)
    else:
        # First amendment: the source of truth is still the live
        # ingestion row. Freeze its full body as revision 1 in this same
        # transaction before appending the replacement.
        source = conn.execute(
            "SELECT device_id, occurred_at, event_type, schema_version, payload "
            "FROM events WHERE event_id = ?",
            (event_id,),
        ).fetchone()
        if source is None:
            raise UnknownEvent(event_id)
        device_id = source["device_id"]
        current_revision = 1
        conn.execute(
            "INSERT INTO event_amendments (revision, amendment_id, base_revision, "
            "device_id, event_id, occurred_at, event_type, schema_version, payload, "
            "created_at) VALUES (1, NULL, NULL, ?, ?, ?, ?, ?, ?, ?)",
            (
                device_id,
                event_id,
                source["occurred_at"],
                source["event_type"],
                source["schema_version"],
                source["payload"],
                to_storage(now),
            ),
        )

    # Optimistic concurrency: the client must build on the current head.
    # Writers serialize on the single writer lock, so only one branch
    # per base revision can reach the insert.
    if base_revision != current_revision:
        raise AmendmentConflict(
            {
                "message": (
                    f"amendment is based on revision {base_revision} but the "
                    f"event is already at revision {current_revision}; refetch "
                    "the current content and retry"
                ),
                "reason": "stale_base_revision",
                "amendment_id": amendment_id,
                "base_revision": base_revision,
                "current_revision": current_revision,
            }
        )

    new_revision = current_revision + 1
    conn.execute(
        "INSERT INTO event_amendments (revision, amendment_id, base_revision, "
        "device_id, event_id, occurred_at, event_type, schema_version, payload, "
        "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            new_revision,
            amendment_id,
            base_revision,
            device_id,
            event_id,
            stored,
            event_type,
            schema_version,
            stored_payload,
            to_storage(now),
        ),
    )
    chain = load_chain(conn, event_id)
    assert chain is not None
    return AmendmentOutcome(replay=False, chain=chain, revision=new_revision)
