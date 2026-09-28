"""Immutable amendment chain for accepted events.

An event is ingested once (``POST /events`` or the stream ledger path) at
``revision`` 1. Amendments never delete or overwrite that history: every
accepted correction appends one immutable row to ``event_amendments`` and
the live ``events`` row is updated in place, so the amendment history and
the current body always agree.

* ``revision`` 1 is the **original ingestion record**. It is frozen
  lazily -- copied verbatim from ``events`` inside the first amendment's
  ``BEGIN IMMEDIATE`` transaction -- and once written it can never change.
* Every later revision is one amendment, identified by a client-chosen
  ``amendment_id`` (unique per ``event_id``) and carrying the
  ``base_revision`` the client started from. ``device_id`` and
  ``event_id`` are immutable; ``occurred_at``, ``event_type``,
  ``schema_version`` and ``payload`` are replaced wholesale, with the
  payload validated against its JSON Schema exactly like first-time
  ingestion.
* Optimistic concurrency: the amendment's ``base_revision`` must equal the
  current revision *inside* the write transaction. Two concurrent branches
  serialize on the single writer lock; exactly one wins and the stale one
  gets :class:`AmendmentConflict` (409) with nothing stored. Re-POSTing the
  same ``amendment_id`` with the identical content is an idempotent replay;
  reusing the id with different content is a 409.

The chain deliberately has no foreign key to ``events``: retention cleanup
deletes only event bodies, while the frozen revisions -- including the
original -- stay readable forever. Existing derived results and the
transparency Merkle receipt continue to be based on the *original*
ingestion record: amendments neither append receipt leaves nor project
state-machine transitions nor recompute alarms (see module ``receipts`` and
``ingestion``).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .ingestion import prepare_event_content
from .sessions import Event, from_storage, normalize_ts, to_storage

# Full column list; every SELECT in this module reads this stable shape.
REVISION_COLUMNS = (
    "revision, amendment_id, base_revision, device_id, event_id, "
    "occurred_at, event_type, schema_version, payload, created_at"
)


class AmendmentConflict(Exception):
    """The amendment cannot be applied (HTTP 409 semantics); nothing persists.

    ``detail`` is the exact response detail payload -- a plain string or a
    structured object -- that the route re-raises as ``409`` after rolling
    back the transaction.
    """

    def __init__(self, detail: Any):
        super().__init__(detail if isinstance(detail, str) else detail.get("message"))
        self.detail = detail


class EventNotFound(LookupError):
    """The target event has no live ``events`` row (HTTP 404 semantics).

    Distinct from the built-in ``LookupError`` family (``IndexError`` /
    ``KeyError``) so an internal programming error is never misreported to
    a client as a missing event.
    """


@dataclass(frozen=True)
class StoredRevision:
    """One revision of an event as persisted in ``event_amendments``."""

    revision: int
    amendment_id: str | None
    base_revision: int | None
    device_id: str
    event_id: str
    occurred_at: str  # UTC ISO-8601 storage form
    event_type: str | None
    schema_version: str | None
    payload: str | None  # canonical JSON text
    created_at: str | None

    @property
    def is_amendment(self) -> bool:
        """False only for the frozen revision-1 ingestion record."""
        return self.revision > 1

    def to_event(self) -> Event:
        return Event(
            device_id=self.device_id,
            event_id=self.event_id,
            occurred_at=from_storage(self.occurred_at),
            event_type=self.event_type,
            schema_version=self.schema_version,
            payload=(
                json.loads(self.payload) if self.payload is not None else None
            ),
        )


@dataclass(frozen=True)
class AmendmentHistory:
    """The full chain of one ``event_id``.

    ``revisions`` is ordered by revision (the original first, the current
    body last). For an event that has never been amended the history is
    synthesized from the live ``events`` row: ``original`` and ``current``
    are that body and ``amendments`` is empty.
    """

    event_id: str
    original: StoredRevision
    current: StoredRevision
    revisions: tuple[StoredRevision, ...]

    @property
    def current_revision(self) -> int:
        return self.current.revision

    @property
    def amendments(self) -> tuple[StoredRevision, ...]:
        return tuple(r for r in self.revisions if r.is_amendment)

    @property
    def frozen(self) -> bool:
        """True once revision 1 has been copied into ``event_amendments``."""
        return self.current_revision > 1


@dataclass(frozen=True)
class AmendmentResult:
    """Result of one accepted amendment, ready to map onto the response."""

    replay: bool
    created: bool
    history: AmendmentHistory


# Same shape as ingestion.ContractResolver: resolve the registered schema
# document for (event_type, version) or None. The app factory injects the
# resolver backed by its parsed-schema cache.
ContractResolver = Callable[[sqlite3.Connection, str, str], dict[str, Any] | None]


def _row_to_revision(row: sqlite3.Row) -> StoredRevision:
    return StoredRevision(
        revision=int(row["revision"]),
        amendment_id=row["amendment_id"],
        base_revision=(
            int(row["base_revision"]) if row["base_revision"] is not None else None
        ),
        device_id=row["device_id"],
        event_id=row["event_id"],
        occurred_at=row["occurred_at"],
        event_type=row["event_type"],
        schema_version=row["schema_version"],
        payload=row["payload"],
        created_at=row["created_at"],
    )


def _live_revision(row: sqlite3.Row) -> StoredRevision:
    """Synthesize a revision-1 view from a live ``events`` row.

    Used while the event has never been amended: ``created_at`` is unknown
    (the ingestion path stores no per-event timestamp), so it is reported as
    null; the first amendment's freeze replaces this view with a persisted
    row carrying the freeze timestamp.
    """
    return StoredRevision(
        revision=1,
        amendment_id=None,
        base_revision=None,
        device_id=row["device_id"],
        event_id=row["event_id"],
        occurred_at=row["occurred_at"],
        event_type=row["event_type"],
        schema_version=row["schema_version"],
        payload=row["payload"],
        created_at=None,
    )


def list_revision_rows(
    conn: sqlite3.Connection, event_id: str
) -> list[StoredRevision]:
    """All frozen revisions of an event ordered by revision (may be empty)."""
    rows = conn.execute(
        f"SELECT {REVISION_COLUMNS} FROM event_amendments "
        "WHERE event_id = ? ORDER BY revision",
        (event_id,),
    ).fetchall()
    return [_row_to_revision(row) for row in rows]


def get_history(conn: sqlite3.Connection, event_id: str) -> AmendmentHistory | None:
    """Return the full amendment chain of an event, or None when unknown.

    Prefers the persisted chain. When the event has never been amended the
    chain is synthesized from the live ``events`` row, so every ingested
    event -- even one predating this module -- is readable through the same
    shape. After retention cleanup removed the live body, a chain still
    frozen (at least one amendment ever applied) remains fully readable.
    """
    revisions = list_revision_rows(conn, event_id)
    if revisions:
        return AmendmentHistory(
            event_id=event_id,
            original=revisions[0],
            current=revisions[-1],
            revisions=tuple(revisions),
        )
    live = conn.execute(
        "SELECT event_id, device_id, occurred_at, event_type, "
        "schema_version, payload FROM events WHERE event_id = ?",
        (event_id,),
    ).fetchone()
    if live is None:
        return None
    original = _live_revision(live)
    return AmendmentHistory(
        event_id=event_id,
        original=original,
        current=original,
        revisions=(original,),
    )


def apply_amendment(
    conn: sqlite3.Connection,
    contract_resolver: ContractResolver,
    *,
    event_id: str,
    amendment_id: str,
    base_revision: int,
    device_id: str,
    occurred_at: datetime,
    event_type: str | None,
    schema_version: str | None,
    payload: Any,
    payload_provided: bool,
    now: datetime | None = None,
) -> AmendmentResult:
    """Apply one amendment inside the caller's open ``BEGIN IMMEDIATE``.

    The route must have acquired the write lock before calling (the base
    revision check is only sound under it) and commits/rolls back
    afterwards; this function performs no transaction control itself.

    Raises:
        EventNotFound: the event is unknown (including a body removed by
            retention cleanup before any amendment froze it); the route
            maps it to 404.
        ContractError: the contract triplet is incomplete, the contract
            unknown, or the payload invalid (mapped to 422).
        AmendmentConflict: a wrong/immutable device_id, a stale
            ``base_revision``, or a reused ``amendment_id`` with different
            content (mapped to 409); nothing is persisted.
    """
    live = conn.execute(
        "SELECT seq, event_id, device_id, occurred_at, event_type, "
        "schema_version, payload FROM events WHERE event_id = ?",
        (event_id,),
    ).fetchone()

    revisions = list_revision_rows(conn, event_id)
    existing_amendment = next(
        (row for row in revisions if row.amendment_id == amendment_id), None
    )

    if live is None:
        # The live body was removed by retention cleanup. New amendments
        # are impossible (there is nothing to replace), but an exact
        # replay of an already-stored amendment still succeeds -- like the
        # ingestion replay of a cleaned-up event -- because the frozen
        # chain retains the amendment's complete content.
        if existing_amendment is not None:
            same_content = (
                existing_amendment.base_revision == base_revision
                and existing_amendment.device_id == device_id
                and existing_amendment.occurred_at == to_storage(occurred_at)
                and (existing_amendment.event_type or None) == event_type
                and (existing_amendment.schema_version or None) == schema_version
                and (existing_amendment.payload or None)
                == prepare_event_content(
                    conn,
                    contract_resolver,
                    event_type=event_type,
                    schema_version=schema_version,
                    payload=payload,
                    payload_provided=payload_provided,
                )
            )
            if same_content:
                history = get_history(conn, event_id)
                assert history is not None
                return AmendmentResult(replay=True, created=False, history=history)
        raise EventNotFound(event_id)

    if live["device_id"] != device_id:
        raise AmendmentConflict(
            {
                "message": (
                    "device_id is immutable and must match the ingested event"
                ),
                "reason": "device_id_immutable",
                "stored_device_id": live["device_id"],
                "submitted_device_id": device_id,
            }
        )

    # Contract/payload validation before any write, identical to the
    # first-time ingestion path.
    stored_payload = prepare_event_content(
        conn,
        contract_resolver,
        event_type=event_type,
        schema_version=schema_version,
        payload=payload,
        payload_provided=payload_provided,
    )

    occurred = normalize_ts(occurred_at)
    stored_occurred = to_storage(occurred)
    if now is None:
        now = datetime.now(timezone.utc)
    frozen_at = to_storage(now)

    current = revisions[-1] if revisions else _live_revision(live)

    if existing_amendment is not None:
        # Idempotent replay only when the whole submission is identical.
        same_content = (
            existing_amendment.base_revision == base_revision
            and existing_amendment.occurred_at == stored_occurred
            and (existing_amendment.event_type or None) == event_type
            and (existing_amendment.schema_version or None) == schema_version
            and (existing_amendment.payload or None) == stored_payload
        )
        if not same_content:
            raise AmendmentConflict(
                {
                    "message": (
                        f"amendment_id {amendment_id!r} already exists for "
                        f"event {event_id!r} with different content"
                    ),
                    "reason": "amendment_id_content_conflict",
                    "amendment_id": amendment_id,
                    "existing_revision": existing_amendment.revision,
                }
            )
        # Re-fetch the history (the live body might theoretically differ
        # only if invariants were broken; in practice the chain is intact).
        history = get_history(conn, event_id)
        assert history is not None
        return AmendmentResult(replay=True, created=False, history=history)

    # The optimistic-concurrency check is the heart of the chain: under
    # the write lock the current revision cannot move between this read
    # and the insert below, so exactly one of two concurrent branches can
    # see its own base_revision as current.
    if base_revision != current.revision:
        raise AmendmentConflict(
            {
                "message": (
                    f"base_revision {base_revision} is stale: the current "
                    f"revision of event {event_id!r} is {current.revision}; "
                    "re-read the amendment history and retry against it"
                ),
                "reason": "base_revision_stale",
                "base_revision": base_revision,
                "current_revision": current.revision,
            }
        )

    new_revision = current.revision + 1
    try:
        if not revisions:
            # Freeze the original ingestion body verbatim on the very first
            # amendment. revision 1 carries no amendment_id/base_revision.
            conn.execute(
                "INSERT INTO event_amendments (revision, amendment_id, "
                "base_revision, device_id, event_id, occurred_at, event_type, "
                "schema_version, payload, created_at) "
                "VALUES (1, NULL, NULL, ?, ?, ?, ?, ?, ?, ?)",
                (
                    live["device_id"],
                    event_id,
                    live["occurred_at"],
                    live["event_type"],
                    live["schema_version"],
                    live["payload"],
                    frozen_at,
                ),
            )
        conn.execute(
            "INSERT INTO event_amendments (revision, amendment_id, "
            "base_revision, device_id, event_id, occurred_at, event_type, "
            "schema_version, payload, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                new_revision,
                amendment_id,
                base_revision,
                device_id,
                event_id,
                stored_occurred,
                event_type,
                schema_version,
                stored_payload,
                frozen_at,
            ),
        )
    except sqlite3.IntegrityError:
        # Defensive: under BEGIN IMMEDIATE the single writer lock
        # serializes branches, so a stale base is normally caught by the
        # revision check above. If a UNIQUE (event_id, revision) or
        # UNIQUE (event_id, amendment_id) is nevertheless hit -- e.g. a
        # future caller that read outside the write lock -- resolve it as
        # the same 409 instead of surfacing a raw IntegrityError (500).
        # The statement failure does not abort the transaction in SQLite,
        # so the reads below still run.
        pass
    else:
        # Replace only the mutable body columns; the persistent events.seq
        # is deliberately untouched (arrival order, receipts, state-machine
        # and alert high-waters all keep keying off the original ingestion).
        conn.execute(
            "UPDATE events SET occurred_at = ?, event_type = ?, "
            "schema_version = ?, payload = ? WHERE event_id = ? AND device_id = ?",
            (stored_occurred, event_type, schema_version, stored_payload,
             event_id, device_id),
        )
        history = get_history(conn, event_id)
        assert history is not None
        return AmendmentResult(replay=False, created=True, history=history)

    # Lost the race: report the committed chain's latest revision.
    latest_id = conn.execute(
        "SELECT amendment_id, base_revision FROM event_amendments "
        "WHERE event_id = ? AND revision = ?",
        (event_id, new_revision),
    ).fetchone()
    if latest_id is not None and latest_id["amendment_id"] == amendment_id:
        # The same amendment_id won the revision slot; let a repeated
        # identical request resolve as replay on the caller's next attempt
        # rather than silently misclassifying it.
        conflict_reason = "amendment_id_content_conflict"
    else:
        conflict_reason = "base_revision_stale"
    committed_revision = conn.execute(
        "SELECT COALESCE(MAX(revision), 1) AS m FROM event_amendments "
        "WHERE event_id = ?",
        (event_id,),
    ).fetchone()["m"]
    raise AmendmentConflict(
        {
            "message": (
                f"base_revision {base_revision} is stale: a concurrent "
                f"amendment advanced event {event_id!r} to revision "
                f"{committed_revision}; re-read the amendment history and "
                "retry against it"
            ),
            "reason": conflict_reason,
            "base_revision": base_revision,
            "current_revision": committed_revision,
        }
    )
