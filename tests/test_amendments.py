"""Tests for the immutable event amendment chain.

Covers the HTTP contract of ``POST/GET /events/{event_id}/amendments``:

* revision 1 is the original ingestion, frozen verbatim on the first
  amendment; later amendments append revisions 2, 3, ...;
* device_id/event_id are immutable, the other body fields are replaced
  wholesale with JSON-Schema validation;
* optimistic concurrency: stale base_revision -> 409, nothing stored,
  exactly one of two concurrent branches wins;
* amendment_id idempotency (identical replay 200) and conflicts (409);
* GET returns original, revision-ordered history and the current body;
* the chain stays readable after retention cleanup deletes the live body;
* transparency receipts and already-computed derived results keep using
  the original ingestion record;
* legacy databases (created before this module) work without migration.
"""

import sqlite3
import threading
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app import db as db_module
from app.amendments import apply_amendment
from app.ingestion import canonical_json
from app.main import create_app

TELEMETRY_SCHEMA = {
    "type": "object",
    "required": ["value"],
    "properties": {
        "value": {"type": "integer", "minimum": 0},
        "unit": {"type": "string"},
    },
    "additionalProperties": False,
}


@pytest.fixture()
def client(tmp_path):
    app = create_app(database_path=str(tmp_path / "test.db"), gap_seconds=300)
    with TestClient(app) as c:
        yield c


def register_schema(client, version="1.0", schema=None, supersedes=None):
    body = {"schema": schema or TELEMETRY_SCHEMA}
    if supersedes is not None:
        body["supersedes"] = supersedes
    resp = client.put(f"/schemas/telemetry/{version}", json=body)
    assert resp.status_code in (200, 201), resp.text
    return resp


def ingest(
    client,
    event_id="e1",
    device_id="dev-1",
    occurred_at="2026-09-20T10:00:00Z",
    payload=None,
):
    body = {
        "device_id": device_id,
        "event_id": event_id,
        "occurred_at": occurred_at,
    }
    if payload is not None:
        body.update(
            {
                "event_type": "telemetry",
                "schema_version": "1.0",
                "payload": payload,
            }
        )
    resp = client.post("/events", json=body)
    assert resp.status_code in (200, 201), resp.text
    return resp


def amend(
    client,
    event_id,
    amendment_id,
    base_revision,
    *,
    device_id="dev-1",
    occurred_at="2026-09-20T10:05:00Z",
    payload,
):
    body = {
        "device_id": device_id,
        "event_id": event_id,
        "occurred_at": occurred_at,
        "amendment_id": amendment_id,
        "base_revision": base_revision,
    }
    if payload is not None:
        body.update(
            {
                "event_type": "telemetry",
                "schema_version": "1.0",
                "payload": payload,
            }
        )
    return client.post(f"/events/{event_id}/amendments", json=body)


# ------------------------------------------------------------------- basics


def test_get_history_before_any_amendment(client):
    register_schema(client)
    ingest(client, payload={"value": 1, "unit": "c"})

    resp = client.get("/events/e1/amendments")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["current_revision"] == 1
    assert data["history"] == []
    assert data["original"]["revision"] == 1
    assert data["original"]["amendment_id"] is None
    assert data["original"]["base_revision"] is None
    assert data["original"]["created_at"] is None
    assert data["original"]["payload"] == {"value": 1, "unit": "c"}
    assert data["current"] == data["original"]


def test_first_amendment_freezes_original_and_appends_revision_2(client):
    register_schema(client)
    ingest(client, occurred_at="2026-09-20T10:00:00Z", payload={"value": 1})

    resp = amend(
        client,
        "e1",
        "fix-1",
        1,
        occurred_at="2026-09-20T11:00:00Z",
        payload={"value": 42, "unit": "c"},
    )
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["replay"] is False
    assert data["revision"] == 2
    assert data["event"]["occurred_at"].startswith("2026-09-20T11:00:00")
    assert data["event"]["payload"] == {"value": 42, "unit": "c"}

    chain = data["amendments"]
    assert chain["current_revision"] == 2
    original = chain["original"]
    assert original["revision"] == 1
    assert original["amendment_id"] is None
    assert original["occurred_at"].startswith("2026-09-20T10:00:00")
    assert original["payload"] == {"value": 1}
    assert original["created_at"] is not None  # frozen on first amendment

    assert len(chain["history"]) == 1
    entry = chain["history"][0]
    assert entry["revision"] == 2
    assert entry["amendment_id"] == "fix-1"
    assert entry["base_revision"] == 1
    assert entry["payload"] == {"value": 42, "unit": "c"}
    assert chain["current"] == entry


def test_amendment_chain_grows_and_live_events_row_follows(client):
    register_schema(client)
    ingest(client, payload={"value": 1})

    r2 = amend(client, "e1", "a2", 1, occurred_at="2026-09-20T10:02:00Z",
               payload={"value": 2})
    assert r2.status_code == 201
    r3 = amend(client, "e1", "a3", 2, occurred_at="2026-09-20T10:03:00Z",
               payload={"value": 3})
    assert r3.status_code == 201
    assert r3.json()["revision"] == 3

    # Stale branch: basing on 1 after revision 3 landed must fail.
    stale = amend(client, "e1", "late", 1,
                  occurred_at="2026-09-20T10:04:00Z", payload={"value": 9})
    assert stale.status_code == 409
    assert stale.json()["detail"]["reason"] == "base_revision_stale"
    assert stale.json()["detail"]["current_revision"] == 3
    # The stale amendment persisted nothing: id absent, chain unchanged.
    conn = sqlite3.connect(client.app.state.db_path)
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM event_amendments WHERE amendment_id='late'"
        ).fetchone()[0] == 0
    finally:
        conn.close()

    chain = client.get("/events/e1/amendments").json()
    assert [r["revision"] for r in [chain["original"], *chain["history"]]] == [
        1,
        2,
        3,
    ]
    assert [r["amendment_id"] for r in chain["history"]] == ["a2", "a3"]
    assert chain["current"]["payload"] == {"value": 3}


def test_404_for_unknown_event(client):
    assert client.get("/events/missing/amendments").status_code == 404
    resp = amend(client, "missing", "a", 1, payload={"value": 1})
    assert resp.status_code == 404


# -------------------------------------------------------- identity/immutable


def test_event_id_path_body_mismatch_is_422(client):
    register_schema(client)
    ingest(client, payload={"value": 1})
    resp = client.post(
        "/events/e1/amendments",
        json={
            "device_id": "dev-1",
            "event_id": "other",
            "occurred_at": "2026-09-20T10:05:00Z",
            "amendment_id": "a",
            "base_revision": 1,
            "event_type": "telemetry",
            "schema_version": "1.0",
            "payload": {"value": 2},
        },
    )
    assert resp.status_code == 422


def test_device_id_is_immutable(client):
    register_schema(client)
    ingest(client, payload={"value": 1})
    resp = amend(client, "e1", "a", 1, device_id="dev-OTHER",
                 payload={"value": 2})
    assert resp.status_code == 409
    assert resp.json()["detail"]["reason"] == "device_id_immutable"
    # Nothing changed.
    chain = client.get("/events/e1/amendments").json()
    assert chain["current_revision"] == 1


def test_payload_validated_against_schema(client):
    register_schema(client)
    ingest(client, payload={"value": 1})
    resp = amend(client, "e1", "bad", 1, payload={"value": -5})
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["message"] == "payload does not satisfy the registered schema"
    assert any(e["path"] == "/value" for e in detail["errors"])
    # Nothing stored.
    assert client.get("/events/e1/amendments").json()["current_revision"] == 1


def test_unknown_contract_rejected(client):
    register_schema(client)
    ingest(client, payload={"value": 1})
    raw = client.post(
        "/events/e1/amendments",
        json={
            "device_id": "dev-1",
            "event_id": "e1",
            "occurred_at": "2026-09-20T10:05:00Z",
            "amendment_id": "bad",
            "base_revision": 1,
            "event_type": "nope",
            "schema_version": "2.0",
            "payload": {},
        },
    )
    assert raw.status_code == 422
    assert "not registered" in raw.json()["detail"]["errors"][0]["error"]
    # Nothing stored.
    assert client.get("/events/e1/amendments").json()["current_revision"] == 1


def test_incomplete_contract_triplet_rejected(client):
    register_schema(client)
    ingest(client, payload={"value": 1})
    raw = client.post(
        "/events/e1/amendments",
        json={
            "device_id": "dev-1",
            "event_id": "e1",
            "occurred_at": "2026-09-20T10:05:00Z",
            "amendment_id": "bad",
            "base_revision": 1,
            "event_type": "telemetry",
            # schema_version/payload omitted: incomplete triplet
        },
    )
    assert raw.status_code == 422
    assert raw.json()["detail"]["message"] == (
        "event_type, schema_version and payload must be provided together"
    )


def test_payload_less_event_can_become_payload_event_and_back(client):
    ingest(client, payload=None)
    # Amend a payload-less event into a contract payload event.
    register_schema(client)
    resp = amend(client, "e1", "enrich", 1, payload={"value": 7})
    assert resp.status_code == 201, resp.text
    assert resp.json()["event"]["event_type"] == "telemetry"
    # ...and back to a payload-less body.
    resp2 = amend(
        client,
        "e1",
        "strip",
        2,
        occurred_at="2026-09-20T10:09:00Z",
        payload=None,
    )
    assert resp2.status_code == 201, resp2.text
    assert resp2.json()["event"]["event_type"] is None
    chain = client.get("/events/e1/amendments").json()
    assert chain["original"]["event_type"] is None
    assert chain["current"]["event_type"] is None


# ------------------------------------------------------------- idempotency


def test_same_amendment_id_same_content_replays_200(client):
    register_schema(client)
    ingest(client, payload={"value": 1})
    kwargs = dict(
        occurred_at="2026-09-20T10:05:00Z", payload={"value": 2, "unit": "c"}
    )
    first = amend(client, "e1", "fix", 1, **kwargs)
    assert first.status_code == 201
    replay = amend(client, "e1", "fix", 1, **kwargs)
    assert replay.status_code == 200
    assert replay.json()["replay"] is True
    assert replay.json()["revision"] == 2
    # Key ordering in payload must not matter.
    raw = client.post(
        "/events/e1/amendments",
        json={
            "device_id": "dev-1",
            "event_id": "e1",
            "occurred_at": "2026-09-20T10:05:00Z",
            "amendment_id": "fix",
            "base_revision": 1,
            "event_type": "telemetry",
            "schema_version": "1.0",
            "payload": {"unit": "c", "value": 2},
        },
    )
    assert raw.status_code == 200
    # Only one amendment row exists.
    chain = client.get("/events/e1/amendments").json()
    assert len(chain["history"]) == 1


def test_same_amendment_id_different_content_is_409(client):
    register_schema(client)
    ingest(client, payload={"value": 1})
    assert amend(client, "e1", "fix", 1,
                 occurred_at="2026-09-20T10:05:00Z",
                 payload={"value": 2}).status_code == 201
    conflict = amend(client, "e1", "fix", 1,
                     occurred_at="2026-09-20T10:06:00Z",
                     payload={"value": 3})
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["reason"] == "amendment_id_content_conflict"
    # The original amendment is untouched.
    chain = client.get("/events/e1/amendments").json()
    assert chain["current_revision"] == 2
    assert chain["current"]["payload"] == {"value": 2}


# ------------------------------------------------------------- concurrency


def test_concurrent_branches_only_one_succeeds(tmp_path):
    """Two BEGIN IMMEDIATE amendments on base 1: exactly one commits.

    Both threads open their write transactions, read revision 1 as the
    current revision and attempt to insert revision 2. BEGIN IMMEDIATE
    serializes them on SQLite's writer lock; the loser observes the
    winner's committed revision and raises the stale-base conflict with no
    rows of its own persisted.
    """
    db_path = str(tmp_path / "conc.db")
    db_module.init_db(db_path)

    def resolver(conn, event_type, version):
        return None  # payload-less event

    conn0 = db_module.connect(db_path)
    conn0.execute(
        "INSERT INTO events (event_id, device_id, occurred_at) VALUES (?, ?, ?)",
        ("e1", "dev-1", "2026-09-20T10:00:00+00:00"),
    )
    conn0.commit()
    conn0.close()

    barrier = threading.Barrier(2)
    results: list[object] = []
    errors: list[Exception] = []

    def worker(amendment_id, new_minute):
        conn = db_module.connect(db_path)
        try:
            barrier.wait()
            conn.execute("BEGIN IMMEDIATE")
            try:
                result = apply_amendment(
                    conn,
                    resolver,
                    event_id="e1",
                    amendment_id=amendment_id,
                    base_revision=1,
                    device_id="dev-1",
                    occurred_at=datetime(
                        2026, 9, 20, 10, new_minute, tzinfo=timezone.utc
                    ),
                    event_type=None,
                    schema_version=None,
                    payload=None,
                    payload_provided=False,
                )
                conn.commit()
                results.append((amendment_id, result.history.current_revision))
            except Exception as exc:  # noqa: BLE001 - asserted below
                conn.rollback()
                errors.append(exc)
        finally:
            conn.close()

    t1 = threading.Thread(target=worker, args=("branch-a", 5))
    t2 = threading.Thread(target=worker, args=("branch-b", 6))
    t1.start()
    t2.start()
    t1.join(timeout=30)
    t2.join(timeout=30)

    # Exactly one branch commits revision 2; which one wins depends on
    # writer-lock scheduling.
    assert len(results) == 1, results
    winner, revision = results[0]
    assert winner in ("branch-a", "branch-b")
    assert revision == 2
    assert len(errors) == 1
    from app.amendments import AmendmentConflict

    assert isinstance(errors[0], AmendmentConflict)
    assert errors[0].detail["reason"] == "base_revision_stale"

    conn = db_module.connect(db_path)
    try:
        revisions = [
            row[0]
            for row in conn.execute(
                "SELECT revision FROM event_amendments ORDER BY revision"
            )
        ]
        assert revisions == [1, 2]
        assert conn.execute(
            "SELECT COUNT(*) FROM event_amendments WHERE revision >= 2"
        ).fetchone()[0] == 1
    finally:
        conn.close()


def test_http_concurrent_amendments_one_wins_and_loser_persists_nothing(tmp_path):
    """End-to-end: two POSTs racing on base_revision 1, exactly one wins."""
    db_path = str(tmp_path / "http.db")
    app = create_app(database_path=db_path, gap_seconds=300)

    def payload(aid, minute):
        return {
            "device_id": "dev-1",
            "event_id": "e1",
            "occurred_at": f"2026-09-20T10:0{minute}:00Z",
            "amendment_id": aid,
            "base_revision": 1,
        }

    with TestClient(app) as client:
        ingest(client, payload=None)
        statuses: list[int] = []
        lock = threading.Lock()
        barrier = threading.Barrier(2)

        def post(aid, minute):
            barrier.wait()
            resp = client.post("/events/e1/amendments", json=payload(aid, minute))
            with lock:
                statuses.append((aid, resp.status_code))

        t1 = threading.Thread(target=post, args=("http-a", 5))
        t2 = threading.Thread(target=post, args=("http-b", 6))
        t1.start()
        t2.start()
        t1.join(30)
        t2.join(30)

        assert sorted(code for _, code in statuses) == [201, 409], statuses
        loser = [aid for aid, code in statuses if code == 409][0]
        chain = client.get("/events/e1/amendments").json()
        assert chain["current_revision"] == 2
        assert len(chain["history"]) == 1
        # The loser's amendment id was never stored.
        assert chain["history"][0]["amendment_id"] != loser
        # Re-basing on revision 2 lets the loser succeed afterwards.
        follow = client.post(
            "/events/e1/amendments", json={**payload(loser, 7), "base_revision": 2}
        )
        assert follow.status_code == 201, follow.text
        assert follow.json()["revision"] == 3


# -------------------------------------------------- receipts/derived results


def test_amendment_adds_no_receipt_leaf_and_keeps_original_proof(client):
    register_schema(client)
    ingest(client, payload={"value": 1})

    proof_before = client.get("/receipts/proof/e1").json()
    leaf_before = proof_before["leaf"]
    checkpoint = client.post("/receipts/checkpoints").json()
    tree_size = checkpoint["tree_size"]

    amended = amend(client, "e1", "fix", 1,
                    occurred_at="2026-09-20T11:00:00Z",
                    payload={"value": 99})
    assert amended.status_code == 201

    proof_after = client.get("/receipts/proof/e1").json()
    # The leaf still binds the ORIGINAL ingestion record.
    assert proof_after["leaf"] == leaf_before
    assert proof_after["leaf"]["payload"] == {"value": 1}
    assert client.post("/receipts/checkpoints").json()["tree_size"] == tree_size
    assert proof_after["root_hash"] == proof_before["root_hash"]


def test_existing_state_transition_and_alarms_keep_original(client):
    """Amendments neither project transitions nor recompute alarms."""
    # Enable a machine whose single transition fires at ingestion time.
    machine = {
        "version": 1,
        "initial_state": "off",
        "states": ["off", "on"],
        "transitions": [
            {
                "from_state": "off",
                "event_type": "telemetry",
                "to_state": "on",
                "conditions": [],
            }
        ],
    }
    assert client.put(
        "/devices/dev-1/state-machine/versions/1", json=machine
    ).status_code in (200, 201)
    assert client.post(
        "/devices/dev-1/state-machine/enable", json={"version": 1}
    ).status_code in (200, 201)

    register_schema(client)
    ingested = ingest(client, payload={"value": 1})
    assert ingested.json()["state_transition"]["to_state"] == "on"

    # Amend the event to a different time/payload that would sort
    # elsewhere; the log and projection must stay exactly as ingested.
    resp = amend(client, "e1", "fix", 1,
                 occurred_at="2030-01-01T00:00:00Z",
                 payload={"value": 123})
    assert resp.status_code == 201

    log = client.get("/devices/dev-1/state-machine/transitions").json()
    assert len(log["transitions"]) == 1
    assert log["transitions"][0]["event_id"] == "e1"
    assert log["transitions"][0]["to_state"] == "on"
    assert log["transitions"][0]["occurred_at"].startswith("2026-09-20T10:00:00")

    status = client.get("/devices/dev-1/state-machine").json()
    assert status["last_event_seq"] == 1


def test_sessions_and_search_see_the_amended_body(client):
    register_schema(client)
    ingest(client, "e1", occurred_at="2026-09-20T10:00:00Z",
           payload={"value": 1})

    assert amend(
        client, "e1", "fix", 1,
        occurred_at="2026-09-20T12:00:00Z",
        payload={"value": 77},
    ).status_code == 201

    sessions = client.get("/devices/dev-1/sessions").json()
    member = sessions[0]["members"][0]
    assert member["occurred_at"].startswith("2026-09-20T12:00:00")
    assert member["payload"] == {"value": 77}

    search = client.post(
        "/events/search",
        json={
            "conditions": [
                {"pointer": "/value", "operator": "eq", "value": 77}
            ]
        },
    ).json()
    assert [e["event_id"] for e in search["events"]] == ["e1"]
    old = client.post(
        "/events/search",
        json={
            "conditions": [
                {"pointer": "/value", "operator": "eq", "value": 1}
            ]
        },
    ).json()
    assert old["events"] == []


# ------------------------------------------------------- retention cleanup


def test_chain_remains_readable_after_cleanup_deletes_live_body(client):
    register_schema(client)
    ingest(client, occurred_at="2026-09-20T10:00:00Z", payload={"value": 1})
    assert amend(
        client, "e1", "fix", 1,
        occurred_at="2026-09-20T10:05:00Z",
        payload={"value": 2},
    ).status_code == 201

    plan = client.post(
        "/retention/cleanup-plans",
        json={
            "device_id": "dev-1",
            "before": "2030-01-01T00:00:00Z",
            "protected_windows": [],
        },
    ).json()
    assert plan["count"] == 1
    confirm = client.post(
        f"/retention/cleanup-plans/{plan['plan_id']}/confirm"
    )
    assert confirm.status_code == 200, confirm.text

    # The live body is gone; the frozen chain still answers.
    chain = client.get("/events/e1/amendments")
    assert chain.status_code == 200
    data = chain.json()
    assert data["original"]["payload"] == {"value": 1}
    assert data["current"]["payload"] == {"value": 2}
    assert [r["revision"] for r in [data["original"], *data["history"]]] == [
        1,
        2,
    ]

    # A new amendment on a cleaned-up event cannot apply (no live body).
    resp = amend(
        client, "e1", "fix2", 2,
        occurred_at="2026-09-20T10:06:00Z",
        payload={"value": 3},
    )
    assert resp.status_code == 404

    # The idempotent replay of the stored amendment still works.
    replay = amend(
        client, "e1", "fix", 1,
        occurred_at="2026-09-20T10:05:00Z",
        payload={"value": 2},
    )
    assert replay.status_code == 200
    assert replay.json()["replay"] is True


def test_unamended_event_disappears_after_cleanup(client):
    ingest(client, payload=None)
    plan = client.post(
        "/retention/cleanup-plans",
        json={
            "device_id": "dev-1",
            "before": "2030-01-01T00:00:00Z",
            "protected_windows": [],
        },
    ).json()
    assert client.post(
        f"/retention/cleanup-plans/{plan['plan_id']}/confirm"
    ).status_code == 200
    assert client.get("/events/e1/amendments").status_code == 404


# --------------------------------------------------------- legacy databases


def test_legacy_database_without_amendments_table_works(tmp_path):
    """A database created before this module gains the chain at startup."""
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE events (
            seq INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL UNIQUE,
            device_id TEXT NOT NULL,
            occurred_at TEXT NOT NULL,
            event_type TEXT,
            schema_version TEXT,
            payload TEXT
        );
        """
    )
    conn.execute(
        "INSERT INTO events (event_id, device_id, occurred_at) "
        "VALUES ('old', 'dev-1', '2026-01-01T00:00:00+00:00')"
    )
    conn.commit()
    conn.close()

    app = create_app(database_path=str(db_path), gap_seconds=300)
    with TestClient(app) as client:
        got = client.get("/events/old/amendments")
        assert got.status_code == 200
        assert got.json()["current_revision"] == 1

        resp = client.post(
            "/events/old/amendments",
            json={
                "device_id": "dev-1",
                "event_id": "old",
                "occurred_at": "2026-01-02T00:00:00Z",
                "amendment_id": "first-fix",
                "base_revision": 1,
            },
        )
        assert resp.status_code == 201, resp.text
        assert resp.json()["revision"] == 2
        # The frozen revision 1 preserves the legacy body verbatim.
        chain = client.get("/events/old/amendments").json()
        assert chain["original"]["occurred_at"].startswith("2026-01-01T00:00:00")
        assert chain["original"]["device_id"] == "dev-1"

    # Restart: tables/rows survive and the migration stays idempotent.
    app2 = create_app(database_path=str(db_path), gap_seconds=300)
    with TestClient(app2) as client:
        chain = client.get("/events/old/amendments").json()
        assert chain["current_revision"] == 2
        assert len(chain["history"]) == 1


def test_amendment_keeps_canonical_payload_and_original_seq(tmp_path):
    """Amendment payload is stored canonically; events.seq never changes."""
    db_path = str(tmp_path / "x.db")
    db_module.init_db(db_path)
    conn = db_module.connect(db_path)
    conn.execute(
        "INSERT INTO events (event_id, device_id, occurred_at, event_type, "
        "schema_version, payload) VALUES (?, ?, ?, ?, ?, ?)",
        (
            "e1",
            "dev-1",
            "2026-09-20T10:00:00+00:00",
            "t",
            "1",
            canonical_json({"b": 1, "a": 2}),
        ),
    )
    conn.commit()
    assert conn.execute("SELECT seq FROM events WHERE event_id='e1'").fetchone()[
        0
    ] == 1
    conn.close()

    app = create_app(database_path=db_path, gap_seconds=300)
    with TestClient(app) as client:
        # Contract 't/1' accepts arbitrary objects.
        client.put(
            "/schemas/t/1",
            json={"schema": {"type": "object"}},
        )
        resp = client.post(
            "/events/e1/amendments",
            json={
                "device_id": "dev-1",
                "event_id": "e1",
                "occurred_at": "2026-09-20T10:01:00Z",
                "amendment_id": "fix",
                "base_revision": 1,
                "event_type": "t",
                "schema_version": "1",
                "payload": {"z": [1, 2], "a": {"y": 2, "x": 1}},
            },
        )
        assert resp.status_code == 201, resp.text

    conn = sqlite3.connect(db_path)
    try:
        # Frozen original verbatim; new revision canonicalized.
        rows = {
            row[0]: row[1]
            for row in conn.execute(
                "SELECT revision, payload FROM event_amendments ORDER BY revision"
            )
        }
        assert rows[1] == canonical_json({"b": 1, "a": 2})
        assert rows[2] == canonical_json({"z": [1, 2], "a": {"y": 2, "x": 1}})
        # Live row follows the new body; seq is untouched.
        live = conn.execute(
            "SELECT seq, payload FROM events WHERE event_id='e1'"
        ).fetchone()
        assert live[0] == 1
        assert live[1] == rows[2]
    finally:
        conn.close()
