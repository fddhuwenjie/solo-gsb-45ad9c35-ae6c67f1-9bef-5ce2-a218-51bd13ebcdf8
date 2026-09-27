"""Tests for the immutable event amendment chain."""

import threading

import pytest
from fastapi.testclient import TestClient

from app.main import create_app

GAP = 300


@pytest.fixture()
def client(tmp_path):
    app = create_app(database_path=str(tmp_path / "test.db"), gap_seconds=GAP)
    with TestClient(app) as c:
        yield c


def put_schema(client, event_type="telemetry", version="1.0", schema=None):
    if schema is None:
        schema = {
            "type": "object",
            "required": ["v"],
            "properties": {"v": {"type": "number"}},
        }
    resp = client.put(f"/schemas/{event_type}/{version}", json={"schema": schema})
    assert resp.status_code in (200, 201), resp.text


def ingest(
    client,
    event_id="e1",
    device_id="dev-1",
    occurred_at="2026-09-20T10:00:00Z",
    *,
    payload=None,
):
    body = {"device_id": device_id, "event_id": event_id, "occurred_at": occurred_at}
    if payload is not None:
        body.update(
            event_type="telemetry", schema_version="1.0", payload=payload
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
    occurred_at="2026-09-20T10:00:05Z",
    payload=Ellipsis,
):
    body = {"amendment_id": amendment_id, "base_revision": base_revision}
    body["occurred_at"] = occurred_at
    if payload is not Ellipsis:
        if payload is None:
            body.update(event_type=None, schema_version=None)
        else:
            body.update(
                event_type="telemetry", schema_version="1.0", payload=payload
            )
    return client.post(f"/events/{event_id}/amendments", json=body)


# ------------------------------------------------------------------- reading


def test_get_chain_before_first_amendment_is_single_revision(client):
    put_schema(client)
    ingest(client, payload={"v": 1})
    resp = client.get("/events/e1/amendments")
    assert resp.status_code == 200
    body = resp.json()
    assert body["event_id"] == "e1"
    assert body["device_id"] == "dev-1"
    assert body["current_revision"] == 1
    assert body["frozen"] is False
    assert body["frozen_at"] is None
    assert len(body["history"]) == 1
    original = body["original"]
    assert original["revision"] == 1
    assert original["amendment_id"] is None
    assert original["base_revision"] is None
    assert original["frozen_at"] is None
    assert original["occurred_at"].startswith("2026-09-20T10:00:00")
    assert original["payload"] == {"v": 1}
    assert body["current"] == original


def test_get_chain_unknown_event_404(client):
    assert client.get("/events/ghost/amendments").status_code == 404


# ------------------------------------------------------------------ writing


def test_first_amendment_freezes_original_and_advances_revision(client):
    put_schema(client)
    ingest(client, payload={"v": 1})

    resp = amend(client, "e1", "a1", 1, payload={"v": 2})
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["replay"] is False
    assert body["revision"] == 2
    assert body["current_revision"] == 2
    assert body["frozen"] is True
    assert body["frozen_at"] is not None

    revisions = body["history"]
    assert [r["revision"] for r in revisions] == [1, 2]
    original, replacement = revisions
    # The frozen original is untouched.
    assert original["amendment_id"] is None
    assert original["base_revision"] is None
    assert original["occurred_at"].startswith("2026-09-20T10:00:00")
    assert original["payload"] == {"v": 1}
    assert original["frozen_at"] == body["frozen_at"]
    # The replacement body.
    assert replacement["amendment_id"] == "a1"
    assert replacement["base_revision"] == 1
    assert replacement["occurred_at"].startswith("2026-09-20T10:00:05")
    assert replacement["payload"] == {"v": 2}

    assert body["current"] == replacement
    assert body["original"] == original


def test_amendment_chain_advances_by_one(client):
    put_schema(client)
    ingest(client)
    assert amend(client, "e1", "a1", 1, payload={"v": 1}).status_code == 201
    assert amend(client, "e1", "a2", 2, payload={"v": 2}).status_code == 201
    resp = amend(client, "e1", "a3", 3, occurred_at="2026-09-20T10:09:00Z")
    assert resp.status_code == 201
    body = resp.json()
    assert body["revision"] == 4
    assert [r["revision"] for r in body["history"]] == [1, 2, 3, 4]
    assert body["current"]["occurred_at"].startswith("2026-09-20T10:09:00")


def test_device_id_and_event_id_cannot_change(client):
    """The identity columns stay bound to the original ingestion."""
    put_schema(client)
    ingest(client, payload={"v": 1})
    amend(client, "e1", "a1", 1, payload={"v": 2})
    body = client.get("/events/e1/amendments").json()
    assert all(r["device_id"] == "dev-1" for r in body["history"])
    assert all(r["event_id"] == "e1" for r in body["history"])


def test_amendment_allows_payload_to_payloadless_and_back(client):
    put_schema(client)
    ingest(client, payload={"v": 1})

    resp = amend(client, "e1", "a1", 1, payload=None)
    assert resp.status_code == 201, resp.text
    current = resp.json()["current"]
    assert current["event_type"] is None
    assert current["schema_version"] is None
    assert current["payload"] is None

    resp = amend(
        client,
        "e1",
        "a2",
        2,
        occurred_at="2026-09-20T10:00:06Z",
        payload={"v": 3},
    )
    assert resp.status_code == 201, resp.text
    current = resp.json()["current"]
    assert current["event_type"] == "telemetry"
    assert current["payload"] == {"v": 3}


def test_timezone_offsets_normalize_to_same_revision_body(client):
    put_schema(client)
    ingest(client, occurred_at="2026-09-20T12:00:00+02:00", payload={"v": 1})
    # Same instant expressed in UTC must store the same occurred_at.
    resp = amend(
        client,
        "e1",
        "a1",
        1,
        occurred_at="2026-09-20T10:00:30Z",
        payload={"v": 2},
    )
    assert resp.status_code == 201
    assert resp.json()["current"]["occurred_at"].startswith("2026-09-20T10:00:30")


# ------------------------------------------------------------- idempotency


def test_same_amendment_id_and_content_is_idempotent_replay(client):
    put_schema(client)
    ingest(client, payload={"v": 1})
    first = amend(client, "e1", "a1", 1, payload={"v": 2})
    assert first.status_code == 201

    second = amend(client, "e1", "a1", 1, payload={"v": 2})
    assert second.status_code == 200
    body = second.json()
    assert body["replay"] is True
    assert body["revision"] == 2
    assert body["current_revision"] == 2
    # The full response matches the first response apart from replay/code.
    assert body["history"] == first.json()["history"]

    # Only one revision row exists.
    body = client.get("/events/e1/amendments").json()
    assert [r["revision"] for r in body["history"]] == [1, 2]


def test_same_amendment_id_with_changed_content_returns_409(client):
    put_schema(client)
    ingest(client, payload={"v": 1})
    assert amend(client, "e1", "a1", 1, payload={"v": 2}).status_code == 201

    resp = amend(client, "e1", "a1", 1, payload={"v": 99})
    assert resp.status_code == 409
    assert resp.json()["detail"]["reason"] == "amendment_id_conflict"

    # Changed base under the same id is also a rewrite -> 409.
    resp = amend(client, "e1", "a1", 2, payload={"v": 2})
    assert resp.status_code == 409
    assert resp.json()["detail"]["reason"] == "amendment_id_conflict"

    # The stored chain is untouched.
    body = client.get("/events/e1/amendments").json()
    assert [r["revision"] for r in body["history"]] == [1, 2]
    assert body["current"]["payload"] == {"v": 2}


def test_replay_of_old_amendment_after_chain_advanced_still_200(client):
    put_schema(client)
    ingest(client)
    assert amend(client, "e1", "a1", 1, payload={"v": 1}).status_code == 201
    assert amend(client, "e1", "a2", 2, payload={"v": 2}).status_code == 201
    # Replaying a1's exact request returns the stored revision 2 even
    # though the chain head has moved.
    resp = amend(client, "e1", "a1", 1, payload={"v": 1})
    assert resp.status_code == 200
    assert resp.json()["replay"] is True
    assert resp.json()["revision"] == 2
    assert resp.json()["current_revision"] == 3


# --------------------------------------------------------- optimistic lock


def test_stale_base_revision_returns_409_and_stores_nothing(client):
    put_schema(client)
    ingest(client, payload={"v": 1})
    assert amend(client, "e1", "a1", 1, payload={"v": 2}).status_code == 201

    resp = amend(client, "e1", "a2", 1, payload={"v": 3})
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["reason"] == "stale_base_revision"
    assert detail["base_revision"] == 1
    assert detail["current_revision"] == 2

    body = client.get("/events/e1/amendments").json()
    assert [r["revision"] for r in body["history"]] == [1, 2]
    assert body["current"]["payload"] == {"v": 2}


def test_concurrent_branches_from_same_base_only_one_wins(tmp_path):
    app = create_app(database_path=str(tmp_path / "race.db"), gap_seconds=GAP)
    with TestClient(app) as main_client:
        put_schema(main_client)
        ingest(main_client)

        statuses: list[int] = []
        barrier = threading.Barrier(8)
        lock = threading.Lock()

        def worker(i):
            with TestClient(app) as c:
                barrier.wait()
                resp = amend(
                    c,
                    "e1",
                    f"a{i}",
                    1,
                    occurred_at=f"2026-09-20T10:00:0{i}Z",
                )
                with lock:
                    statuses.append(resp.status_code)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert statuses.count(201) == 1
    assert statuses.count(409) == 7
    body = main_client.get("/events/e1/amendments").json()
    assert [r["revision"] for r in body["history"]] == [1, 2]


def test_base_revision_zero_rejected(client):
    ingest(client)
    resp = client.post(
        "/events/e1/amendments",
        json={
            "amendment_id": "a1",
            "base_revision": 0,
            "occurred_at": "2026-09-20T10:00:05Z",
        },
    )
    assert resp.status_code == 422


# ----------------------------------------------------------- contract checks


def test_amendment_payload_validated_against_schema(client):
    put_schema(client)
    ingest(client, payload={"v": 1})
    resp = amend(client, "e1", "a1", 1, payload={"v": "not-a-number"})
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["message"] == "payload does not satisfy the registered schema"
    assert any(err["path"] == "/v" for err in detail["errors"])

    # Nothing stored: still at revision 1, still unfrozen.
    body = client.get("/events/e1/amendments").json()
    assert body["current_revision"] == 1
    assert body["frozen"] is False


def test_amendment_unknown_contract_returns_422(client):
    ingest(client)
    resp = client.post(
        "/events/e1/amendments",
        json={
            "amendment_id": "a1",
            "base_revision": 1,
            "occurred_at": "2026-09-20T10:00:05Z",
            "event_type": "telemetry",
            "schema_version": "9.9",
            "payload": {"v": 1},
        },
    )
    assert resp.status_code == 422
    assert client.get("/events/e1/amendments").json()["current_revision"] == 1


def test_amendment_requires_contract_triplet(client):
    ingest(client)
    resp = client.post(
        "/events/e1/amendments",
        json={
            "amendment_id": "a1",
            "base_revision": 1,
            "occurred_at": "2026-09-20T10:00:05Z",
            "event_type": "telemetry",
            "schema_version": "1.0",
        },
    )
    assert resp.status_code == 422
    paths = {err["path"] for err in resp.json()["detail"]["errors"]}
    assert paths == {"/payload"}


# --------------------------------------------------------------- not found


def test_amend_never_ingested_event_returns_404(client):
    resp = client.post(
        "/events/ghost/amendments",
        json={
            "amendment_id": "a1",
            "base_revision": 1,
            "occurred_at": "2026-09-20T10:00:00Z",
        },
    )
    assert resp.status_code == 404


# ---------------------------------------------- derived results stay original


def test_sessions_state_machine_alerts_and_receipt_stay_original(client):
    """Amendments never re-run ingestion-side derivations."""
    put_schema(client)
    # Two events far apart -> two sessions; amending e1's timestamp to
    # bridge them must NOT merge the sessions.
    ingest(client, event_id="e1", occurred_at="2026-09-20T10:00:00Z")
    ingest(client, event_id="e2", occurred_at="2026-09-20T10:08:00Z")
    assert len(client.get("/devices/dev-1/sessions").json()) == 2

    resp = amend(
        client,
        "e1",
        "a1",
        1,
        occurred_at="2026-09-20T10:07:30Z",
        payload=None,
    )
    assert resp.status_code == 201

    sessions = client.get("/devices/dev-1/sessions").json()
    assert len(sessions) == 2
    member = sessions[0]["members"][0]
    # Live session view keeps the original ingestion body.
    assert member["occurred_at"].startswith("2026-09-20T10:00:00")
    assert member["event_type"] is None

    # Search sees the original body too.
    found = client.post(
        "/events/search",
        json={"conditions": [{"pointer": "/v", "operator": "exists", "value": True}]},
    ).json()["events"]
    assert found == []

    # The Merkle receipt leaf still commits to the original ingestion.
    leaf = client.get("/receipts/proof/e1").json()["leaf"]
    assert leaf["occurred_at"].startswith("2026-09-20T10:00:00")
    assert leaf["payload"] is None


def test_post_events_replay_of_amended_event_keeps_original_semantics(client):
    put_schema(client)
    ingest(client, payload={"v": 1})
    assert amend(client, "e1", "a1", 1, payload={"v": 2}).status_code == 201

    # Re-posting the ORIGINAL content stays an idempotent replay.
    resp = ingest(client, payload={"v": 1})
    assert resp.status_code == 200
    assert resp.json()["replay"] is True

    # Posting the amended body as a "new" content still conflicts: the
    # live event row is the original.
    resp = client.post(
        "/events",
        json={
            "device_id": "dev-1",
            "event_id": "e1",
            "occurred_at": "2026-09-20T10:00:05Z",
            "event_type": "telemetry",
            "schema_version": "1.0",
            "payload": {"v": 2},
        },
    )
    assert resp.status_code == 409


# ------------------------------------------------------ retention interaction


def _cleanup(client, device_id="dev-1", before="2027-01-01T00:00:00Z"):
    plan = client.post(
        "/retention/cleanup-plans",
        json={"device_id": device_id, "before": before, "protected_windows": []},
    ).json()
    resp = client.post(f"/retention/cleanup-plans/{plan['plan_id']}/confirm")
    assert resp.status_code == 200, resp.text


def test_frozen_chain_remains_readable_after_cleanup(client):
    put_schema(client)
    ingest(client, payload={"v": 1})
    assert amend(client, "e1", "a1", 1, payload={"v": 2}).status_code == 201
    assert amend(client, "e1", "a2", 2, payload={"v": 3}).status_code == 201

    _cleanup(client)

    resp = client.get("/events/e1/amendments")
    assert resp.status_code == 200
    body = resp.json()
    assert body["frozen"] is True
    assert [r["revision"] for r in body["history"]] == [1, 2, 3]
    assert body["original"]["payload"] == {"v": 1}
    assert body["current"]["payload"] == {"v": 3}

    # The Merkle proof survives on the retained leaf as well.
    assert client.get("/receipts/proof/e1").status_code == 200


def test_amendments_can_continue_after_cleanup(client):
    put_schema(client)
    ingest(client)
    assert amend(client, "e1", "a1", 1, occurred_at="2026-09-20T10:01:00Z").status_code == 201
    _cleanup(client)

    resp = amend(
        client,
        "e1",
        "a2",
        2,
        occurred_at="2026-09-20T10:02:00Z",
        payload=None,
    )
    assert resp.status_code == 201
    assert resp.json()["revision"] == 3
    body = client.get("/events/e1/amendments").json()
    assert [r["revision"] for r in body["history"]] == [1, 2, 3]


def test_never_ingested_event_stays_404_after_cleanup_window(client):
    # An id that exists nowhere is still 404.
    assert client.get("/events/ghost/amendments").status_code == 404


# ------------------------------------------------------------ migration


def test_legacy_database_gains_amendments_table(tmp_path):
    import sqlite3

    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE events (
            seq INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL UNIQUE,
            device_id TEXT NOT NULL,
            occurred_at TEXT NOT NULL,
            event_type TEXT, schema_version TEXT, payload TEXT
        );
        """
    )
    conn.execute(
        "INSERT INTO events VALUES (1, 'e1', 'dev-1', "
        "'2026-09-20T10:00:00+00:00', NULL, NULL, NULL)"
    )
    conn.commit()
    conn.close()

    app = create_app(database_path=str(db_path), gap_seconds=GAP)
    with TestClient(app) as client:
        body = client.get("/events/e1/amendments").json()
        assert body["current_revision"] == 1
        assert body["frozen"] is False

        resp = amend(client, "e1", "a1", 1, occurred_at="2026-09-20T10:00:30Z")
        assert resp.status_code == 201

    # Restart: schema creation is idempotent and the chain survives.
    app2 = create_app(database_path=str(db_path), gap_seconds=GAP)
    with TestClient(app2) as client:
        body = client.get("/events/e1/amendments").json()
        assert [r["revision"] for r in body["history"]] == [1, 2]
        assert body["current"]["occurred_at"].startswith("2026-09-20T10:00:30")
