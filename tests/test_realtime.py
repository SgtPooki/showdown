"""Tests for Server-Sent Events (SSE) streaming and live match synchronization."""

import asyncio
import json
from unittest.mock import MagicMock, patch
import pytest
from starlette.testclient import TestClient

from showdown.models import Candidate, TaskType, Tournament
from showdown.realtime import TournamentEventHub, format_sse
from showdown.server import app
from showdown.storage import Storage


@pytest.fixture
def client():
    return TestClient(app)


def test_format_sse():
    """Verify SSE line protocol formatting."""
    msg = format_sse("test_event", {"status": "ok", "value": 42})
    assert msg.startswith("event: test_event\n")
    assert "data: " in msg
    assert '"status": "ok"' in msg
    assert msg.endswith("\n\n")


@pytest.mark.anyio
async def test_event_hub_pub_sub():
    """Verify TournamentEventHub subscription, broadcast, and unsubscription."""
    hub = TournamentEventHub()
    q1 = hub.subscribe("tourney_1")
    q2 = hub.subscribe("tourney_1")
    q_other = hub.subscribe("tourney_2")

    hub.publish("tourney_1", "match_recorded", {"match_id": "m1"})

    # Both subscribers of tourney_1 receive event
    msg1 = await q1.get()
    msg2 = await q2.get()
    assert "event: match_recorded" in msg1
    assert "match_id" in msg1
    assert "event: match_recorded" in msg2

    # tourney_2 subscriber should have no messages
    assert q_other.empty()

    # Unsubscribe
    hub.unsubscribe("tourney_1", q1)
    hub.publish("tourney_1", "match_recorded", {"match_id": "m2"})
    assert q1.empty()
    msg2_next = await q2.get()
    assert '"match_id": "m2"' in msg2_next


def test_api_tournament_events_stream(client: TestClient):
    """Verify GET /api/tournaments/{id}/events streams initial handshake."""
    res_create = client.post("/api/tournaments", json={
        "title": "SSE Stream Test",
        "task_type": "text",
        "candidates": [
            {"id": "c1", "content": "Option 1"},
            {"id": "c2", "content": "Option 2"},
        ],
    })
    t_id = res_create.json()["id"]

    with client.stream("GET", f"/api/tournaments/{t_id}/events?timeout=0.1") as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        for line in response.iter_lines():
            if line.startswith("data:"):
                data = json.loads(line.replace("data: ", ""))
                assert data["tournament_id"] == t_id
                assert data["status"] == "listening"
                break


def test_api_evolve_stream_endpoint(client: TestClient):
    """Verify POST /api/tournaments/{id}/evolve/stream yields complete SSE lifecycle."""
    res_create = client.post("/api/tournaments", json={
        "title": "SSE Evolve Test",
        "task_type": "text",
        "candidates": [
            {"id": "c1", "content": "Strong prompt baseline"},
        ],
    })
    t_id = res_create.json()["id"]

    mock_llm_json = json.dumps([
        {"label": "Streamed 1", "content": "Streamed Content 1", "differs_by": "Point A"},
        {"label": "Streamed 2", "content": "Streamed Content 2", "differs_by": "Point B"},
    ])

    events_received = []

    with patch("showdown.evolve.call_generation_backend", return_value=mock_llm_json):
        with client.stream(
            "POST",
            f"/api/tournaments/{t_id}/evolve/stream",
            json={
                "count": 2,
                "backend": "mock",
                "mode": "refine",
            },
        ) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]

            current_event = None
            for line in response.iter_lines():
                if line.startswith("event: "):
                    current_event = line.replace("event: ", "").strip()
                elif line.startswith("data: ") and current_event:
                    data = json.loads(line.replace("data: ", "").strip())
                    events_received.append((current_event, data))
                    current_event = None

    event_names = [e[0] for e in events_received]
    assert "synthesizing_preferences" in event_names
    assert "provider_dispatched" in event_names
    assert "candidate_ready" in event_names
    assert "complete" in event_names

    # Check that candidate_ready was received per item
    candidate_events = [e[1] for e in events_received if e[0] == "candidate_ready"]
    assert len(candidate_events) == 2
    assert candidate_events[0]["label"] == "Streamed 1"

    # Check complete payload
    complete_events = [e[1] for e in events_received if e[0] == "complete"]
    assert len(complete_events) == 1
    assert len(complete_events[0]["new_candidates"]) == 2


def test_api_evolve_stream_fanout_multi_agent(client: TestClient):
    """Verify evolution streaming fans out across multiple providers and streams candidates."""
    from showdown.providers import registry

    res_create = client.post("/api/tournaments", json={
        "title": "SSE Fanout Stream Test",
        "task_type": "text",
        "candidates": [
            {"id": "c1", "content": "Baseline Content"},
        ],
    })
    t_id = res_create.json()["id"]

    claude_json = json.dumps([
        {"label": "Claude Stream", "content": "From Claude", "differs_by": "Speed"},
    ])
    omp_json = json.dumps([
        {"label": "OMP Stream", "content": "From Homelab", "differs_by": "Local"},
    ])

    def mock_backend(prompt, backend):
        if backend == "claude":
            return claude_json
        return omp_json

    with patch.object(registry.get("claude"), "is_available", return_value=True):
        with patch.object(registry.get("omp"), "is_available", return_value=True):
            with patch("showdown.evolve.call_generation_backend", side_effect=mock_backend):
                with client.stream(
                    "POST",
                    f"/api/tournaments/{t_id}/evolve/stream",
                    json={
                        "count": 2,
                        "providers": ["claude", "omp"],
                        "mode": "refine",
                    },
                ) as response:
                    assert response.status_code == 200

                    events = []
                    curr_evt = None
                    for line in response.iter_lines():
                        if line.startswith("event: "):
                            curr_evt = line.replace("event: ", "").strip()
                        elif line.startswith("data: ") and curr_evt:
                            events.append((curr_evt, json.loads(line.replace("data: ", "").strip())))
                            curr_evt = None

                    event_types = [e[0] for e in events]
                    assert "provider_dispatched" in event_types
                    assert "candidate_ready" in event_types
                    assert "complete" in event_types

                    # Check providers in dispatched event
                    disp = next(e[1] for e in events if e[0] == "provider_dispatched")
                    assert "claude" in disp["providers"]
                    assert "omp" in disp["providers"]
                    assert disp["fan_out"] is True


def test_live_match_sync_broadcast(client: TestClient):
    """Verify live tournament events stream receives match_recorded and candidate_accepted events."""
    from showdown.realtime import event_hub

    res_create = client.post("/api/tournaments", json={
        "title": "Live Match Sync Test",
        "task_type": "text",
        "candidates": [
            {"id": "a", "content": "Option A"},
            {"id": "b", "content": "Option B"},
        ],
    })
    t_id = res_create.json()["id"]

    sub_queue = event_hub.subscribe(t_id)
    try:
        # Record a vote
        res_vote = client.post(f"/api/tournaments/{t_id}/vote", json={
            "id_a": "a",
            "id_b": "b",
            "winner": "a",
            "voter": "human_1",
        })
        assert res_vote.status_code == 200

        # Verify event was published
        assert not sub_queue.empty()
        msg = sub_queue.get_nowait()
        assert "event: match_recorded" in msg
        assert '"winner": "a"' in msg

        # Accept candidate
        res_accept = client.post(f"/api/tournaments/{t_id}/accept", json={
            "candidate_id": "a",
            "notes": "Winner!",
        })
        assert res_accept.status_code == 200
        assert not sub_queue.empty()
        msg_accept = sub_queue.get_nowait()
        assert "event: candidate_accepted" in msg_accept
        assert '"candidate_id": "a"' in msg_accept
    finally:
        event_hub.unsubscribe(t_id, sub_queue)
