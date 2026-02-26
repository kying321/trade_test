from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import unittest

from lie_engine.orchestration.events import (
    append_event_envelope,
    build_event_envelope,
    build_traceparent,
    trace_id_from_traceparent,
)


class EventEnvelopeTests(unittest.TestCase):
    def test_build_event_envelope_has_required_fields(self) -> None:
        env = build_event_envelope(
            source="unit.test",
            event_type="unit.event",
            payload={"k": 1, "v": "x"},
            as_of=date(2026, 2, 21),
            trace_id="trace_fixed",
            parent_event_id="parent_1",
        )
        row = env.to_dict()
        self.assertEqual(row["trace_id"], "trace_fixed")
        self.assertEqual(row["parent_event_id"], "parent_1")
        self.assertEqual(row["source"], "unit.test")
        self.assertEqual(row["event_type"], "unit.event")
        self.assertTrue(bool(row["event_id"]))
        self.assertTrue(bool(row["payload_hash"]))
        self.assertTrue(re.match(r"^[0-9a-f]{2}-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}$", str(row["traceparent"])))
        trace_hex = str(trace_id_from_traceparent(row["traceparent"]))
        self.assertTrue(bool(re.match(r"^[0-9a-f]{32}$", trace_hex)))
        expected_hex = hashlib.sha256(str(row["trace_id"]).encode("utf-8")).hexdigest()[:32]
        self.assertEqual(trace_hex, expected_hex)

    def test_append_event_envelope_writes_ndjson(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        env = build_event_envelope(
            source="unit.writer",
            event_type="write.once",
            payload={"a": 1},
            as_of="2026-02-21",
            trace_id="trace_write",
        )
        out_path = append_event_envelope(output_dir=td, envelope=env, payload={"a": 1})
        self.assertTrue(out_path.exists())
        lines = [x for x in out_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        self.assertEqual(len(lines), 1)
        doc = json.loads(lines[0])
        self.assertEqual(str((doc.get("envelope", {}) or {}).get("trace_id", "")), "trace_write")
        self.assertEqual(int((doc.get("payload", {}) or {}).get("a", 0)), 1)

    def test_build_event_envelope_uses_incoming_traceparent_when_valid(self) -> None:
        tp = build_traceparent(trace_id="a" * 32, span_id="b" * 16, sampled=True)
        env = build_event_envelope(
            source="unit.tp",
            event_type="tp.event",
            payload={},
            traceparent=tp,
        )
        row = env.to_dict()
        self.assertEqual(str(row["traceparent"]), tp)
        self.assertEqual(str(row["trace_id"]), "a" * 32)


if __name__ == "__main__":
    unittest.main()
