from __future__ import annotations

import json
import subprocess
from pathlib import Path

SCRIPT_PATH = Path('/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/build_commodity_paper_execution_retro.py')


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding='utf-8')


def test_build_commodity_paper_execution_retro(tmp_path: Path) -> None:
    review_dir = tmp_path / 'review'
    _write_json(
        review_dir / '20260310T122754Z_commodity_paper_execution_review.json',
        {
            'status': 'ok',
            'route_status': 'paper-first',
            'ticket_book_status': 'paper-ready',
            'execution_preview_status': 'paper-execution-ready',
            'execution_artifact_status': 'paper-execution-artifact-ready',
            'execution_queue_status': 'paper-execution-queued',
            'execution_review_status': 'paper-execution-review-pending',
            'execution_mode': 'paper_only',
            'execution_batch': 'metals_all',
            'execution_symbols': ['XAUUSD', 'XAGUSD', 'COPPER'],
            'execution_ticket_ids': [
                'commodity-paper-ticket:metals_all:XAUUSD',
                'commodity-paper-ticket:metals_all:XAGUSD',
                'commodity-paper-ticket:metals_all:COPPER',
            ],
            'execution_regime_gate': 'paper_only',
            'execution_weight_hint_sum': 2.3,
            'execution_item_count': 3,
            'actionable_execution_item_count': 3,
            'queue_depth': 3,
            'actionable_queue_depth': 3,
            'review_item_count': 3,
            'actionable_review_item_count': 3,
            'next_review_execution_id': 'commodity-paper-execution:metals_all:XAUUSD',
            'next_review_execution_symbol': 'XAUUSD',
            'review_stack_brief': 'paper-execution-review:metals_all:XAUUSD, XAGUSD, COPPER',
            'review_items': [
                {
                    'execution_id': 'commodity-paper-execution:metals_all:XAUUSD',
                    'symbol': 'XAUUSD',
                    'queue_rank': 1,
                    'review_status': 'awaiting_paper_execution_review',
                    'execution_status': 'queued',
                    'weight_hint': 1.0,
                    'regime_gate': 'paper_only',
                },
                {
                    'execution_id': 'commodity-paper-execution:metals_all:XAGUSD',
                    'symbol': 'XAGUSD',
                    'queue_rank': 2,
                    'review_status': 'awaiting_paper_execution_review',
                    'execution_status': 'queued',
                    'weight_hint': 0.8,
                    'regime_gate': 'paper_only',
                },
                {
                    'execution_id': 'commodity-paper-execution:metals_all:COPPER',
                    'symbol': 'COPPER',
                    'queue_rank': 3,
                    'review_status': 'awaiting_paper_execution_review',
                    'execution_status': 'queued',
                    'weight_hint': 0.5,
                    'regime_gate': 'paper_only',
                },
            ],
        },
    )

    proc = subprocess.run(
        ['python3', str(SCRIPT_PATH), '--review-dir', str(review_dir), '--now', '2026-03-10T12:30:00Z'],
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload['execution_retro_status'] == 'paper-execution-retro-pending'
    assert payload['execution_batch'] == 'metals_all'
    assert payload['next_retro_execution_id'] == 'commodity-paper-execution:metals_all:XAUUSD'
    assert payload['next_retro_execution_symbol'] == 'XAUUSD'
    assert payload['retro_item_count'] == 3
    assert payload['actionable_retro_item_count'] == 3
    assert payload['retro_pending_symbols'] == ['XAUUSD', 'XAGUSD', 'COPPER']
    assert payload['fill_evidence_pending_symbols'] == []
    assert payload['retro_stack_brief'] == 'paper-execution-retro:metals_all:XAUUSD, XAGUSD, COPPER'
    assert Path(str(payload['artifact'])).exists()
    assert Path(str(payload['markdown'])).exists()
    assert Path(str(payload['checksum'])).exists()


def test_build_commodity_paper_execution_retro_waits_for_fill_evidence(tmp_path: Path) -> None:
    review_dir = tmp_path / 'review'
    _write_json(
        review_dir / '20260310T122754Z_commodity_paper_execution_review.json',
        {
            'status': 'ok',
            'route_status': 'paper-first',
            'ticket_book_status': 'paper-ready',
            'execution_preview_status': 'paper-execution-ready',
            'execution_artifact_status': 'paper-execution-artifact-ready',
            'execution_queue_status': 'paper-execution-queued',
            'execution_review_status': 'paper-execution-awaiting-fill-evidence',
            'execution_mode': 'paper_only',
            'execution_batch': 'metals_all',
            'execution_symbols': ['XAUUSD'],
            'execution_ticket_ids': ['commodity-paper-ticket:metals_all:XAUUSD'],
            'execution_regime_gate': 'paper_only',
            'execution_weight_hint_sum': 1.0,
            'execution_item_count': 1,
            'actionable_execution_item_count': 1,
            'queue_depth': 1,
            'actionable_queue_depth': 1,
            'review_item_count': 1,
            'actionable_review_item_count': 0,
            'fill_evidence_pending_count': 1,
            'next_review_execution_id': '',
            'next_review_execution_symbol': '',
            'next_fill_evidence_execution_id': 'commodity-paper-execution:metals_all:XAUUSD',
            'next_fill_evidence_execution_symbol': 'XAUUSD',
            'review_stack_brief': 'paper-execution-review:metals_all:XAUUSD',
            'review_items': [
                {
                    'execution_id': 'commodity-paper-execution:metals_all:XAUUSD',
                    'symbol': 'XAUUSD',
                    'queue_rank': 1,
                    'review_status': 'awaiting_paper_execution_fill',
                    'execution_status': 'queued',
                    'paper_execution_evidence_present': False,
                    'weight_hint': 1.0,
                    'regime_gate': 'paper_only',
                }
            ],
        },
    )

    proc = subprocess.run(
        ['python3', str(SCRIPT_PATH), '--review-dir', str(review_dir), '--now', '2026-03-10T12:30:00Z'],
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload['execution_retro_status'] == 'paper-execution-awaiting-fill-evidence'
    assert payload['actionable_retro_item_count'] == 0
    assert payload['fill_evidence_pending_count'] == 1
    assert payload['retro_pending_symbols'] == []
    assert payload['fill_evidence_pending_symbols'] == ['XAUUSD']
    assert payload['next_retro_execution_id'] == ''
    assert payload['next_retro_execution_symbol'] == ''
    assert payload['next_fill_evidence_execution_id'] == 'commodity-paper-execution:metals_all:XAUUSD'
    assert payload['next_fill_evidence_execution_symbol'] == 'XAUUSD'
    assert payload['retro_items'][0]['retro_status'] == 'awaiting_paper_execution_fill'


def test_build_commodity_paper_execution_retro_uses_explicit_review_json(tmp_path: Path) -> None:
    review_dir = tmp_path / 'review'
    old_review = review_dir / '20260310T122754Z_commodity_paper_execution_review.json'
    new_review = review_dir / '20260311T084601Z_commodity_paper_execution_review.json'
    _write_json(
        old_review,
        {
            'status': 'ok',
            'execution_review_status': 'paper-execution-review-pending',
            'execution_batch': 'metals_all',
            'execution_symbols': ['XAUUSD'],
            'review_items': [
                {
                    'execution_id': 'commodity-paper-execution:metals_all:XAUUSD',
                    'symbol': 'XAUUSD',
                    'review_status': 'awaiting_paper_execution_review',
                }
            ],
        },
    )
    _write_json(
        new_review,
        {
            'status': 'ok',
            'execution_review_status': 'paper-execution-review-pending',
            'execution_batch': 'metals_all',
            'execution_symbols': ['COPPER'],
            'review_items': [
                {
                    'execution_id': 'commodity-paper-execution:metals_all:COPPER',
                    'symbol': 'COPPER',
                    'review_status': 'awaiting_paper_execution_review',
                }
            ],
        },
    )

    proc = subprocess.run(
        [
            'python3',
            str(SCRIPT_PATH),
            '--review-dir',
            str(review_dir),
            '--execution-review-json',
            str(new_review),
            '--now',
            '2026-03-10T12:30:00Z',
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload['source_execution_review_artifact'] == str(new_review.resolve())
    assert payload['execution_symbols'] == ['COPPER']


def test_build_commodity_paper_execution_retro_marks_partial_fill_remainder(tmp_path: Path) -> None:
    review_dir = tmp_path / 'review'
    _write_json(
        review_dir / '20260310T122754Z_commodity_paper_execution_review.json',
        {
            'status': 'ok',
            'execution_review_status': 'paper-execution-review-pending-fill-remainder',
            'execution_batch': 'metals_all',
            'execution_symbols': ['XAUUSD', 'XAGUSD'],
            'review_items': [
                {
                    'execution_id': 'commodity-paper-execution:metals_all:XAUUSD',
                    'symbol': 'XAUUSD',
                    'review_status': 'awaiting_paper_execution_review',
                },
                {
                    'execution_id': 'commodity-paper-execution:metals_all:XAGUSD',
                    'symbol': 'XAGUSD',
                    'review_status': 'awaiting_paper_execution_fill',
                },
            ],
        },
    )

    proc = subprocess.run(
        ['python3', str(SCRIPT_PATH), '--review-dir', str(review_dir), '--now', '2026-03-10T12:30:00Z'],
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload['execution_retro_status'] == 'paper-execution-retro-pending-fill-remainder'
    assert payload['actionable_retro_item_count'] == 1
    assert payload['fill_evidence_pending_count'] == 1
    assert payload['retro_pending_symbols'] == ['XAUUSD']
    assert payload['fill_evidence_pending_symbols'] == ['XAGUSD']
    assert payload['next_retro_execution_id'] == 'commodity-paper-execution:metals_all:XAUUSD'
    assert payload['next_retro_execution_symbol'] == 'XAUUSD'
    assert payload['next_fill_evidence_execution_id'] == 'commodity-paper-execution:metals_all:XAGUSD'
    assert payload['next_fill_evidence_execution_symbol'] == 'XAGUSD'


def test_build_commodity_paper_execution_retro_marks_close_evidence_pending_fill_remainder(tmp_path: Path) -> None:
    review_dir = tmp_path / 'review'
    _write_json(
        review_dir / '20260310T122754Z_commodity_paper_execution_review.json',
        {
            'status': 'ok',
            'execution_review_status': 'paper-execution-review-pending-fill-remainder',
            'execution_batch': 'metals_all',
            'execution_symbols': ['XAUUSD', 'XAGUSD'],
            'review_items': [
                {
                    'execution_id': 'commodity-paper-execution:metals_all:XAUUSD',
                    'symbol': 'XAUUSD',
                    'review_status': 'awaiting_paper_execution_close_evidence',
                    'paper_execution_evidence_present': True,
                    'paper_execution_status': 'OPEN',
                },
                {
                    'execution_id': 'commodity-paper-execution:metals_all:XAGUSD',
                    'symbol': 'XAGUSD',
                    'review_status': 'awaiting_paper_execution_fill',
                },
            ],
        },
    )

    proc = subprocess.run(
        ['python3', str(SCRIPT_PATH), '--review-dir', str(review_dir), '--now', '2026-03-10T12:30:00Z'],
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload['execution_retro_status'] == 'paper-execution-close-evidence-pending-fill-remainder'
    assert payload['actionable_retro_item_count'] == 0
    assert payload['retro_pending_symbols'] == []
    assert payload['close_evidence_pending_count'] == 1
    assert payload['close_evidence_pending_symbols'] == ['XAUUSD']
    assert payload['next_close_evidence_execution_id'] == 'commodity-paper-execution:metals_all:XAUUSD'
    assert payload['next_close_evidence_execution_symbol'] == 'XAUUSD'
    assert payload['fill_evidence_pending_symbols'] == ['XAGUSD']
    assert payload['retro_items'][0]['retro_status'] == 'awaiting_paper_execution_close_evidence'


def test_build_commodity_paper_execution_retro_preserves_evidence_snapshots(tmp_path: Path) -> None:
    review_dir = tmp_path / 'review'
    _write_json(
        review_dir / '20260310T122754Z_commodity_paper_execution_review.json',
        {
            'status': 'ok',
            'execution_review_status': 'paper-execution-review-pending',
            'execution_batch': 'metals_all',
            'execution_symbols': ['XAUUSD'],
            'review_items': [
                {
                    'execution_id': 'commodity-paper-execution:metals_all:XAUUSD',
                    'symbol': 'XAUUSD',
                    'queue_rank': 1,
                    'review_status': 'awaiting_paper_execution_close_evidence',
                    'paper_execution_evidence_present': True,
                    'paper_entry_price': 5198.10009765625,
                    'paper_stop_price': 4847.7998046875,
                    'paper_target_price': 5758.58056640625,
                    'paper_quote_usdt': 0.15896067200583952,
                    'paper_order_mode': 'paper_bridge_proxy_reference',
                    'paper_signal_price_reference_source': 'yfinance:GC=F',
                    'paper_execution_evidence_snapshot': {
                        'position': {'entry_price': 5198.10009765625},
                        'ledger': {'order_mode': 'paper_bridge_proxy_reference'},
                        'trade_plan': {'target_price': 5758.58056640625},
                        'executed_plan': {'status': 'OPEN'},
                    },
                }
            ],
        },
    )

    proc = subprocess.run(
        ['python3', str(SCRIPT_PATH), '--review-dir', str(review_dir), '--now', '2026-03-10T12:30:00Z'],
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload['execution_retro_status'] == 'paper-execution-close-evidence-pending'
    assert payload['retro_pending_symbols'] == []
    assert payload['close_evidence_pending_symbols'] == ['XAUUSD']
    assert payload['next_close_evidence_execution_symbol'] == 'XAUUSD'
    row = payload['retro_items'][0]
    assert row['retro_status'] == 'awaiting_paper_execution_close_evidence'
    assert row['paper_execution_evidence_present'] is True
    assert row['paper_entry_price'] == 5198.10009765625
    assert row['paper_order_mode'] == 'paper_bridge_proxy_reference'
    assert row['paper_execution_evidence_snapshot']['trade_plan']['target_price'] == 5758.58056640625
    assert 'entry=`5198.100098`' in Path(payload['markdown']).read_text(encoding='utf-8')
