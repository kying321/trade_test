from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

from lie_engine.research.commodity_reasoning_boundary import build_commodity_reasoning_boundary_strength
from lie_engine.research.commodity_reasoning_scenario import build_commodity_reasoning_scenario_tree
from lie_engine.research.commodity_reasoning_summary import build_commodity_reasoning_summary
from lie_engine.research.commodity_reasoning_transmission import build_commodity_reasoning_transmission_map
from lie_engine.research.commodity_reasoning_validation import build_commodity_reasoning_validation_ring


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_pipeline(
    *,
    output_root: Path,
    contract_focus: str,
    event_artifacts: Sequence[Mapping[str, object]],
    research_artifacts: Sequence[Mapping[str, object]],
    cross_section_news: Sequence[Mapping[str, object]],
    cross_section_data: Sequence[Mapping[str, object]],
    generated_at: str,
) -> Mapping[str, Path]:
    scenario_tree = build_commodity_reasoning_scenario_tree(
        event_artifacts=event_artifacts,
        research_artifacts=research_artifacts,
        contract_focus=contract_focus,
        generated_at=generated_at,
    )
    transmission_map = build_commodity_reasoning_transmission_map(
        scenario_tree=scenario_tree,
        contract_focus=contract_focus,
        generated_at=generated_at,
    )
    validation_ring = build_commodity_reasoning_validation_ring(
        transmission_map=transmission_map,
        cross_section_news=cross_section_news,
        cross_section_data=cross_section_data,
        generated_at=generated_at,
    )
    boundary_strength = build_commodity_reasoning_boundary_strength(
        transmission_map=transmission_map,
        validation_ring=validation_ring,
        generated_at=generated_at,
    )
    summary = build_commodity_reasoning_summary(
        scenario_tree=scenario_tree,
        transmission_map=transmission_map,
        boundary_strength=boundary_strength,
        generated_at=generated_at,
    )

    review_dir = output_root / "review"
    paths = {
        "scenario_tree": review_dir / "latest_commodity_reasoning_scenario_tree.json",
        "transmission_map": review_dir / "latest_commodity_reasoning_transmission_map.json",
        "boundary_strength": review_dir / "latest_commodity_reasoning_boundary_strength.json",
        "summary": review_dir / "latest_commodity_reasoning_summary.json",
    }
    _write_json(paths["scenario_tree"], scenario_tree)
    _write_json(paths["transmission_map"], transmission_map)
    _write_json(paths["boundary_strength"], boundary_strength)
    _write_json(paths["summary"], summary)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run domestic commodity reasoning and write latest review artifacts.")
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--contract", default="BU2606")
    parser.add_argument("--now", required=True)
    args = parser.parse_args()
    artifacts = run_pipeline(
        output_root=Path(args.output_root),
        contract_focus=args.contract,
        event_artifacts=[],
        research_artifacts=[],
        cross_section_news=[],
        cross_section_data=[],
        generated_at=args.now,
    )
    print(json.dumps({"artifacts": {k: str(v) for k, v in artifacts.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
