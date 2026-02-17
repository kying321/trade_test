from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

from lie_engine.config import SystemSettings


@dataclass(slots=True)
class ObservabilityOrchestrator:
    settings: SystemSettings
    output_dir: Path
    sqlite_path: Path
    run_eod: Callable[[date], dict[str, Any]]

    def health_check(self, as_of: date | None = None, require_review: bool = True) -> dict[str, Any]:
        tz = ZoneInfo(self.settings.timezone)
        target = as_of or datetime.now(tz).date()
        day = target.isoformat()

        daily_dir = self.output_dir / "daily"
        review_dir = self.output_dir / "review"
        required = {
            "daily_briefing": daily_dir / f"{day}_briefing.md",
            "daily_signals": daily_dir / f"{day}_signals.json",
            "daily_positions": daily_dir / f"{day}_positions.csv",
            "sqlite": self.sqlite_path,
        }
        if require_review:
            required["review_report"] = review_dir / f"{day}_review.md"
            required["review_delta"] = review_dir / f"{day}_param_delta.yaml"

        checks = {k: p.exists() for k, p in required.items()}
        missing = [k for k, ok in checks.items() if not ok]
        status = "healthy" if not missing else "degraded"
        return {
            "date": day,
            "status": status,
            "checks": checks,
            "missing": missing,
        }

    def stable_replay_check(self, as_of: date, days: int | None = None) -> dict[str, Any]:
        replay_days = int(days or self.settings.validation.get("required_stable_replay_days", 3))
        replay_days = max(1, replay_days)

        checks: list[dict[str, Any]] = []
        all_passed = True
        for i in range(replay_days):
            target = as_of - timedelta(days=i)
            try:
                self.run_eod(target)
                health = self.health_check(target, require_review=(i == 0))
                day_ok = bool(health["status"] == "healthy")
                checks.append({"date": target.isoformat(), "ok": day_ok, "health": health})
                if not day_ok:
                    all_passed = False
            except Exception as exc:
                checks.append({"date": target.isoformat(), "ok": False, "error": str(exc)})
                all_passed = False

        return {
            "as_of": as_of.isoformat(),
            "replay_days": replay_days,
            "passed": all_passed,
            "checks": checks,
        }
