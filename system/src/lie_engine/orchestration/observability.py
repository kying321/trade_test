from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

from lie_engine.config import SystemSettings
from lie_engine.data.storage import collect_sqlite_stats


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
        sqlite_stats = collect_sqlite_stats(self.sqlite_path)
        page_count = int(sqlite_stats.get("page_count", 0) or 0)
        freelist_count = int(sqlite_stats.get("freelist_count", 0) or 0)
        freelist_ratio = (float(freelist_count) / float(page_count)) if page_count > 0 else 0.0
        file_bytes = int(sqlite_stats.get("file_bytes", 0) or 0)
        val = self.settings.validation if isinstance(self.settings.validation, dict) else {}
        size_warn = int(float(val.get("ops_health_sqlite_size_warn_bytes", 1_500_000_000)))
        freelist_warn = float(val.get("ops_health_sqlite_freelist_warn_ratio", 0.20))
        sqlite_alerts: list[str] = []
        if file_bytes >= max(1, size_warn):
            sqlite_alerts.append("sqlite_size_warn")
        if freelist_ratio >= max(0.0, freelist_warn):
            sqlite_alerts.append("sqlite_freelist_warn")
        sqlite_health = {
            "db_exists": bool(sqlite_stats.get("exists", False)),
            "file_bytes": int(file_bytes),
            "page_count": int(page_count),
            "page_size": int(sqlite_stats.get("page_size", 0) or 0),
            "freelist_count": int(freelist_count),
            "freelist_ratio": float(freelist_ratio),
            "size_warn_bytes": int(size_warn),
            "freelist_warn_ratio": float(freelist_warn),
            "alerts": sqlite_alerts,
        }
        return {
            "date": day,
            "status": status,
            "checks": checks,
            "missing": missing,
            "sqlite_health": sqlite_health,
        }

    def stable_replay_check(
        self,
        as_of: date,
        days: int | None = None,
        run_eod_replay: bool = True,
    ) -> dict[str, Any]:
        replay_days = int(days or self.settings.validation.get("required_stable_replay_days", 3))
        replay_days = max(1, replay_days)

        checks: list[dict[str, Any]] = []
        all_passed = True
        for i in range(replay_days):
            target = as_of - timedelta(days=i)
            try:
                if run_eod_replay:
                    self.run_eod(target)
                health = self.health_check(target, require_review=(i == 0))
                day_ok = bool(health["status"] == "healthy")
                checks.append(
                    {
                        "date": target.isoformat(),
                        "ok": day_ok,
                        "health": health,
                        "replay_executed": bool(run_eod_replay),
                    }
                )
                if not day_ok:
                    all_passed = False
            except Exception as exc:
                checks.append({"date": target.isoformat(), "ok": False, "error": str(exc)})
                all_passed = False

        return {
            "as_of": as_of.isoformat(),
            "replay_days": replay_days,
            "replay_executed": bool(run_eod_replay),
            "passed": all_passed,
            "checks": checks,
        }
