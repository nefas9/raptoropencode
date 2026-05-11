"""Tests for ``packages.sca.bump.evaluator``.

The evaluator emits ``SupplyChainFinding`` rows for bump-tier
detectors; ``review._compute_verdict`` consumes them via the
``bump_supply_chain_findings=`` parameter. These tests cover the
recent_publish detector (Phase 1.b's only detector); subsequent
detectors get their own test groups."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import pytest

from packages.sca.bump.evaluator import evaluate_bump_supply_chain


# ---------------------------------------------------------------------------
# Stub registry clients
# ---------------------------------------------------------------------------

class _StubPyPIClient:
    """Minimal stand-in for ``PyPIClient`` exposing ``get_metadata``
    with operator-supplied per-version upload times."""

    def __init__(self, packages: Dict[str, Dict[str, Any]]):
        self._packages = packages

    def get_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        return self._packages.get(name)


class _StubNpmClient:
    """Minimal stand-in for ``NpmClient`` exposing ``get_metadata``
    with operator-supplied per-version time map."""

    def __init__(self, packages: Dict[str, Dict[str, Any]]):
        self._packages = packages

    def get_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        return self._packages.get(name)


# ---------------------------------------------------------------------------
# recent_publish detector
# ---------------------------------------------------------------------------

def test_recent_publish_target_published_today_fires() -> None:
    """Target version published <30 days ago → ``recent_publish``
    finding at medium severity. The rapid-release window is the
    most defensible bump-tier signal: an attacker publishes
    malicious v1.2.4 and hopes auto-bumpers pull it in before
    takedown."""
    now = datetime(2026, 5, 11, tzinfo=timezone.utc)
    pypi = _StubPyPIClient({
        "django": {"releases": {
            "4.2.30": [{"upload_time_iso_8601": "2026-05-10T00:00:00Z"}],
        }},
    })
    findings = evaluate_bump_supply_chain(
        ecosystem="PyPI", name="django",
        current_version="4.2.10", target_version="4.2.30",
        pypi_client=pypi, npm_client=None, now=now,
    )
    assert len(findings) == 1
    f = findings[0]
    assert f.kind == "recent_publish"
    assert f.severity == "medium"
    assert "2026-05-10" in f.detail
    # Evidence carries machine-readable fields for the PR-comment renderer.
    assert f.evidence["age_days"] == 1
    assert f.evidence["target_version"] == "4.2.30"


def test_recent_publish_target_older_than_threshold_silent() -> None:
    """Target published >30 days ago → no finding. The detector
    is silent unless the rapid-release window is open."""
    now = datetime(2026, 5, 11, tzinfo=timezone.utc)
    pypi = _StubPyPIClient({
        "django": {"releases": {
            "4.2.26": [{"upload_time_iso_8601": "2025-12-01T00:00:00Z"}],
        }},
    })
    findings = evaluate_bump_supply_chain(
        ecosystem="PyPI", name="django",
        current_version="4.2.10", target_version="4.2.26",
        pypi_client=pypi, npm_client=None, now=now,
    )
    assert findings == []


def test_recent_publish_npm_target_recent_fires() -> None:
    """npm packument's ``time[version]`` field gives the
    per-version publish timestamp. Target <30 days ago → finding."""
    now = datetime(2026, 5, 11, tzinfo=timezone.utc)
    npm = _StubNpmClient({
        "lodash": {"time": {
            "4.17.30": "2026-04-30T12:00:00.000Z",
        }},
    })
    findings = evaluate_bump_supply_chain(
        ecosystem="npm", name="lodash",
        current_version="4.17.21", target_version="4.17.30",
        pypi_client=None, npm_client=npm, now=now,
    )
    assert len(findings) == 1
    assert findings[0].kind == "recent_publish"
    # Apr 30 12:00 → May 11 00:00 = 10 days 12 hours; timedelta.days
    # floors to whole 24-hour periods.
    assert findings[0].evidence["age_days"] == 10


def test_recent_publish_threshold_boundary() -> None:
    """A target published exactly at the threshold boundary is
    NOT flagged (strict less-than comparison). Off-by-one
    matters: an operator who set the threshold to 30 days
    expects "no flag at 30 days old" — they'd already have moved
    past the rapid-release window."""
    now = datetime(2026, 5, 11, tzinfo=timezone.utc)
    pypi = _StubPyPIClient({
        "foo": {"releases": {
            "1.0.0": [{"upload_time_iso_8601": "2026-04-11T00:00:00Z"}],
        }},
    })
    findings = evaluate_bump_supply_chain(
        ecosystem="PyPI", name="foo",
        current_version="0.9", target_version="1.0.0",
        pypi_client=pypi, npm_client=None, now=now,
        rapid_release_days=30,
    )
    assert findings == []


def test_recent_publish_custom_threshold() -> None:
    """Operators with stricter policies can tighten the rapid-
    release window. A 60-day window flags a 45-day-old release;
    a 14-day window doesn't flag a 21-day-old release."""
    now = datetime(2026, 5, 11, tzinfo=timezone.utc)
    pypi = _StubPyPIClient({
        "foo": {"releases": {
            "1.0.0": [{"upload_time_iso_8601": "2026-03-27T00:00:00Z"}],
        }},
    })
    strict = evaluate_bump_supply_chain(
        ecosystem="PyPI", name="foo",
        current_version="0.9", target_version="1.0.0",
        pypi_client=pypi, npm_client=None, now=now,
        rapid_release_days=60,
    )
    assert len(strict) == 1

    relaxed = evaluate_bump_supply_chain(
        ecosystem="PyPI", name="foo",
        current_version="0.9", target_version="1.0.0",
        pypi_client=pypi, npm_client=None, now=now,
        rapid_release_days=14,
    )
    assert relaxed == []


def test_missing_target_version_in_releases_silent() -> None:
    """Target version not listed in registry's releases map → no
    finding (can't compute the date). The bumper sees an empty
    list and falls through to vuln-only verdict."""
    now = datetime(2026, 5, 11, tzinfo=timezone.utc)
    pypi = _StubPyPIClient({
        "foo": {"releases": {"0.9": []}},   # 1.0.0 unknown
    })
    findings = evaluate_bump_supply_chain(
        ecosystem="PyPI", name="foo",
        current_version="0.9", target_version="1.0.0",
        pypi_client=pypi, npm_client=None, now=now,
    )
    assert findings == []


def test_unsupported_ecosystem_returns_empty() -> None:
    """Ecosystems without per-version publish-date lookup
    (Maven / Cargo / Go / others — future-detector territory)
    return [] silently. The bumper treats the absence as "no
    bump-tier signal available" and falls through."""
    now = datetime(2026, 5, 11, tzinfo=timezone.utc)
    findings = evaluate_bump_supply_chain(
        ecosystem="Maven", name="org.foo:bar",
        current_version="1.0", target_version="2.0",
        pypi_client=None, npm_client=None, now=now,
    )
    assert findings == []


def test_missing_client_for_ecosystem_returns_empty() -> None:
    """If the right ecosystem client is None (e.g. caller built
    a PyPI scan without an NpmClient and is then evaluating an
    npm bump), we get [] silently rather than raising."""
    now = datetime(2026, 5, 11, tzinfo=timezone.utc)
    findings = evaluate_bump_supply_chain(
        ecosystem="npm", name="x",
        current_version="1", target_version="2",
        pypi_client=None, npm_client=None, now=now,
    )
    assert findings == []


def test_registry_returns_none_handled_gracefully() -> None:
    """Registry-client ``get_metadata`` returns None (404 / offline /
    cache miss) → no finding, no exception."""
    now = datetime(2026, 5, 11, tzinfo=timezone.utc)
    pypi = _StubPyPIClient({})       # empty: every lookup returns None
    findings = evaluate_bump_supply_chain(
        ecosystem="PyPI", name="missing",
        current_version="1", target_version="2",
        pypi_client=pypi, npm_client=None, now=now,
    )
    assert findings == []


def test_finding_id_includes_target_version() -> None:
    """Bump-tier findings include the target version in their
    finding_id so repeat-bump-evaluations don't dedup against
    each other in the bumper's PR-comment renderer."""
    now = datetime(2026, 5, 11, tzinfo=timezone.utc)
    pypi = _StubPyPIClient({
        "x": {"releases": {
            "2.0.0": [{"upload_time_iso_8601": "2026-05-10T00:00:00Z"}],
        }},
    })
    findings = evaluate_bump_supply_chain(
        ecosystem="PyPI", name="x",
        current_version="1.0", target_version="2.0.0",
        pypi_client=pypi, npm_client=None, now=now,
    )
    assert "2.0.0" in findings[0].finding_id
    assert "PyPI" in findings[0].finding_id


def test_pypi_chooses_earliest_upload_time_across_files() -> None:
    """A PyPI release can have multiple distribution files (.whl
    for each platform + .tar.gz source). The earliest upload
    timestamp is the canonical publish moment — picking the
    latest would falsely make the version look more recent than
    it is."""
    now = datetime(2026, 5, 11, tzinfo=timezone.utc)
    pypi = _StubPyPIClient({
        "x": {"releases": {
            "2.0.0": [
                # 60 days ago — outside the 30-day window
                {"upload_time_iso_8601": "2026-03-12T00:00:00Z"},
                # 10 days ago — would be inside the window if picked
                {"upload_time_iso_8601": "2026-05-01T00:00:00Z"},
            ],
        }},
    })
    findings = evaluate_bump_supply_chain(
        ecosystem="PyPI", name="x",
        current_version="1.9", target_version="2.0.0",
        pypi_client=pypi, npm_client=None, now=now,
    )
    # Earliest upload = 60 days ago = outside threshold → no finding.
    assert findings == []
