"""Bump-time supply-chain evaluator.

Given a proposed ``(current_version, target_version)`` bump for one
dep, emits ``SupplyChainFinding`` rows for whichever bump-tier
detectors fire. The verdict ladder
(``review._compute_verdict``) consumes the result to gate the bump:

  * ``high``+ finding → Block
  * ``medium`` finding → escalate to Review
  * Two or more ``medium``+ findings → Block (compound red flags)

Detector roster (Phase 1.b ships only the first; the others gate
on per-ecosystem metadata extraction work):

  * ``recent_publish``      — target version published <N days ago
                              (rapid-release attack class)
  * ``maintainer_change``   — maintainer set differs between current
                              and target's publish windows
                              (account-takeover / handover class)
  * ``install_hook_delta``  — target adds install hooks the current
                              version didn't have (payload injection
                              class)

Per-ecosystem metadata access varies:

  * **npm**: per-version maintainers + dependencies + scripts via
    ``versions[v].maintainers / dependencies / scripts``. Best
    surface for all three detectors.
  * **PyPI**: per-version upload timestamps via
    ``releases[v][n].upload_time_iso_8601``; package-level
    maintainers only. Supports ``recent_publish`` precisely;
    ``maintainer_change`` is best-effort proxy at package level.
  * Other ecosystems: minimal per-version surface; for now the
    evaluator returns an empty list with a debug log."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

from ..models import Confidence, Dependency, PinStyle, SupplyChainFinding

logger = logging.getLogger(__name__)

# Default rapid-release window. A target version published less
# than this many days ago surfaces as ``recent_publish`` at medium
# severity. 30 days matches the operationally accepted window for
# "the package's been in the wild long enough that obvious bad
# behaviour would have been reported".
_RAPID_RELEASE_DAYS = 30


def evaluate_bump_supply_chain(
    *,
    ecosystem: str,
    name: str,
    current_version: str,
    target_version: str,
    pypi_client=None,
    npm_client=None,
    now: Optional[datetime] = None,
    rapid_release_days: int = _RAPID_RELEASE_DAYS,
) -> List[SupplyChainFinding]:
    """Return the bump-tier supply-chain findings for a proposed bump.

    Callers wire the per-ecosystem registry clients in (already
    cached / offline-aware / egress-allowlisted). Missing clients
    or unsupported ecosystems return an empty list — the bumper
    treats that as "no bump-tier signals available, fall through
    to vuln-only verdict".
    """
    now = now or datetime.now(timezone.utc)
    findings: List[SupplyChainFinding] = []

    target_publish = _target_publish_date(
        ecosystem=ecosystem, name=name, version=target_version,
        pypi_client=pypi_client, npm_client=npm_client,
    )
    if target_publish is not None:
        age = now - target_publish
        if age < timedelta(days=rapid_release_days):
            findings.append(_recent_publish_finding(
                ecosystem=ecosystem, name=name,
                target_version=target_version,
                target_publish=target_publish,
                age=age, threshold=rapid_release_days,
            ))
    else:
        logger.debug(
            "sca.bump: no publish date available for %s:%s@%s; "
            "skipping recent_publish detector",
            ecosystem, name, target_version,
        )

    return findings


# ---------------------------------------------------------------------------
# Per-ecosystem publish-date extraction
# ---------------------------------------------------------------------------

def _target_publish_date(
    *,
    ecosystem: str,
    name: str,
    version: str,
    pypi_client,
    npm_client,
) -> Optional[datetime]:
    """Return the publish datetime for ``ecosystem:name@version``
    via the appropriate registry client, or ``None`` if the
    registry doesn't expose it or the lookup fails.
    """
    if ecosystem == "PyPI" and pypi_client is not None:
        return _pypi_version_publish(name, version, pypi_client)
    if ecosystem == "npm" and npm_client is not None:
        return _npm_version_publish(name, version, npm_client)
    # Other ecosystems land here. Per-version publish-date lookup
    # is doable for Cargo (crates.io API) / RubyGems / NuGet /
    # Maven Central (rest/v2) — future detector commits add them.
    return None


def _pypi_version_publish(
    name: str, version: str, client,
) -> Optional[datetime]:
    """Earliest upload_time across the version's distribution files."""
    meta = client.get_metadata(name)
    if not isinstance(meta, dict):
        return None
    releases = meta.get("releases") or {}
    files = releases.get(version)
    if not files:
        return None
    earliest: Optional[datetime] = None
    for entry in files:
        if not isinstance(entry, dict):
            continue
        ts = entry.get("upload_time_iso_8601") or entry.get("upload_time")
        if not isinstance(ts, str):
            continue
        parsed = _parse_iso(ts)
        if parsed is None:
            continue
        if earliest is None or parsed < earliest:
            earliest = parsed
    return earliest


def _npm_version_publish(
    name: str, version: str, client,
) -> Optional[datetime]:
    """``time[version]`` field of the npm packument."""
    meta = client.get_metadata(name)
    if not isinstance(meta, dict):
        return None
    times = meta.get("time") or {}
    ts = times.get(version)
    if not isinstance(ts, str):
        return None
    return _parse_iso(ts)


def _parse_iso(ts: str) -> Optional[datetime]:
    """ISO-8601 parser that tolerates trailing ``Z`` and missing
    fractional seconds (covers both PyPI and npm shapes)."""
    cleaned = ts.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(cleaned)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Finding constructors
# ---------------------------------------------------------------------------

def _recent_publish_finding(
    *,
    ecosystem: str,
    name: str,
    target_version: str,
    target_publish: datetime,
    age: timedelta,
    threshold: int,
) -> SupplyChainFinding:
    """Construct a ``recent_publish`` SupplyChainFinding for the target.

    Severity ``medium``: the rapid-release window is a Review-tier
    signal alone (operators may legitimately track unstable
    releases). It compounds to Block via the verdict ladder when
    paired with another medium+ bump-tier finding.

    The ``Dependency`` row carries the proposed target's
    coordinates so PR-comment rendering shows the right
    ``eco:name@version`` in the verdict table.
    """
    placeholder_dep = Dependency(
        ecosystem=ecosystem,
        name=name,
        version=target_version,
        declared_in=Path("/<bump>"),
        scope="main",
        is_lockfile=False,
        pin_style=PinStyle.EXACT,
        direct=True,
        purl=f"pkg:{ecosystem.lower()}/{name}@{target_version}",
        parser_confidence=Confidence(
            "high",
            reason="bump-evaluator synthetic dep",
        ),
    )
    days = max(0, age.days)
    return SupplyChainFinding(
        finding_id=(
            f"sca:bump:recent_publish:{ecosystem}:{name}@{target_version}"
        ),
        kind="recent_publish",
        dependency=placeholder_dep,
        detail=(
            f"target version {target_version} published "
            f"{target_publish.date().isoformat()} "
            f"({days} day(s) ago; rapid-release threshold "
            f"{threshold})"
        ),
        evidence={
            "target_version": target_version,
            "target_publish": target_publish.isoformat(),
            "age_days": days,
            "threshold_days": threshold,
        },
        severity="medium",
        confidence=Confidence(
            "high",
            reason="publish timestamp from registry",
        ),
    )
