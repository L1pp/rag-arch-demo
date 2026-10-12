"""Evidence-qualified lifecycle views for registered whole-standard families.

The caller must pass ONLY confirmed whole-standard successor edges. Technical
inheritance, adoption, aliases and partial/clause repeal are not successor edges.
No metadata claim, document availability or edition year establishes legal status.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime
import re


def _key(value):
    return re.sub(r"\s+", "", str(value or "")).upper().translate(str.maketrans({
        "－": "-", "—": "-", "–": "-", "：": ":", "／": "/", "．": ".",
        "（": "(", "）": ")"}))


def _date(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10]) if value else None
    except (TypeError, ValueError):
        return None


def _rows(value, field):
    return list(value.get(field, [])) if isinstance(value, dict) else list(value or [])


def _codes(source):
    value = source.get("std_nos", [])
    return [value] if isinstance(value, str) else list(value or [])


def collect_status_observations(catalog):
    """Copy only explicit status observations, never promote metadata_claims.

    Qualification is performed by LifecycleRegistry. Missing std_nos is filled
    from the containing catalog item; evidence fields are not invented.
    """
    observations = []
    for entry in _rows(catalog, "standards"):
        coverage = entry.get("history_coverage") or {}
        if not isinstance(coverage, dict):
            continue
        for observation in coverage.get("status_observations", []) or []:
            if not isinstance(observation, dict):
                continue
            item = deepcopy(observation)
            if not _codes(item):
                code = entry.get("code") or entry.get("std_no")
                item["std_nos"] = [code] if code else []
            observations.append(item)
    return observations


class LifecycleRegistry:
    """Resolve exact editions using saved observations and dated whole edges.

    This is an offline, incomplete registry. ``as_of_date`` bounds both observed
    evidence and implementation events; it is not a claim of live verification.
    ``whole_edges`` must already be confirmed and semantically allowlisted by
    the caller. The helper intentionally does not classify relationship types.
    """

    def __init__(self, catalog, sources, whole_edges, as_of_date=None):
        self.as_of = _date(as_of_date) if as_of_date is not None else date.today()
        if self.as_of is None:
            raise ValueError("as_of_date must be an ISO date")
        self.catalog = {}
        self.names = {}
        self.observations = {}
        self.edges = []
        self.adjacent = {}
        for entry in _rows(catalog, "standards"):
            code = entry.get("code") or entry.get("std_no")
            if code:
                self.names[_key(code)] = code
                self.catalog[_key(code)] = entry
        # Deduplicate identical embedded/external evidence without dropping
        # competing observations that happen to share a source_id.
        observations = _rows(sources, "sources") + collect_status_observations(catalog)
        for source in observations:
            if not isinstance(source, dict):
                continue
            checked = _date(source.get("checked_at"))
            if (source.get("supports_current_status") is not True
                    or source.get("evidence_level") != "current_registry"
                    or checked is None or checked > self.as_of):
                continue
            for code in _codes(source):
                key = _key(code)
                if not key:
                    continue
                self.names.setdefault(key, code)
                group = self.observations.setdefault(key, [])
                if source not in group:
                    group.append(deepcopy(source))
        for edge in whole_edges or []:
            old = edge.get("source_code") or edge.get("old_code")
            new = edge.get("target_code") or edge.get("new_code")
            if not old or not new or _key(old) == _key(new):
                continue
            left, right = _key(old), _key(new)
            self.names.setdefault(left, old)
            self.names.setdefault(right, new)
            self.adjacent.setdefault(left, set()).add(right)
            self.adjacent.setdefault(right, set()).add(left)
            self.edges.append((left, right, deepcopy(edge)))

    def _latest(self, key):
        values = self.observations.get(key, [])
        if not values:
            return []
        newest = max(_date(item["checked_at"]) for item in values)
        return [item for item in values if _date(item["checked_at"]) == newest]

    def _edge_date(self, edge):
        left, right, record = edge
        for field in ("target_effective_date", "effective_date", "implementation_date"):
            value = _date(record.get(field))
            if value:
                return value
        dates = {_date(source.get("effective_date")) for source in self._latest(right)} - {None}
        return next(iter(dates)) if len(dates) == 1 else None

    @staticmethod
    def _observed_status(source):
        value = str(source.get("observed_status", "")).lower()
        return {"withdrawn": "abolished", "repealed": "abolished",
                "upcoming": "not_yet_effective", "published_not_effective": "not_yet_effective"}.get(value, value)

    def _version(self, key):
        selected = self._latest(key)
        checked = max((_date(s["checked_at"]) for s in selected), default=None)
        statuses = {self._observed_status(s) for s in selected}
        status = next(iter(statuses)) if len(statuses) == 1 else "unknown"
        if status not in {"current", "abolished", "not_yet_effective"}:
            status = "unknown"
        conflicts = []
        needs_review = False
        if len(statuses) > 1:
            conflicts.append({"kind": "same_date_status_conflict", "observed_statuses": sorted(statuses),
                              "source_ids": [s.get("source_id") for s in selected if s.get("source_id")]})
        date_values = {}
        for field in ("publish_date", "effective_date"):
            values = {_date(s.get(field)) for s in selected} - {None}
            date_values[field] = next(iter(values)) if len(values) == 1 else None
            if len(values) > 1:
                conflicts.append({"kind": "observation_date_conflict", "field": field,
                                  "values": sorted(value.isoformat() for value in values)})
        published, effective = date_values["publish_date"], date_values["effective_date"]
        if published and published > self.as_of:
            status = "unknown"
        elif effective and effective > self.as_of and status == "current":
            # Registries may label a published but unimplemented standard current.
            status = "not_yet_effective"
        if status == "not_yet_effective" and effective and effective <= self.as_of:
            status, needs_review = "unknown", True
        level = "current_registry" if selected and status != "unknown" else "insufficient_evidence"
        events = [edge for edge in self.edges if edge[0] == key
                  and self._edge_date(edge) is not None and self._edge_date(edge) <= self.as_of]
        derived = []
        if events:
            first_event = min(self._edge_date(edge) for edge in events)
            if status == "current" and checked >= first_event:
                conflicts.append({"kind": "current_observation_after_replacement_effective",
                                  "event_date": first_event.isoformat(),
                                  "relation_ids": [e[2].get("relation_id", e[2].get("id")) for e in events]})
            elif status != "abolished" and not conflicts:
                status, level = "abolished", "confirmed_whole_replacement"
                derived = [dict(e[2], evidence_level="confirmed_whole_replacement",
                                derived_status="abolished", event_effective_date=self._edge_date(e).isoformat())
                           for e in events]
                event_checks = [_date(row.get("checked_at") or row.get("verified_at")) for row in derived]
                known_checks = [value for value in [checked, *event_checks] if value is not None and value <= self.as_of]
                checked = max(known_checks) if known_checks else checked
        if conflicts:
            status, level, needs_review = "unknown", "conflicting_evidence", True
        entry = self.catalog.get(key, {})
        return {
            "std_no": self.names[key], "standard_status": status,
            "standard_status_evidence_level": level,
            "status_checked_at": checked.isoformat() if checked else None,
            "status_sources": deepcopy(selected) + derived, "status_conflicts": conflicts,
            "publish_date": published.isoformat() if published else None,
            "effective_date": effective.isoformat() if effective else None,
            "requires_status_recheck": needs_review,
            "library_class": entry.get("library_class"),
            "status_observation_history": deepcopy(self.observations.get(key, [])),
        }

    def _walk(self, start, adjacency):
        seen, queue = set(), [start]
        while queue:
            key = queue.pop()
            if key in seen:
                continue
            seen.add(key)
            queue.extend(adjacency.get(key, set()) - seen)
        return seen

    def describe(self, query):
        key = _key(query)
        family = self._walk(key, self.adjacent) if key in self.names else set()
        versions = {code: self._version(code) for code in family}
        directed, active = {}, {}
        published = set()
        upcoming = []
        for code, item in versions.items():
            publication = _date(item["publish_date"])
            observations = self._latest(code)
            has_current_or_abolished = any(self._observed_status(s) in {"current", "abolished"} for s in observations)
            if ((publication and publication <= self.as_of)
                    or publication is None and has_current_or_abolished):
                published.add(code)
            effective = _date(item["effective_date"])
            if code in published and effective and effective > self.as_of:
                upcoming.append({"std_no": self.names[code], "publish_date": item["publish_date"],
                                 "effective_date": item["effective_date"],
                                 "source_ids": list(dict.fromkeys(s["source_id"] for s in observations if s.get("source_id")))})
        for edge in self.edges:
            left, right, record = edge
            if left not in family:
                continue
            directed.setdefault(left, set()).add(right)
            effective = self._edge_date(edge)
            if effective and effective <= self.as_of:
                active.setdefault(left, set()).add(right)
                published.update((left, right))
            publication = _date(record.get("target_publish_date"))
            if publication and publication <= self.as_of:
                published.add(right)
        descendants = self._walk(key, directed) - {key}
        effective_descendants = self._walk(key, active) - {key}
        latest = [code for code in published if not ((self._walk(code, directed) - {code}) & published)]
        # Conflicting family observations remain visible in version_statuses;
        # only directly supported current versions qualify for the current list.
        current = [code for code, item in versions.items() if item["standard_status"] == "current"]
        future = {code for code in descendants if versions[code]["standard_status"] == "not_yet_effective"}
        future.update(right for edge in self.edges for left, right, record in [edge]
                      if right in descendants and self._edge_date(edge) and self._edge_date(edge) > self.as_of)
        def names(keys):
            return sorted((self.names[code] for code in keys), key=_key)
        selected = versions.get(key, {"standard_status": "unknown",
            "standard_status_evidence_level": "insufficient_evidence", "status_checked_at": None,
            "status_sources": [], "status_conflicts": [], "requires_status_recheck": False})
        result = {field: deepcopy(selected[field]) for field in (
            "standard_status", "standard_status_evidence_level", "status_checked_at", "status_sources",
            "status_conflicts", "requires_status_recheck")}
        result.update(status_as_of=self.as_of.isoformat(), current_effective_versions=names(current),
            latest_published_versions=names(latest), upcoming_versions=sorted(upcoming, key=lambda x: _key(x["std_no"])),
            known_effective_successors=names(effective_descendants - future), known_future_successors=names(future),
            version_statuses=[versions[code] for code in sorted(versions)], complete_registry_coverage=False,
            latest_published_scope="已登记且具有发布或生效依据的整本沿革范围；不代表全球实时完整目录")
        for field in ("current_effective", "latest_published"):
            values = result[field + "_versions"]
            result[field + "_version"] = values[0] if len(values) == 1 else None
        return result
