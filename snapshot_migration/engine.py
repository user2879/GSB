"""Snapshot migration engine: packing, inheritance, cleanup and snapshots."""

import hashlib
import json
from typing import Dict, List, Optional, Tuple

from .chain import parse_chain
from .errors import CleanError, InheritError
from .fragments import (Fragment, canonical_bytes, fragment_digest,
                        make_fragment_id, pack_range,
                        verify_fragment_invariants)
from .model import Generation, KeyRange, SnapshotChain
from .repository import SCHEMA_VERSION, Repository
from .readview import ReadView

SNAPSHOT_FORMAT_VERSION = 3


class MigrationEngine:
    def __init__(self, db_path, chain: SnapshotChain):
        self.db_path = db_path
        self.chain = chain
        self.repo = Repository(db_path)
        self._seq = 0

    # ------------------------------------------------------------ helpers
    def _next_seq(self):
        self._seq += 1
        return self._seq

    def _parent_fragments_by_range(self, parent_id):
        result: Dict[str, List[Fragment]] = {}
        for frag in self.repo.active_fragments(parent_id):
            result.setdefault(frag.range_id, []).append(frag)
        for items in result.values():
            items.sort(key=lambda item: item.ordinal)
        return result

    def _generation_digest(self, generation: Generation):
        payload = [generation.generation_id, generation.seq, generation.parent_id]
        for item in sorted(generation.ranges, key=lambda r: r.range_id):
            payload.append([item.range_id, item.lower, item.upper,
                            item.action, item.tombstone, list(item.keys)])
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")).hexdigest()

    # ------------------------------------------------------------ migration
    def migrate_generation(self, generation_id) -> dict:
        gen = self.chain.generation(generation_id)
        if gen is None:
            raise InheritError("unknown generation", generation_id=generation_id)
        if self.repo.is_committed(generation_id):
            return self._migration_summary(generation_id)

        parent_map = {}
        if gen.parent_id:
            if not self.repo.is_committed(gen.parent_id):
                raise InheritError(
                    "parent generation is not committed before migration",
                    generation_id=generation_id)
            parent_map = self._parent_fragments_by_range(gen.parent_id)

        report = {"generation_id": generation_id, "ranges": []}
        with self.repo.transaction():
            self.repo.upsert_generation(
                generation_id, gen.seq, gen.parent_id,
                self._generation_digest(gen), committed=False)

            for item in gen.ranges:
                if item.action == "inherit":
                    entry = self._apply_inherit(gen, item, parent_map)
                elif item.action == "rewrite":
                    entry = self._apply_rewrite(gen, item, parent_map)
                elif item.action == "new":
                    entry = self._apply_new(gen, item)
                else:  # discard
                    entry = self._apply_discard(gen, item, parent_map)
                report["ranges"].append(entry)

            fragments = self.repo.active_fragments(generation_id)
            verify_fragment_invariants(
                fragments, self.chain.limits.lower, self.chain.limits.upper)
            self._verify_coverage(gen, fragments)
            self.repo.mark_committed(generation_id)

        self.repo.set_meta("head_generation", generation_id)
        return report

    def _apply_inherit(self, gen, item: KeyRange, parent_map):
        parent_fragments = parent_map.get(item.range_id)
        if not parent_fragments:
            raise InheritError(
                "inherit rule references a range absent from the parent",
                generation_id=gen.generation_id, range_id=item.range_id)
        parent_gen_id = gen.parent_id
        parent_keys: List[str] = []
        for frag in parent_fragments:
            parent_keys.extend(frag.keys)
        if tuple(parent_keys) != item.keys:
            raise InheritError(
                "inherited range keys differ from parent; byte-identical reuse "
                "requires identical ordered keys",
                generation_id=gen.generation_id, range_id=item.range_id)

        affected = []
        for ordinal, source in enumerate(parent_fragments):
            inherited = Fragment(
                fragment_id=make_fragment_id(
                    gen.generation_id, item.range_id, ordinal),
                generation_id=gen.generation_id, range_id=item.range_id,
                ordinal=ordinal, lower=source.lower, upper=source.upper,
                keys=source.keys, key_count=source.key_count,
                digest=source.digest, version=source.version,
                inherited_from=parent_gen_id,
                source_fragment_id=source.fragment_id)
            self.repo.insert_fragment(inherited)
            self.repo.add_event(
                self._next_seq(), gen.generation_id, item.range_id,
                inherited.fragment_id, "inherit",
                {"reused_bytes": canonical_bytes(source.keys).decode("utf-8"),
                 "source_fragment_id": source.fragment_id,
                 "source_generation_id": parent_gen_id,
                 "digest": source.digest})
            affected.append(inherited.fragment_id)

        evidence = {
            "byte_identical": True,
            "digest": parent_fragments[0].digest if len(parent_fragments) == 1
            else fragment_digest(tuple(parent_keys)),
            "source_generation_id": parent_gen_id,
            "source_fragments": [f.fragment_id for f in parent_fragments],
        }
        self.repo.add_migration_entry(
            self._next_seq(), gen.generation_id, item.range_id,
            "inherit", evidence, affected)
        return {"range_id": item.range_id, "disposition": "inherit",
                "evidence": evidence, "affected_fragments": affected}

    def _apply_rewrite(self, gen, item: KeyRange, parent_map):
        old = parent_map.get(item.range_id, [])
        fragments = pack_range(
            gen.generation_id, item.range_id, item.lower, item.upper,
            item.keys, self.chain.limits.lower, self.chain.limits.upper)
        for frag in fragments:
            self.repo.insert_fragment(frag)
            self.repo.add_event(
                self._next_seq(), gen.generation_id, item.range_id,
                frag.fragment_id, "rewrite",
                {"version": frag.version, "split_sizes": [frag.key_count]})
        affected = [f.fragment_id for f in fragments]
        evidence = {
            "regenerated": True,
            "old_versions_dereferenced": [
                {"fragment_id": f.fragment_id, "version": f.version}
                for f in old],
            "new_digests": [f.digest for f in fragments],
        }
        self.repo.add_migration_entry(
            self._next_seq(), gen.generation_id, item.range_id,
            "rewrite", evidence, affected)
        return {"range_id": item.range_id, "disposition": "rewrite",
                "evidence": evidence, "affected_fragments": affected}

    def _apply_new(self, gen, item: KeyRange):
        fragments = pack_range(
            gen.generation_id, item.range_id, item.lower, item.upper,
            item.keys, self.chain.limits.lower, self.chain.limits.upper)
        for frag in fragments:
            self.repo.insert_fragment(frag)
            self.repo.add_event(
                self._next_seq(), gen.generation_id, item.range_id,
                frag.fragment_id, "new", {"version": frag.version})
        affected = [f.fragment_id for f in fragments]
        evidence = {"newly_generated": True,
                    "digests": [f.digest for f in fragments]}
        self.repo.add_migration_entry(
            self._next_seq(), gen.generation_id, item.range_id,
            "new", evidence, affected)
        return {"range_id": item.range_id, "disposition": "new",
                "evidence": evidence, "affected_fragments": affected}

    def _apply_discard(self, gen, item: KeyRange, parent_map):
        old = parent_map.get(item.range_id, [])
        affected = [f.fragment_id for f in old]
        evidence = {
            "marked_pending_cleanup": True,
            "ancestor_chain": self.chain.ancestors(gen.generation_id),
            "candidate_fragments": affected,
        }
        self.repo.add_event(
            self._next_seq(), gen.generation_id, item.range_id,
            old[0].fragment_id if old else f"frag:*:{item.range_id}:*",
            "discard", evidence)
        self.repo.add_migration_entry(
            self._next_seq(), gen.generation_id, item.range_id,
            "discard", evidence, affected)
        return {"range_id": item.range_id, "disposition": "discard",
                "evidence": evidence, "affected_fragments": affected}

    def _verify_coverage(self, gen: Generation, fragments: List[Fragment]):
        active = gen.active_ranges
        by_range: Dict[str, List[Fragment]] = {}
        for frag in fragments:
            by_range.setdefault(frag.range_id, []).append(frag)
        if set(by_range) != {item.range_id for item in active}:
            missing = {item.range_id for item in active} - set(by_range)
            raise CleanError("fragment set does not cover active key ranges",
                             generation_id=gen.generation_id,
                             range_id=next(iter(missing), None))
        for item in active:
            items = sorted(by_range[item.range_id], key=lambda f: f.ordinal)
            if items[0].lower != item.lower or items[-1].upper != item.upper:
                raise CleanError(
                    "fragment bounds do not cover the full range",
                    generation_id=gen.generation_id, range_id=item.range_id)
            keys = tuple(key for frag in items for key in frag.keys)
            if keys != item.keys:
                raise CleanError(
                    "fragment keys do not match declared ordered range keys",
                    generation_id=gen.generation_id, range_id=item.range_id)

    def _migration_summary(self, generation_id):
        return {"generation_id": generation_id, "already_committed": True,
                "ranges": [
                    {"range_id": row["range_id"],
                     "disposition": row["disposition"],
                     "evidence": row["evidence"],
                     "affected_fragments": row["affected_fragments"]}
                    for row in self.repo.migration_ledger(generation_id)]}

    # ------------------------------------------------------------ read refs
    def open_read_view(self, view_id: str, generation_id: str):
        if not self.repo.is_committed(generation_id):
            raise InheritError("cannot open read view on uncommitted generation",
                               generation_id=generation_id)
        with self.repo.transaction():
            for ancestor in self.chain.ancestors(generation_id):
                self.repo.open_ref(
                    f"{view_id}::{ancestor}", ancestor, self._seq)
        return ReadView(self, view_id, generation_id)

    def close_read_view(self, view_id: str, generation_id: str):
        with self.repo.transaction():
            for ancestor in self.chain.ancestors(generation_id):
                self.repo.close_ref(f"{view_id}::{ancestor}")

    def active_read_views(self):
        grouped: Dict[str, List[str]] = {}
        for row in self.repo.active_refs():
            view_id = row["view_id"].split("::", 1)[0]
            grouped.setdefault(view_id, []).append(row["generation_id"])
        return grouped

    # ------------------------------------------------------------ cleanup
    def _range_owner(self, range_id):
        for gen in self.chain.generations:
            item = gen.range(range_id)
            if item and item.is_active:
                return gen.generation_id
        return None

    def run_cleanup(self) -> List[dict]:
        """Physically clean discarded fragments once no ancestor view remains.

        Stable order: discard ledger entries ordered by migration id; each entry
        is idempotent via the UNIQUE cleanup key and a status recheck.
        """
        results: List[dict] = []
        discards = [row for row in self.repo.migration_ledger()
                    if row["disposition"] == "discard"]
        for row in discards:
            gen_id, range_id = row["generation_id"], row["range_id"]
            owner = self._range_owner(range_id)
            protected = set(self.chain.ancestors(gen_id))
            active = {ref["generation_id"] for ref in self.repo.active_refs()}
            blockers = sorted(protected & active)
            if blockers:
                results.append({"generation_id": gen_id, "range_id": range_id,
                                "action": "deferred", "blockers": blockers})
                continue

            cleaned_any = False
            candidate_ids = row["affected_fragments"]
            for fragment_id in candidate_ids:
                versions = self.repo.fragment_versions(fragment_id)
                for version_row in versions:
                    if version_row.status != "active":
                        continue
                    if self._fragment_referenced_elsewhere(version_row, gen_id):
                        raise CleanError(
                            "active read view still references a discard candidate",
                            generation_id=version_row.generation_id,
                            fragment_id=fragment_id)
                    evidence = {
                        "discarded_by": gen_id, "range_id": range_id,
                        "protected_ancestors": sorted(protected),
                        "active_refs_at_cleanup": sorted(active),
                        "replaced_by_owner": owner,
                        "digest": version_row.digest}
                    with self.repo.transaction():
                        self.repo.mark_cleaned(fragment_id, version_row.version)
                        self.repo.add_cleanup_entry(
                            self._next_seq(), version_row.generation_id,
                            range_id, fragment_id,
                            "no ancestor read view remains; range discarded",
                            evidence)
                        self.repo.add_event(
                            self._next_seq(), version_row.generation_id,
                            range_id, fragment_id, "clean", evidence)
                    cleaned_any = True
                    results.append({"generation_id": version_row.generation_id,
                                    "range_id": range_id,
                                    "fragment_id": fragment_id,
                                    "action": "cleaned", "evidence": evidence})
            if not cleaned_any:
                already = [c for c in self.repo.cleanup_ledger()
                           if c["range_id"] == range_id]
                results.append({"generation_id": gen_id, "range_id": range_id,
                                "action": "already_cleaned",
                                "ledger_entries": len(already)})
        return results

    def _fragment_referenced_elsewhere(self, fragment, discarding_gen):
        active = {ref["generation_id"] for ref in self.repo.active_refs()}
        return fragment.generation_id in active

    # ------------------------------------------------------------ snapshot
    def build_manifest(self) -> dict:
        fragments = [frag.to_manifest()
                     for frag in self.repo.all_active_fragments()]
        inheritance = {}
        for frag in self.repo.all_active_fragments():
            if frag.inherited_from:
                inheritance[frag.fragment_id] = {
                    "from_generation": frag.inherited_from,
                    "source_fragment_id": frag.source_fragment_id,
                    "digest": frag.digest}
        manifest = {
            "format_version": SNAPSHOT_FORMAT_VERSION,
            "schema_version": SCHEMA_VERSION,
            "title": self.chain.title,
            "limits": {"lower": self.chain.limits.lower,
                       "upper": self.chain.limits.upper},
            "capacity_params_version": self._capacity_params_version(),
            "chain": self._chain_declaration_digest(),
            "generations": [
                {"generation_id": g.generation_id, "seq": g.seq,
                 "parent_id": g.parent_id,
                 "digest": self._generation_digest(g)}
                for g in self.chain.generations],
            "generation_pointer": self.repo.get_meta("head_generation"),
            "fragments": fragments,
            "inheritance": inheritance,
            "cleanup_ledger": self.repo.cleanup_ledger(),
        }
        manifest["content_digest"] = self._manifest_digest(manifest)
        return manifest

    def _capacity_params_version(self):
        limits = self.chain.limits
        return hashlib.sha256(
            f"{limits.lower}:{limits.upper}".encode("utf-8")).hexdigest()[:12]

    def _chain_declaration_digest(self):
        payload = []
        for gen in self.chain.generations:
            payload.append(self._generation_digest(gen))
        payload.append([self.chain.limits.lower, self.chain.limits.upper])
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def _manifest_digest(self, manifest):
        copy = {key: value for key, value in manifest.items()}
        copy.pop("content_digest", None)
        canonical = json.dumps(copy, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def save_snapshot(self, path):
        manifest = self.build_manifest()
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2,
                      sort_keys=True)
        self.repo.set_meta("snapshot_content_digest", manifest["content_digest"])
        self.repo.set_meta("chain_declaration_digest",
                           self._chain_declaration_digest())
        self.repo.set_meta("capacity_params_version",
                           self._capacity_params_version())
        return manifest

    def snapshot_diff(self, previous_manifest: Optional[dict]) -> dict:
        """Classify fragments vs a prior snapshot manifest (stable ordering)."""
        current = self.build_manifest()
        old = (previous_manifest or {}).get("fragments", [])
        added, reused, rewritten, ownership_changed = [], [], [], []
        old_digests = {item["digest"] for item in old}

        # Primary identity is the (range, ordinal) slot. Digest distinguishes
        # reuse from rewrite; generation id is excluded from identity because
        # byte-identical inherited fragments legitimately change ownership.
        old_by_slot = {}
        for item in old:
            old_by_slot.setdefault((item["range_id"], item["ordinal"]), []).append(item)
        new_slots = {(item["range_id"], item["ordinal"]): item
                     for item in current["fragments"]}
        for slot, item in sorted(new_slots.items()):
            candidates = old_by_slot.get(slot, [])
            exact = next((candidate for candidate in candidates
                          if candidate["digest"] == item["digest"]), None)
            different = next((candidate for candidate in candidates
                              if candidate["digest"] != item["digest"]), None)
            if exact is not None:
                if exact["generation_id"] != item["generation_id"] or \
                        exact["inherited_from"] != item["inherited_from"]:
                    ownership_changed.append(
                        {"fragment_id": item["fragment_id"],
                         "old_generation": exact["generation_id"],
                         "new_generation": item["generation_id"],
                         "inherited_from": item["inherited_from"]})
                reused.append(item["fragment_id"])
            elif different is not None:
                rewritten.append({"fragment_id": item["fragment_id"],
                                  "old_digest": different["digest"],
                                  "new_digest": item["digest"]})
            elif item["digest"] in old_digests:
                ownership_changed.append(
                    {"fragment_id": item["fragment_id"],
                     "old_generation": "(matched by digest)",
                     "new_generation": item["generation_id"],
                     "inherited_from": item["inherited_from"]})
                reused.append(item["fragment_id"])
            else:
                added.append(item["fragment_id"])
        old_slots = set(old_by_slot)
        return {
            "added": added,
            "inherited_reused": reused,
            "rewritten": rewritten,
            "ownership_changed": ownership_changed,
            "removed": sorted(
                candidates[0]["fragment_id"]
                for slot, candidates in old_by_slot.items()
                if slot not in new_slots),
            "content_digest": current["content_digest"],
        }

    def replace_from_chain(self, previous_manifest: Optional[dict]) -> dict:
        """Whole-snapshot replacement; never an incremental merge."""
        with self.repo.transaction():
            self.repo.replace_all()
            for gen in self.chain.generations:
                self.repo.upsert_generation(
                    gen.generation_id, gen.seq, gen.parent_id,
                    self._generation_digest(gen), committed=False)
        for gen in self.chain.generations:
            self.migrate_generation(gen.generation_id)
        for entry in (previous_manifest or {}).get("cleanup_ledger", []):
            versions = self.repo.fragment_versions(entry["fragment_id"])
            for version_row in versions:
                if version_row.status == "active":
                    self.repo.mark_cleaned(entry["fragment_id"],
                                           version_row.version)
                    self.repo.add_cleanup_entry(
                        self._next_seq(), entry["generation_id"],
                        entry["range_id"], entry["fragment_id"],
                        entry["reason"], entry["evidence"])
        diff = self.snapshot_diff(previous_manifest)
        with self.repo.transaction():
            self.repo.set_meta("snapshot_content_digest",
                               diff["content_digest"])
            self.repo.set_meta("chain_declaration_digest",
                               self._chain_declaration_digest())
            self.repo.set_meta("capacity_params_version",
                               self._capacity_params_version())
            self.repo.set_meta("head_generation",
                               self.chain.generations[-1].generation_id)
        return diff

    def close(self):
        self.repo.close()
