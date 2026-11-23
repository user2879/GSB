"""Reverse reference index: which tickets cite each rule / role / type."""


def build_references(tickets, rules):
    refs = {"type": {}, "role": {}, "relation": {}}
    for ticket in tickets:
        tid = ticket.tid
        for ev in ticket.evidences:
            refs["type"].setdefault(ev.etype, set()).add(tid)
            refs["role"].setdefault(ev.role, set()).add(tid)
        referenced_types = {ev.etype for ev in ticket.evidences}
        for entry in rules.all_relations():
            if entry.left in referenced_types and \
                    entry.right in referenced_types:
                refs["relation"].setdefault(entry.rid, set()).add(tid)

    def materialize(mapping):
        return {key: sorted(values) for key, values in mapping.items()}

    return {
        "type": materialize(refs["type"]),
        "role": materialize(refs["role"]),
        "relation": materialize(refs["relation"]),
    }


def affected_for_type(refs, etype):
    return set(refs.get("type", {}).get(etype, []))


def affected_for_role(refs, role):
    return set(refs.get("role", {}).get(role, []))


def affected_for_relation(refs, rid):
    return set(refs.get("relation", {}).get(rid, []))
