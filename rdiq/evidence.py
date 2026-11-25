"""Evidence retrieval, topic-scoped dedup and fixed-point arbitration.

The arbitration is a pure deterministic function. All collections are
normalized through explicit ordering before serialization so fragment load
order and dictionary iteration order cannot affect the output bytes.
"""

from .canonical import canonical_text, digest
from .errors import GraphError
from .fixedpoint import (
    WEIGHT_BASE,
    clamp,
    div_trunc,
    mul_wide,
    quantize_weight,
    sat_add,
)


def assign_topic(rule, query_terms):
    """Assign one round to the first rule topic containing an anchor hit.

    Rules are inspected in sorted topic-label order. Ties on anchor-hit count
    are therefore resolved by topic label, never by dictionary iteration.
    """
    query = set(query_terms)
    ranked = []
    for topic in sorted(rule["topics"]):
        anchors = rule["topics"][topic]["anchors"]
        hits = sum(1 for anchor in anchors if anchor in query)
        ranked.append((hits, topic))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    best_hits, topic = ranked[0]
    if best_hits == 0:
        return None, ranked
    return topic, ranked


def evidence_content_digest(fragment):
    return digest(
        {
            "topic": fragment["topic"],
            "channel": fragment["channel"],
            "terms": list(fragment["terms"]),
        }
    )


def _candidate_sort_key(item):
    fragment, score = item
    return (
        -score,
        fragment["channel"],
        fragment["fragment_id"],
    )


def _stable_chain_key(chain):
    return (
        -chain["quantized_weight"],
        chain["channel"],
        chain["fragment_id"],
    )


def arbitrate_round(dataset, session_id, request):
    config = dataset.config
    width = config.precision["width"]
    scale = config.precision["scale"]
    query_terms = list(request["query_terms"])
    topic, topic_scores = assign_topic(config.rule, query_terms)
    base = {
        "session_id": session_id,
        "round": request["round"],
        "topic": topic,
        "query_terms": sorted(query_terms),
        "scale": scale,
        "width": width,
        "topic_scores": [
            {"topic": label, "anchor_hits": hits} for hits, label in topic_scores
        ],
        "adopted": [],
        "deduped": [],
        "weight_steps": [],
        "count": 0,
        "weighted_sum": 0,
        "sum_saturated": False,
        "mean_residual": 0,
        "confidence": 0,
        "confidence_saturated": False,
        "evidence_digests": [],
        "tie": None,
    }
    if topic is None:
        base["round_digest"] = digest(base)
        return base

    query = set(query_terms)
    candidates = []
    for fragment in sorted(dataset.fragments, key=lambda item: item["fragment_id"]):
        if fragment["topic"] != topic or fragment["channel"] not in config.channels:
            continue
        score = sum(1 for term in fragment["terms"] if term in query)
        if score > 0:
            candidates.append((fragment, score))
    candidates.sort(key=_candidate_sort_key)

    groups = {}
    for fragment, score in candidates:
        content_digest = evidence_content_digest(fragment)
        groups.setdefault(content_digest, []).append((fragment, score))

    adopted = []
    deduped = []
    weight_steps = []
    weighted_sum = 0
    sum_saturated = False

    for content_digest in sorted(groups):
        members = sorted(
            groups[content_digest],
            key=lambda item: (
                -item[1],
                item[0]["channel"],
                item[0]["fragment_id"],
            ),
        )
        representative, relevance = members[0]
        channel_raw = config.weights["channel_weights"][representative["channel"]]
        topic_raw = config.weights["topic_weights"][topic]
        channel_quant = quantize_weight(channel_raw, scale, width)
        topic_quant = quantize_weight(topic_raw, scale, width)
        rel_quant, rel_residual = div_trunc(relevance * scale, WEIGHT_BASE)
        rel_quant, rel_saturated = clamp(rel_quant, width)
        combined, combined_sat_a = sat_add(
            channel_quant["quantized"], topic_quant["quantized"], width
        )
        combined, combined_sat_b = sat_add(combined, rel_quant, width)
        combined, weight_saturated = clamp(combined, width)
        product, product_saturated = mul_wide(combined, relevance, width)
        weighted_sum, add_saturated = sat_add(weighted_sum, product, width)
        sum_saturated = any(
            (
                sum_saturated,
                product_saturated,
                add_saturated,
                combined_sat_a,
                combined_sat_b,
                rel_saturated,
                weight_saturated,
            )
        )
        chain = {
            "fragment_id": representative["fragment_id"],
            "topic": topic,
            "channel": representative["channel"],
            "relevance_hits": relevance,
            "channel_raw": channel_raw,
            "topic_raw": topic_raw,
            "channel_quantized": channel_quant["quantized"],
            "topic_quantized": topic_quant["quantized"],
            "relevance_quantized": rel_quant,
            "quantized_weight": combined,
            "weighted_contribution": product,
            "content_digest": content_digest,
            "weight_saturated": weight_saturated,
        }
        adopted.append(chain)
        weight_steps.append(
            {
                "fragment_id": representative["fragment_id"],
                "basis": (
                    f"truncate((channel_raw*{scale})/{WEIGHT_BASE}) + "
                    f"truncate((topic_raw*{scale})/{WEIGHT_BASE}) + "
                    f"truncate((relevance_hits*{scale})/{WEIGHT_BASE}); "
                    "then product = quantized_weight * relevance_hits"
                ),
                "channel_residual": channel_quant["residual"],
                "topic_residual": topic_quant["residual"],
                "relevance_residual": abs(rel_residual),
                "quantized_weight": combined,
                "weighted_contribution": product,
            }
        )
        for duplicate, duplicate_score in members[1:]:
            deduped.append(
                {
                    "fragment_id": duplicate["fragment_id"],
                    "channel": duplicate["channel"],
                    "relevance_hits": duplicate_score,
                    "content_digest": content_digest,
                    "kept_representative": representative["fragment_id"],
                    "reason": "identical topic/channel/term content digest within topic",
                }
            )

    adopted.sort(key=_stable_chain_key)
    deduped.sort(
        key=lambda item: (
            item["content_digest"],
            item["channel"],
            item["fragment_id"],
        )
    )
    weight_steps.sort(key=lambda item: item["fragment_id"])
    count = len(adopted)
    if count:
        mean, mean_residual = div_trunc(weighted_sum, count)
        confidence, confidence_saturated = clamp(mean, width)
    else:
        confidence, mean_residual = 0, 0

    tie = None
    if len(adopted) >= 2:
        primary, secondary = adopted[0], adopted[1]
        tie = {
            "primary_fragment_id": primary["fragment_id"],
            "secondary_fragment_id": secondary["fragment_id"],
            "quantized_delta": abs(
                primary["quantized_weight"] - secondary["quantized_weight"]
            ),
            "resolved_by": "higher quantized weight, then channel, then fragment_id",
            "within_error_bound": None,
            "error_affected": False,
        }

    base.update(
        {
            "adopted": adopted,
            "deduped": deduped,
            "weight_steps": weight_steps,
            "count": count,
            "weighted_sum": weighted_sum,
            "sum_saturated": sum_saturated,
            "mean_residual": abs(mean_residual),
            "confidence": confidence,
            "confidence_saturated": confidence_saturated,
            "evidence_digests": sorted(item["content_digest"] for item in adopted),
            "tie": tie,
        }
    )
    base["round_digest"] = digest(base)
    return base


def build_reference_graph(dataset, round_records):
    """Build source -> evidence -> round -> session nodes and verify acyclicity.

    Edges always point from source data toward the session view. A white/gray/
    black DFS over sorted node identifiers rejects a missing reference or a
    cycle before any confidence is aggregated.
    """
    fragment_ids = {fragment["fragment_id"] for fragment in dataset.fragments}
    nodes = set()
    edges = set()
    for session in sorted(dataset.sessions, key=lambda item: item["session_id"]):
        session_id = session["session_id"]
        session_node = f"session:{session_id}"
        nodes.add(session_node)
        for record in sorted(
            round_records[session_id].values(),
            key=lambda item: item["round"],
        ):
            round_node = f"round:{session_id}:{record['round']}"
            nodes.add(round_node)
            edges.add((round_node, session_node))
            for chain in record["adopted"]:
                fragment_id = chain["fragment_id"]
                if fragment_id not in fragment_ids:
                    raise GraphError(
                        "round references a missing source fragment",
                        entry=session_id,
                        field=["round", record["round"], "fragment_id"],
                        fragment_id=fragment_id,
                    )
                evidence_node = (
                    f"evidence:{session_id}:{record['round']}:{fragment_id}"
                )
                source_node = f"source:{fragment_id}"
                nodes.add(evidence_node)
                nodes.add(source_node)
                edges.add((evidence_node, round_node))
                edges.add((source_node, evidence_node))

    adjacency = {}
    for source, target in edges:
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set())

    color = {node: "white" for node in nodes}

    def visit(node, stack):
        color[node] = "gray"
        for neighbor in sorted(adjacency[node]):
            if color[neighbor] == "gray":
                raise GraphError(
                    "reference graph contains a cycle",
                    entry=node,
                    field="reference_chain",
                    cycle=stack + [node, neighbor],
                )
            if color[neighbor] == "white":
                visit(neighbor, stack + [node])
        color[node] = "black"

    for node in sorted(nodes):
        if color[node] == "white":
            visit(node, [])

    canonical_edges = [
        {"from": source, "to": target}
        for source, target in sorted(edges)
    ]
    graph = {
        "nodes": sorted(nodes),
        "edges": canonical_edges,
        "edge_count": len(edges),
        "node_count": len(nodes),
    }
    graph["graph_digest"] = digest(graph)
    return graph


def synthesize_session_view(config, session_id, records, graph_digest):
    width = config.precision["width"]
    ordered = sorted(records, key=lambda item: item["round"])
    round_confidences = [item["confidence"] for item in ordered]
    if round_confidences:
        total = 0
        saturated = False
        for value in round_confidences:
            total, step_saturated = sat_add(total, value, width)
            saturated = saturated or step_saturated
        view_confidence, residual = div_trunc(total, len(round_confidences))
        view_confidence, narrow_saturated = clamp(view_confidence, width)
        saturated = saturated or narrow_saturated
    else:
        view_confidence, residual, saturated = 0, 0, False

    chain = []
    for record in ordered:
        chain.append(
            {
                "round": record["round"],
                "topic": record["topic"],
                "confidence": record["confidence"],
                "round_digest": record["round_digest"],
                "evidence_digests": list(record["evidence_digests"]),
            }
        )
    view = {
        "session_id": session_id,
        "round_count": len(ordered),
        "round_confidences": round_confidences,
        "view_confidence": view_confidence,
        "view_residual": abs(residual),
        "view_saturated": saturated,
        "reference_chain": chain,
        "graph_digest": graph_digest,
    }
    view["view_digest"] = digest(view)
    view["canonical"] = canonical_text(view)
    return view
