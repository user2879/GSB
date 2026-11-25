"""Reference-graph change propagation and subset invalidation (requirement 5).

A configuration change is first mapped to concrete source/fragment nodes. The
reference graph is then followed in its edge direction (source -> evidence ->
round -> session). Output ordering is by session identifier and round index,
never by set or dictionary iteration.
"""

from .evidence import assign_topic


def _fragment_node(fragment_id):
    return f"source:{fragment_id}"


def _affected_fragments(old_dataset, new_dataset, changed):
    old_config, new_config = old_dataset.config, new_dataset.config
    affected = set()
    reasons = []

    if changed["rule_version"]:
        affected.update(
            fragment["fragment_id"] for fragment in new_dataset.fragments
        )
        reasons.append(
            {
                "step": "rule-change",
                "basis": "topic anchors changed, so every fragment-side topic mapping is suspect",
                "fragment_ids": sorted(
                    fragment["fragment_id"] for fragment in new_dataset.fragments
                ),
            }
        )
    if changed["precision_version"]:
        affected.update(
            fragment["fragment_id"] for fragment in new_dataset.fragments
        )
        reasons.append(
            {
                "step": "scale-change",
                "basis": "fixed-point scale/width changed, so every quantized chain is invalid",
                "fragment_ids": sorted(
                    fragment["fragment_id"] for fragment in new_dataset.fragments
                ),
            }
        )
    if changed["weight_version"]:
        old_channel = old_dataset.config.weights["channel_weights"]
        new_channel = new_dataset.config.weights["channel_weights"]
        old_topic = old_dataset.config.weights["topic_weights"]
        new_topic = new_dataset.config.weights["topic_weights"]
        changed_channels = {
            channel
            for channel in sorted(set(old_channel) | set(new_channel))
            if old_channel.get(channel) != new_channel.get(channel)
        }
        changed_topics = {
            topic
            for topic in sorted(set(old_topic) | set(new_topic))
            if old_topic.get(topic) != new_topic.get(topic)
        }
        weight_affected = {
            fragment["fragment_id"]
            for fragment in new_dataset.fragments
            if fragment["channel"] in changed_channels
            or fragment["topic"] in changed_topics
        }
        affected.update(weight_affected)
        reasons.append(
            {
                "step": "weight-change",
                "basis": "only fragments on changed channels or changed topics are invalid",
                "changed_channels": sorted(changed_channels),
                "changed_topics": sorted(changed_topics),
                "fragment_ids": sorted(weight_affected),
            }
        )

    old_channels = old_config.channels
    new_channels = new_config.channels
    removed_channels = sorted(old_channels - new_channels)
    added_channels = sorted(new_channels - old_channels)
    if removed_channels or added_channels:
        channel_affected = set()
        for fragment in new_dataset.fragments:
            if (
                fragment["channel"] in removed_channels
                or fragment["channel"] in added_channels
            ):
                channel_affected.add(fragment["fragment_id"])
        affected.update(channel_affected)
        reasons.append(
            {
                "step": "channel-change",
                "basis": "recall channel set changed",
                "removed_channels": removed_channels,
                "added_channels": added_channels,
                "fragment_ids": sorted(channel_affected),
            }
        )
    return affected, reasons


def diff_versions(old_dataset, new_dataset):
    old, new = old_dataset.config, new_dataset.config
    return {
        "rule_version": old.rule["version"] != new.rule["version"],
        "weight_version": old.weights["version"] != new.weights["version"],
        "precision_version": old.precision["version"]
        != new.precision["version"],
        "content_digest": old.digest != new.digest,
        "old": dict(old.versions),
        "new": dict(new.versions),
        "old_content_digest": old.digest,
        "new_content_digest": new.digest,
    }


def propagate_invalidation(old_dataset, new_dataset, graph):
    changed = diff_versions(old_dataset, new_dataset)
    fragment_ids, reasons = _affected_fragments(old_dataset, new_dataset, changed)
    adjacency = {}
    for edge in graph["edges"]:
        adjacency.setdefault(edge["from"], set()).add(edge["to"])

    frontier = {_fragment_node(fragment_id) for fragment_id in fragment_ids}
    reached = set(frontier)
    queue = sorted(frontier)
    while queue:
        node = queue.pop(0)
        for neighbor in sorted(adjacency.get(node, ())):
            if neighbor not in reached:
                reached.add(neighbor)
                queue.append(neighbor)

    round_keys = set()
    session_ids = set()
    for node in reached:
        if node.startswith("round:"):
            _, session_id, round_index = node.split(":", 2)
            round_keys.add((session_id, int(round_index)))
            session_ids.add(session_id)
        elif node.startswith("session:"):
            session_ids.add(node.split(":", 1)[1])

    topic_rounds = set()
    for session in sorted(new_dataset.sessions, key=lambda item: item["session_id"]):
        for request in sorted(session["rounds"], key=lambda item: item["round"]):
            topic, _ = assign_topic(new_dataset.config.rule, request["query_terms"])
            if topic is None:
                continue
            key = (session["session_id"], request["round"])
            if key in round_keys:
                topic_rounds.add((topic, key[0], key[1]))

    invalid_rounds = [
        {"session_id": session_id, "round": round_index, "topic": topic}
        for topic, session_id, round_index in sorted(
            topic_rounds, key=lambda item: (item[1], item[2], item[0])
        )
    ]
    propagation = []
    for reason in reasons:
        propagation.append(reason)
    propagation.append(
        {
            "step": "graph-follow",
            "basis": "follow source -> evidence -> round -> session edges",
            "reached_nodes": sorted(reached),
        }
    )
    propagation.append(
        {
            "step": "stable-output",
            "basis": "sort invalid rounds by session_id then round index",
            "invalid_sessions": sorted(session_ids),
        }
    )
    return {
        "changed": changed,
        "invalid_fragment_ids": sorted(fragment_ids),
        "invalid_topics": sorted({item["topic"] for item in invalid_rounds}),
        "invalid_sessions": sorted(session_ids),
        "invalid_rounds": invalid_rounds,
        "propagation": propagation,
    }
