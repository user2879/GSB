"""Edit command application for incremental sessions.

Commands (JSON objects):
  {"op": "replace_row",     "word_id": int, "vector": [float, ...]}
  {"op": "replace_block",   "block_index": int,
   "rows": [{"word_id": int, "vector": [...]}, ...]}
  {"op": "append_block",    "rows": [{"word_id": int, "vector": [...]}, ...]}
  {"op": "set_quant_level", "quant_level": int}
  {"op": "set_agg_mode",    "agg_mode": "sum"|"mean"}

Row vector dimension may never change; structural fields (dim,
rows_per_block, frac_bits) are immutable across a session.
"""

from .fixedpoint import is_power_of_two


def apply_edit_command(table, config, cmd):
    """Apply one command; return an impact descriptor.

    Impact fields:
      affected_word_ids : row ids whose content changed (gather
                          invalidation is driven by this set)
      added_word_ids    : ids that did not exist before (append)
      invalidate_all    : True when every gather result must be rebuilt
      table_changed     : True when validate/fingerprint artifacts
                          must be recommitted
    """
    op = cmd.get("op")
    impact = {
        "op": op,
        "affected_word_ids": set(),
        "added_word_ids": set(),
        "invalidate_all": False,
        "table_changed": False,
    }
    if op == "replace_row":
        impact["affected_word_ids"] = table.replace_row(
            cmd["word_id"], cmd["vector"])
        impact["table_changed"] = True
    elif op == "replace_block":
        impact["affected_word_ids"] = table.replace_block(
            cmd["block_index"], cmd["rows"])
        impact["table_changed"] = True
    elif op == "append_block":
        block_index, added = table.append_block(cmd["rows"])
        impact["added_word_ids"] = added
        impact["block_index"] = block_index
        impact["table_changed"] = True
    elif op == "set_quant_level":
        level = int(cmd["quant_level"])
        if not is_power_of_two(level):
            raise ValueError("quant_level must be a power of two >= 2")
        config.quant_level = level
        # Parameter switch: validate/fingerprint stages are NOT
        # invalidated; row fingerprints are recomputed from cached
        # vectors at the new level and only the gather stage reruns.
        table.requantize(config.quant_level, config.frac_bits)
        impact["invalidate_all"] = True
    elif op == "set_agg_mode":
        mode = cmd["agg_mode"]
        if mode not in ("sum", "mean"):
            raise ValueError("agg_mode must be 'sum' or 'mean'")
        config.agg_mode = mode
        impact["invalidate_all"] = True
    else:
        raise ValueError(f"unknown edit op: {op}")
    return impact
