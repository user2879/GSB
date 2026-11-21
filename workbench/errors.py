"""Business error codes and failure-list entries."""

E_SAMPLE_INVALID = "E_SAMPLE_INVALID"
E_LABEL_ENUM = "E_LABEL_ENUM"
E_BOX_OUT_OF_RANGE = "E_BOX_OUT_OF_RANGE"
E_ORDER_GAP = "E_ORDER_GAP"
E_ARBITRATION_DEADLOCK = "E_ARBITRATION_DEADLOCK"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"


def failure(sample_id, source, code, detail):
    return {
        "sample_id": sample_id,
        "source": source,
        "code": code,
        "detail": str(detail),
    }


def failure_sort_key(item):
    return (item["sample_id"], item["source"], item["code"], item["detail"])
