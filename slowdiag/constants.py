from __future__ import annotations


APP_LOG_SUMMARY = "app_log_summary"
DB_EXEC_STATS = "db_exec_stats"
SLOW_LOG_EXCERPT = "slow_log_excerpt"

SOURCE_TYPES = (APP_LOG_SUMMARY, DB_EXEC_STATS, SLOW_LOG_EXCERPT)

TYPE_LABELS = {
    APP_LOG_SUMMARY: "应用日志摘要",
    DB_EXEC_STATS: "库执行统计",
    SLOW_LOG_EXCERPT: "慢日志摘录",
}

TYPE_ALIASES = {
    APP_LOG_SUMMARY: APP_LOG_SUMMARY,
    "application_log_summary": APP_LOG_SUMMARY,
    "app_log": APP_LOG_SUMMARY,
    "应用日志摘要": APP_LOG_SUMMARY,
    DB_EXEC_STATS: DB_EXEC_STATS,
    "database_execution_stats": DB_EXEC_STATS,
    "db_stats": DB_EXEC_STATS,
    "库执行统计": DB_EXEC_STATS,
    SLOW_LOG_EXCERPT: SLOW_LOG_EXCERPT,
    "slow_log": SLOW_LOG_EXCERPT,
    "慢日志摘录": SLOW_LOG_EXCERPT,
}

KNOWN_VERSIONS = {
    APP_LOG_SUMMARY: (1, 2),
    DB_EXEC_STATS: (1, 2),
    SLOW_LOG_EXCERPT: (1, 2),
}

ALLOWED_ELAPSED_SEMANTICS = {
    APP_LOG_SUMMARY: ("p95", "mean"),
    DB_EXEC_STATS: ("p95", "mean", "p50"),
    SLOW_LOG_EXCERPT: ("max", "p95", "mean"),
}

ALLOWED_SCAN_SEMANTICS = {
    APP_LOG_SUMMARY: ("estimated", "sampled"),
    DB_EXEC_STATS: ("exact",),
    SLOW_LOG_EXCERPT: ("exact", "sampled"),
}

# Reported value = canonical value * factor. Decoding divides by the factor.
ELAPSED_FACTOR_BP = {
    "p95": 10000,
    "mean": 13000,
    "p50": 17500,
    "max": 9000,
}

SCAN_FACTOR_BP = {
    "exact": 10000,
    "estimated": 10500,
    "sampled": 12000,
}

# Stable arbitration: larger bias wins, then format version descending, then id ascending.
SOURCE_TYPE_BIAS = {
    APP_LOG_SUMMARY: 100,
    SLOW_LOG_EXCERPT: 200,
    DB_EXEC_STATS: 300,
}

METRICS = ("elapsed_ms", "scanned_rows")
METRIC_LABELS = {
    "elapsed_ms": "耗时代数",
    "scanned_rows": "扫描行数",
}
