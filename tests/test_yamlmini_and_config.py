from conftest import ROOT
from qcalib import yamlmini
from qcalib.loaders import load_config


def test_yamlmini_roundtrip_and_types():
    text = (
        "name: demo\n"
        "workers: 4\n"
        "ratio: 0.25\n"
        "enabled: true\n"
        "items:\n"
        "  - 1\n"
        "  - 2\n"
        "nested:\n"
        "  key: value\n"
        "# a comment line\n"
        "inline: [3, 4]\n"
    )
    data = yamlmini.loads(text)
    assert data == {
        "name": "demo", "workers": 4, "ratio": 0.25, "enabled": True,
        "items": [1, 2], "nested": {"key": "value"}, "inline": [3, 4]}
    reloaded = yamlmini.loads(yamlmini.dumps(data))
    assert reloaded == data


def test_pipeline_config_resolution():
    config = load_config(ROOT / "configs" / "pipeline.yaml")
    embedding = config.resolve_layer("embedding_proj")
    assert embedding.level == "int16" and embedding.scheme == "sym"
    fallback = config.resolve_layer("unknown_layer")
    assert fallback.level == "int8" and fallback.group_size == 4
    assert config.tier_for("embedding_proj") == "small"
    assert config.tier_for("unknown_layer") == config.default_tier
