import pytest

from decchain.errors import BizError, BizCode
from decchain.exchange import (
    ExchangeDocument, ExchangeRecord, V1_MAGIC, V2_MAGIC,
    parse, render, documents_equal, write_sidecar, load_sidecar,
)


def _doc():
    records = [
        ExchangeRecord(request_no=1, request_id="R001", seq_id=1,
                       step_index=1, token=100, mille_prob=458,
                       token_name="tok100", quant_residual=2,
                       ref_int=4580, covered_ref="residual:2",
                       precision_label="W4"),
        ExchangeRecord(request_no=1, request_id="R001", seq_id=1,
                       step_index=1, token=101, mille_prob=307,
                       token_name="tok101", quant_residual=0,
                       ref_int=3070, precision_label="W4"),
    ]
    return ExchangeDocument(
        version=V2_MAGIC, width=4, tier="ROUND_HALF_UP",
        config_fingerprint="c" * 64,
        request_index={"R001": 1}, token_names={100: "tok100",
                                                 101: "tok101"},
        records=records)


def test_v1_v2_round_trip_equivalence(tmp_path):
    doc = _doc()
    v2_text = render(doc, "v2")
    v1_text = render(doc, "v1")
    assert v1_text.startswith("HDR " + V1_MAGIC)

    sidecar = tmp_path / "sidecar.json"
    write_sidecar(doc, str(sidecar))
    v1_doc = load_sidecar(parse(v1_text, "v1"), str(sidecar))
    v2_back = parse(render(v1_doc, "v2"), "v2")
    assert documents_equal(v2_back, parse(v2_text, "v2"))


def test_v1_short_line_raises_business_code():
    doc = _doc()
    text = render(doc, "v1")
    broken = "\n".join(line[:-1] if line.startswith("DAT ") else line
                       for line in text.splitlines())
    with pytest.raises(BizError) as exc:
        parse(broken, "v1")
    assert exc.value.code == BizCode.V1_LINE_TOO_SHORT


def test_v1_illegal_character_raises_business_code():
    doc = _doc()
    text = render(doc, "v1")
    broken = text.replace("DAT ", "DAT\t", 1)
    with pytest.raises(BizError) as exc:
        parse(broken, "v1")
    assert exc.value.code == BizCode.V1_ILLEGAL_CHAR


def test_v2_bad_json_and_probability():
    with pytest.raises(BizError) as exc:
        parse("{not json", "v2")
    assert exc.value.code == BizCode.V2_BAD_JSON

    doc = _doc()
    payload = render(doc, "v2")
    import json
    data = json.loads(payload)
    data["records"][0]["probability"] = 1.5
    with pytest.raises(BizError) as exc:
        parse(json.dumps(data), "v2")
    assert exc.value.code == BizCode.V2_BAD_PROBABILITY


def test_sidecar_preserves_residual_and_reference(tmp_path):
    doc = _doc()
    sidecar = tmp_path / "sc.json"
    write_sidecar(doc, str(sidecar))
    restored = load_sidecar(parse(render(doc, "v1"), "v1"), str(sidecar))
    rec = next(r for r in restored.records if r.token == 100)
    assert rec.quant_residual == 2
    assert rec.ref_int == 4580
    assert rec.covered_ref == "residual:2"
