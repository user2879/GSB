"""v1 flat fixed-width key-value line format.

Line layout (24 chars + '\\n'), fields are positional fixed width::

    kind(1) id(4) bw(2) scale(6) stride(2) pad(2) kernel(2) n_in(1) inputs(4)

Input nodes ('N') only fill id/bitwidth/scale; conv nodes ('C') use all
fields. Node numbering is implicit from the 4-digit integer id.
"""
from typing import List

from .errors import QCGError, E_GRAPH_INVALID
from .models import Node, Graph, NODE_INPUT, NODE_CONV

MAGIC = "QCGV1"
LINE_LEN = 24
FIELDS = ((0, 1, "kind"), (1, 5, "id"), (5, 7, "bitwidth"),
          (7, 13, "scale_permille"), (13, 15, "stride"), (15, 17, "padding"),
          (17, 19, "kernel"), (19, 20, "n_inputs"), (20, 24, "inputs"))
BW_CODE = {4: "04", 8: "08", 16: "16"}
CODE_BW = {v: k for k, v in BW_CODE.items()}


def _err(message: str, node_id: str = "-", field: str = "line") -> QCGError:
    return QCGError(E_GRAPH_INVALID, message, node_id=node_id, field=field)


def parse_v1(text: str) -> Graph:
    lines = [ln for ln in text.splitlines() if ln.strip() != ""]
    if not lines or not lines[0].startswith(MAGIC):
        raise _err("missing %s magic header" % MAGIC)
    header = lines[0].split()
    if len(header) != 2 or not header[1].isdigit():
        raise _err("malformed header; expected '%s <node_count>'" % MAGIC)
    declared = int(header[1])
    body = lines[1:]
    if declared != len(body):
        raise _err("header node count %d != %d body lines"
                   % (declared, len(body)))
    nodes: List[Node] = []
    seen = set()
    for line_no, line in enumerate(body):
        if len(line) != LINE_LEN:
            raise _err("line %d length %d != fixed width %d"
                       % (line_no + 2, len(line), LINE_LEN))
        parts = {name: line[a:b] for a, b, name in FIELDS}
        if not parts["id"].isdigit() or not parts["n_inputs"].isdigit():
            raise _err("non-digit id field on line %d" % (line_no + 2))
        nid = "n%04d" % int(parts["id"])
        if nid in seen:
            raise _err("duplicate node id", node_id=nid, field="id")
        seen.add(nid)
        kind = parts["kind"]
        if kind not in ("N", "C"):
            raise _err("kind must be N or C", node_id=nid, field="kind")
        if parts["bitwidth"] not in CODE_BW:
            raise _err("bitwidth code %r not in %s"
                       % (parts["bitwidth"], sorted(CODE_BW)),
                       node_id=nid, field="bitwidth")
        bw = CODE_BW[parts["bitwidth"]]
        scale_text = parts["scale_permille"]
        if scale_text[0] not in "+-" or not scale_text[1:].isdigit():
            raise _err("scale field must be a signed 6-digit integer",
                       node_id=nid, field="scale_permille")
        scale = int(scale_text)
        int_fields = ("stride", "padding", "kernel", "n_inputs")
        for name in int_fields:
            if not parts[name].isdigit():
                raise _err("%s must be digits" % name, node_id=nid, field=name)
        stride, pad, kernel, n_in = (int(parts[name]) for name in int_fields)
        if n_in > 4:
            raise _err("n_inputs width supports at most 4 input ids",
                       node_id=nid, field="n_inputs")
        input_text = parts["inputs"]
        if n_in == 0:
            if input_text.strip() != "":
                raise _err("declared 0 inputs but input field not blank",
                           node_id=nid, field="inputs")
            inputs = []
        else:
            if len(input_text) != n_in * 4:
                raise _err("input field width %d != 4 * n_inputs %d"
                           % (len(input_text), n_in * 4),
                           node_id=nid, field="inputs")
            inputs = []
            for j in range(n_in):
                token = input_text[j * 4:(j + 1) * 4]
                if not token.isdigit():
                    raise _err("input id must be 4 digits",
                               node_id=nid, field="inputs")
                inputs.append("n%04d" % int(token))
        if kind == "N":
            nodes.append(Node(node_id=nid, kind=NODE_INPUT, bitwidth=bw,
                              scale_permille=scale, inputs=inputs))
        else:  # "C"
            nodes.append(Node(node_id=nid, kind=NODE_CONV, bitwidth=bw,
                              scale_permille=scale, kernel=kernel,
                              stride=stride, padding=pad, inputs=inputs))
    return Graph(nodes=nodes, source="v1")


def _field(line_part: str) -> str:
    return line_part


def dump_v1(graph: Graph) -> str:
    out = ["%s %d" % (MAGIC, len(graph.nodes))]
    for n in graph.nodes:
        kind = "N" if n.kind == NODE_INPUT else "C"
        num = int(n.node_id[1:])
        bw = BW_CODE[n.bitwidth]
        scale = "%+06d" % n.scale_permille
        n_in = len(n.inputs)
        if n_in > 4:
            raise _err("too many inputs for v1 fixed width",
                       node_id=n.node_id, field="inputs")
        inputs = "".join("%04d" % int(pid[1:]) for pid in n.inputs)
        inputs = inputs.ljust(4)
        line = (kind + "%04d" % num + bw + scale
                + "%02d%02d%02d" % (n.stride, n.padding, n.kernel)
                + str(n_in) + inputs)
        if len(line) != LINE_LEN:
            raise _err("serialized line width %d != %d" % (len(line), LINE_LEN),
                       node_id=n.node_id)
        out.append(line)
    return "\n".join(out) + "\n"
