"""确定性内容指纹（FNV-1a 64 位）与规范化字节编码。

所有参与缓存键的对象先经过 fp_material 规范化为可比较的基本类型，
再以带类型标签的字节序列喂给 FNV-1a，保证同一内容在任意机器、任意
Python 字典顺序下指纹逐字段一致。指纹以 16 位十六进制字符串表示。
"""

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK64 = 0xFFFFFFFFFFFFFFFF


def fnv1a64(data: bytes) -> str:
    h = FNV_OFFSET
    for byte in data:
        h ^= byte
        h = (h * FNV_PRIME) & MASK64
    return f"{h:016x}"


def fp_material(value):
    """把任意输入转换为确定性的、可递归比较的基本结构。"""
    if value is None:
        return ("null",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        raise TypeError("指纹输入不允许使用浮点，请先量化为定点整数")
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, (list, tuple)):
        return ("list", [fp_material(item) for item in value])
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda kv: kv[0])
        return ("dict", [(key, fp_material(val)) for key, val in items])
    if hasattr(value, "fingerprint_material"):
        return fp_material(value.fingerprint_material())
    if hasattr(value, "to_dict"):
        return fp_material(value.to_dict())
    raise TypeError(f"不支持指纹编码的类型: {type(value)!r}")


def _encode(material) -> bytes:
    tag = material[0]
    if tag == "null":
        return b"N|"
    if tag == "bool":
        return b"B|1|" if material[1] else b"B|0|"
    if tag == "int":
        return b"I|" + str(material[1]).encode("utf-8") + b"|"
    if tag == "str":
        payload = material[1].encode("utf-8")
        return b"S|" + str(len(payload)).encode("ascii") + b"|" + payload
    if tag == "list":
        out = b"L|" + str(len(material[1])).encode("ascii") + b"|"
        for item in material[1]:
            out += _encode(item)
        return out
    if tag == "dict":
        out = b"D|" + str(len(material[1])).encode("ascii") + b"|"
        for key, val in material[1]:
            out += _encode(fp_material(key))
            out += _encode(val)
        return out
    raise ValueError(f"未知指纹标签: {tag}")


def content_fingerprint(value) -> str:
    return fnv1a64(_encode(fp_material(value)))


def composite_fingerprint(*parts) -> str:
    """把多个已规范化材料合成为一个指纹。"""
    return content_fingerprint([fp_material(part) for part in parts])
