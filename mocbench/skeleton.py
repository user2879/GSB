from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from . import quat
from .errors import BenchError


@dataclass
class Bone:
    name: str
    parent: Optional[str]
    rest_rotation: np.ndarray
    length: float


@dataclass
class Skeleton:
    bones: Dict[str, Bone]
    order: List[str]
    root: str
    children: Dict[str, List[str]]
    rest_offsets: Dict[str, np.ndarray]
    rest_positions: Dict[str, np.ndarray]

    def forward_kinematics(self, root_position, local_rotations):
        world_rot = {}
        world_pos = {}
        root_position = np.asarray(root_position, dtype=np.float64)
        world_rot[self.root] = quat.normalize(local_rotations[self.root])
        world_pos[self.root] = root_position.copy()
        for name in self.order[1:]:
            bone = self.bones[name]
            parent_wr = world_rot[bone.parent]
            world_rot[name] = quat.mul(parent_wr, quat.normalize(local_rotations[name]))
            world_pos[name] = world_pos[bone.parent] + quat.rotate(
                parent_wr, self.rest_offsets[name]
            )
        return world_pos, world_rot


def build_skeleton(bone_defs):
    if not isinstance(bone_defs, list) or not bone_defs:
        raise BenchError("E_INPUT", "skeleton.bones must be a non-empty list")
    names = []
    bones = {}
    for item in bone_defs:
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise BenchError("E_BONE", "every bone requires a string name", item)
        if name in names:
            raise BenchError("E_BONE", f"duplicate bone name: {name}", {"bone": name})
        parent = item.get("parent")
        if parent is not None and not isinstance(parent, str):
            raise BenchError("E_BONE", "parent must be null or a string", {"bone": name})
        length = item.get("length")
        if not isinstance(length, (int, float)) or not np.isfinite(length) or length <= 0:
            raise BenchError(
                "E_LENGTH",
                f"bone length must be a finite positive number: {name}",
                {"bone": name, "length": length},
            )
        rest = item.get("rest_rotation", [1.0, 0.0, 0.0, 0.0])
        rest = _finite_quat(rest, name, allow_nonunit=True)
        bones[name] = Bone(name=name, parent=parent, rest_rotation=rest, length=float(length))
        names.append(name)

    roots = [n for n in names if bones[n].parent is None]
    if len(roots) != 1:
        raise BenchError(
            "E_HIERARCHY",
            f"skeleton must have exactly one root, found {len(roots)}",
            {"roots": roots},
        )
    for name in names:
        parent = bones[name].parent
        if parent is not None and parent not in bones:
            raise BenchError(
                "E_HIERARCHY",
                f"bone {name} references unknown parent {parent}",
                {"bone": name, "parent": parent},
            )

    indegree = {n: 0 for n in names}
    children = {n: [] for n in names}
    for name in names:
        parent = bones[name].parent
        if parent is not None:
            indegree[name] += 1
            children[parent].append(name)
    ready = [n for n in names if indegree[n] == 0]
    order = []
    while ready:
        current = ready.pop(0)
        order.append(current)
        for child in children[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if len(order) != len(names):
        unresolved = [n for n in names if indegree[n] > 0]
        raise BenchError(
            "E_HIERARCHY",
            "skeleton hierarchy contains a cycle",
            {"bones_in_cycle": unresolved},
        )

    rest_offsets = {}
    rest_positions = {}
    for name in order:
        bone = bones[name]
        if bone.parent is None:
            rest_offsets[name] = np.zeros(3, dtype=np.float64)
            rest_positions[name] = np.zeros(3, dtype=np.float64)
        else:
            offset = quat.rotate(bone.rest_rotation, np.array([0.0, 0.0, bone.length]))
            rest_offsets[name] = offset
            rest_positions[name] = rest_positions[bone.parent] + offset

    root = roots[0]
    return Skeleton(
        bones=bones,
        order=order,
        root=root,
        children=children,
        rest_offsets=rest_offsets,
        rest_positions=rest_positions,
    )


def _finite_quat(value, bone, allow_nonunit):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise BenchError("E_QUATERNION", f"quaternion for {bone} must have 4 components")
    arr = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise BenchError("E_QUATERNION", f"non-finite quaternion for {bone}", {"bone": bone})
    if not allow_nonunit and not quat.is_unit(arr):
        raise BenchError("E_QUATERNION", f"non-unit quaternion for {bone}", {"bone": bone})
    return quat.normalize(arr)
