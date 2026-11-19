import math

import numpy as np

from . import quat
from .parser import Constraints, InputBundle
from .skeleton import Skeleton

SOLVE_ORDER = ["angle_limit", "ik", "bone_length", "contact"]


class PoseCache:
    def __init__(self, skeleton: Skeleton, root_pos, local):
        self.skeleton = skeleton
        self.order = skeleton.order
        self.root_pos = np.asarray(root_pos, dtype=np.float64).copy()
        self.local = {name: quat.normalize(local[name]) for name in self.order}
        self.world_rot = {}
        self.world_pos = {}
        self.rebuild_all()

    def rebuild_all(self):
        root = self.skeleton.root
        self.world_rot[root] = self.local[root].copy()
        self.world_pos[root] = self.root_pos.copy()
        for name in self.order[1:]:
            self._recompute(name)

    def _recompute(self, name):
        bone = self.skeleton.bones[name]
        parent = bone.parent
        parent_wr = self.world_rot[parent]
        self.world_rot[name] = quat.mul(parent_wr, self.local[name])
        self.world_pos[name] = self.world_pos[parent] + quat.rotate(
            parent_wr, self.skeleton.rest_offsets[name]
        )

    def update_subtree(self, joint):
        stack = [joint]
        while stack:
            current = stack.pop()
            if current != self.skeleton.root:
                self._recompute(current)
            stack.extend(self.skeleton.children[current])

    def set_local(self, name, rotation):
        self.local[name] = quat.normalize(rotation)
        self.update_subtree(name)


def solve_frame(frame_index, initial_local, initial_root, bundle: InputBundle):
    skeleton: Skeleton = bundle.skeleton
    constraints: Constraints = bundle.constraints
    params = bundle.params
    damping = params.damping
    max_iterations = params.max_iterations
    tolerance = params.tolerance

    cache = PoseCache(skeleton, initial_root, initial_local)

    ik_by_frame = {}
    for target in constraints.ik_targets:
        for sub_index, frame_id in enumerate(target["frames"]):
            ik_by_frame.setdefault(frame_id, {})[target["bone"]] = target["targets"][sub_index]
    active_ik = ik_by_frame.get(frame_index, {})
    active_contact = frame_index in constraints.contact_frames and bool(constraints.contact_bones)

    active = ["angle_limit", "bone_length"]
    if active_ik:
        active.insert(1, "ik")
    if active_contact:
        active.append("contact")
    traces = {name: [] for name in active}

    for _iteration in range(max_iterations):
        traces["angle_limit"].append(_apply_angle_limits(cache, constraints, damping))
        if active_ik:
            traces["ik"].append(_apply_ccd(cache, active_ik, constraints, damping))
        traces["bone_length"].append(_apply_bone_length(cache))
        if active_contact:
            traces["contact"].append(_apply_contact(cache, constraints, damping))
        if all(traces[name][-1] <= tolerance for name in active):
            break

    iterations_used = len(traces[active[0]])
    final_residuals = []
    converged = True
    for name in active:
        final_value = traces[name][-1]
        passed = final_value <= tolerance
        converged = converged and passed
        final_residuals.append(
            {
                "constraint": name,
                "target": _target_name(name, active_ik),
                "tolerance": tolerance,
                "final_residual": final_value,
                "passed": passed,
                "last_iteration_trace": list(traces[name]),
            }
        )

    return {
        "frame": frame_index,
        "timestamp": frame_index / params.fps,
        "iterations_used": iterations_used,
        "converged": converged,
        "local_rotations": {name: cache.local[name].copy() for name in skeleton.order},
        "root_position": cache.root_pos.copy(),
        "world_positions": {name: cache.world_pos[name].copy() for name in skeleton.order},
        "residuals": final_residuals,
        "solve_order": list(active),
    }


def _target_name(name, active_ik):
    if name == "angle_limit":
        return "limited_bones"
    if name == "bone_length":
        return "all_bones"
    if name == "contact":
        return "contact_bones"
    return ",".join(sorted(active_ik))


def _joint_angle(local_rotation, axis):
    rotation = np.asarray(local_rotation, dtype=np.float64)
    w = float(np.clip(rotation[0], -1.0, 1.0))
    axis_part = rotation[1:]
    projection = float(np.dot(axis_part, axis))
    magnitude = np.linalg.norm(axis_part)
    if magnitude < 1e-12:
        return 0.0
    signed = math.copysign(2.0 * math.acos(w), projection)
    if abs(signed) > math.pi:
        signed -= math.copysign(2.0 * math.pi, signed)
    return signed


def _apply_angle_limits(cache: PoseCache, constraints: Constraints, damping):
    residual = 0.0
    for limit in constraints.angle_limits:
        name = limit["bone"]
        angle = _joint_angle(cache.local[name], limit["axis"])
        violation = 0.0
        direction = 0.0
        if angle < limit["lower"]:
            violation = limit["lower"] - angle
            direction = 1.0
        elif angle > limit["upper"]:
            violation = angle - limit["upper"]
            direction = -1.0
        residual = max(residual, violation)
        if violation > 0.0:
            delta = quat.axis_angle(limit["axis"], direction * violation * damping)
            cache.set_local(name, quat.mul(delta, cache.local[name]))
    return residual


def _chain_to_root(skeleton: Skeleton, effector):
    chain = []
    current = skeleton.bones[effector].parent
    while current is not None:
        chain.append(current)
        current = skeleton.bones[current].parent
    return chain


def _apply_ccd(cache: PoseCache, active_ik, constraints: Constraints, damping):
    limit_by_bone = {item["bone"]: item for item in constraints.angle_limits}
    residual = 0.0
    for effector, target in active_ik.items():
        residual = max(residual, float(np.linalg.norm(cache.world_pos[effector] - target)))
        chain = _chain_to_root(cache.skeleton, effector)
        for joint in chain:
            joint_pos = cache.world_pos[joint]
            to_effector = cache.world_pos[effector] - joint_pos
            to_target = target - joint_pos
            if np.linalg.norm(to_effector) < 1e-12 or np.linalg.norm(to_target) < 1e-12:
                continue
            full_rotation = quat.rotation_between(to_effector, to_target)
            damped_rotation = _scale_rotation(full_rotation, damping)
            parent = cache.skeleton.bones[joint].parent
            parent_world = (
                np.array([1.0, 0.0, 0.0, 0.0])
                if parent is None
                else cache.world_rot[parent]
            )
            new_world = quat.normalize(quat.mul(damped_rotation, cache.world_rot[joint]))
            cache.set_local(joint, quat.relative_rotation(parent_world, new_world))
            if joint in limit_by_bone:
                _project_joint(cache, joint, limit_by_bone[joint])
    return residual


def _scale_rotation(rotation, damping):
    w = float(np.clip(rotation[0], -1.0, 1.0))
    angle = 2.0 * math.acos(w)
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = quat.normalize_vec(rotation[1:])
    return quat.axis_angle(axis, angle * damping)


def _project_joint(cache: PoseCache, name, limit):
    angle = _joint_angle(cache.local[name], limit["axis"])
    if angle < limit["lower"]:
        delta = quat.axis_angle(limit["axis"], limit["lower"] - angle)
        cache.set_local(name, quat.mul(delta, cache.local[name]))
    elif angle > limit["upper"]:
        delta = quat.axis_angle(limit["axis"], limit["upper"] - angle)
        cache.set_local(name, quat.mul(delta, cache.local[name]))


def _apply_bone_length(cache: PoseCache):
    residual = 0.0
    for name in cache.order:
        bone = cache.skeleton.bones[name]
        if bone.parent is None:
            continue
        actual = float(np.linalg.norm(cache.world_pos[name] - cache.world_pos[bone.parent]))
        residual = max(residual, abs(actual - bone.length))
    return residual


def _apply_contact(cache: PoseCache, constraints: Constraints, damping):
    heights = [
        cache.world_pos[name][1] - constraints.ground_height
        for name in constraints.contact_bones
    ]
    lowest = min(heights)
    residual = max(0.0, -float(lowest))
    if lowest < 0.0:
        cache.root_pos = cache.root_pos.copy()
        cache.root_pos[1] += -lowest * damping
        cache.rebuild_all()
    return residual
