"""Track initialization, scoring, and deletion helpers.

Part H supplies lidar-driven existence decisions (docs/HUONG_DAN_KY_THUAT.md §2).
Use tracking parameters for the score window, thresholds, and covariance limit.
"""

from __future__ import annotations

import os
# Cho phép Intel OpenMP chạy cùng PyTorch trên Windows tránh lỗi crash abort
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from typing import Any

from fusion_lab.workspace_support import get_tracking_params
import numpy as np


def _get_sensor_to_vehicle_transform(sensor: Any) -> tuple[np.ndarray, np.ndarray]:
    """Lấy ma trận quay R (3x3) và vector tịnh tiến t (3x1) từ sensor frame sang vehicle frame."""
    if hasattr(sensor, "sens_to_veh"):
        T = sensor.sens_to_veh
        if hasattr(T, "shape") and len(T.shape) == 2:
            return np.asarray(T[:3, :3]), np.asarray(T[:3, 3]).reshape(3, 1)
        if hasattr(T, "R") and hasattr(T, "t"):
            return np.asarray(T.R), np.asarray(T.t).reshape(3, 1)

    if hasattr(sensor, "veh_to_sens"):
        v2s = sensor.veh_to_sens
        if hasattr(v2s, "shape") and len(v2s.shape) == 2:
            T_inv = np.linalg.inv(v2s)
            return np.asarray(T_inv[:3, :3]), np.asarray(T_inv[:3, 3]).reshape(3, 1)

    if hasattr(sensor, "R") and hasattr(sensor, "t"):
        return np.asarray(sensor.R), np.asarray(sensor.t).reshape(3, 1)

    return np.eye(3), np.zeros((3, 1))


def _get_param(params: Any, keys: list[str], default: float) -> float:
    """Truy xuất tham số an toàn từ đối tượng tracking params."""
    for k in keys:
        if hasattr(params, k):
            return float(getattr(params, k))
        if isinstance(params, dict) and k in params:
            return float(params[k])
    return default


def init_track_state_from_meas(meas: Any) -> dict[str, Any]:
    """Initialize track state, covariance, lifecycle state, and score from a measurement.

    Args:
        meas: Lidar measurement with ``z``, ``R``, ``sensor``.

    Returns:
        Dict with keys ``x``, ``P``, ``state``, ``score`` (matrices as ``np.matrix``).
    """
    # vi: TODO Part H — đổi meas.z sang vehicle frame; x = [pos; 0 velocity];
    # vi: P block pos từ R xoay, vel từ sigma_p44/55/66; score = 1/window; state initialized.
    params = get_tracking_params()
    sensor = getattr(meas, "sensor", None)

    # 1. Đổi meas.z sang vehicle frame
    R_rot, t_vec = _get_sensor_to_vehicle_transform(sensor)
    z_vec = np.asarray(meas.z)[:3].reshape(3, 1)
    pos = R_rot @ z_vec + t_vec

    # Trạng thái x = [px, py, pz, vx, vy, vz]^T với vận tốc khởi tạo = 0
    x = np.matrix(np.zeros((6, 1), dtype=float))
    x[:3, 0] = pos

    # 2. Xây dựng ma trận hiệp phương sai P (6x6)
    # Block vị trí (3x3) từ R xoay: R_rot @ meas.R @ R_rot^T
    R_meas = np.asarray(meas.R)[:3, :3]
    P_pos = R_rot @ R_meas @ R_rot.T

    # Block vận tốc từ sigma_p44, sigma_p55, sigma_p66
    p44 = _get_param(params, ["sigma_p44", "p44", "sigma_vel_x"], 50.0)
    p55 = _get_param(params, ["sigma_p55", "p55", "sigma_vel_y"], 50.0)
    p66 = _get_param(params, ["sigma_p66", "p66", "sigma_vel_z"], 5.0)

    P = np.matrix(np.zeros((6, 6), dtype=float))
    P[:3, :3] = P_pos
    P[3, 3] = p44
    P[4, 4] = p55
    P[5, 5] = p66

    # 3. Điểm số ban đầu: score = 1 / window
    window = _get_param(params, ["window", "score_window"], 10.0)
    score = 1.0 / window

    # 4. Trạng thái vòng đời ban đầu: "initialized"
    state = "initialized"

    return {
        "x": x,
        "P": P,
        "state": state,
        "score": score,
    }


def update_track_score(track: dict[str, Any], associated: bool) -> dict[str, Any]:
    """Update existence once per lidar frame; camera passes never call this helper.

    A hit adds 1/window, capped at one; an in-FOV miss subtracts 1/window.
    Confirm above confirmed_threshold, and preserve confirmed state after misses.

    Args:
        track: Dict-like track with ``score``, ``state``.
        associated: True for a lidar hit; False for a lidar miss within the lidar FOV.

    Returns:
        Updated track dict.
    """
    # vi: TODO Part H — chỉ lidar: hit +1/window (tối đa 1), miss trong FOV -1/window.
    # vi: score > confirmed_threshold → confirmed; đã confirmed không hạ trạng thái.
    # vi: Camera không gọi hàm này; track chưa confirmed với hit → tentative.
    params = get_tracking_params()
    window = _get_param(params, ["window", "score_window"], 10.0)
    confirmed_threshold = _get_param(params, ["confirmed_threshold"], 0.6)

    score = float(track["score"] if "score" in track else track.score)
    state = str(track["state"] if "state" in track else track.state)

    if associated:
        # Hit LiDAR: cộng 1/window, tối đa bằng 1.0
        score = min(1.0, score + 1.0 / window)
        if score > confirmed_threshold:
            state = "confirmed"
        elif state != "confirmed":
            state = "tentative"
    else:
        # Miss LiDAR trong FOV: trừ 1/window
        score = score - 1.0 / window
        # Lưu ý: Track đã "confirmed" KHÔNG bị hạ cấp trạng thái khi miss

    if isinstance(track, dict):
        track["score"] = score
        track["state"] = state
    else:
        track.score = score
        track.state = state

    return track


def should_delete_track(track: dict[str, Any]) -> bool:
    """Return whether a lidar lifecycle pass should remove this track.

    Delete if either horizontal variance exceeds max_P, or if a confirmed
    track has score < delete_threshold, or an unconfirmed track has score <= 0.
    Camera passes never trigger deletion.

    Args:
        track: Dict with ``score``, ``state``, ``P``.

    Returns:
        True if track should be removed.
    """
    # vi: TODO Part H — Pxx hoặc Pyy > max_P: xóa bất kể score.
    # vi: confirmed: xóa khi score < delete_threshold; chưa confirmed: score <= 0.
    # vi: Các điều kiện là OR; camera không đánh giá/xóa track.
    params = get_tracking_params()
    max_P = _get_param(params, ["max_P", "max_p"], 10.0)
    delete_threshold = _get_param(params, ["delete_threshold"], 0.4)

    P = track["P"] if "P" in track else track.P
    score = float(track["score"] if "score" in track else track.score)
    state = str(track["state"] if "state" in track else track.state)

    # 1. Bất định vị trí ngang P[0, 0] hoặc P[1, 1] vượt quá max_P: xóa bất kể điểm số
    pxx = float(P[0, 0])
    pyy = float(P[1, 1])
    if pxx > max_P or pyy > max_P:
        return True

    # 2. Track đã confirmed: xóa khi score < delete_threshold
    if state == "confirmed":
        if score < delete_threshold:
            return True
    else:
        # 3. Track chưa confirmed (initialized/tentative): xóa khi score <= 0
        if score <= 0:
            return True

    return False
