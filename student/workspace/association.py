"""Measurement-to-track association via Mahalanobis gating and greedy matching.

Part F supplies the association stage shown in docs/HUONG_DAN_KY_THUAT.md §2.
Load ``kalman`` with ``load_workspace_module`` for innovation helpers and tracking parameters
for the chi-square gate.
"""

from __future__ import annotations
import os
# Tránh lỗi xung đột OpenMP trên Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from typing import Any
from typing import Sequence

import numpy as np
from scipy.stats import chi2

from fusion_lab.workspace_support import get_tracking_params
from fusion_lab.workspace_loader import load_workspace_module
kalman = load_workspace_module("kalman")  # không dùng `import kalman`


def _is_in_fov(track: Any, sensor: Any) -> bool:
    """Kiểm tra track có nằm trong FOV của sensor hay không."""
    if hasattr(sensor, "in_fov") and callable(sensor.in_fov):
        try:
            return bool(sensor.in_fov(track.x))
        except Exception:
            return bool(sensor.in_fov(track))
    try:
        camera_fusion = load_workspace_module("camera_fusion")
        return bool(camera_fusion.is_in_field_of_view(track.x, sensor))
    except Exception:
        return True


def mahalanobis_distance(track: Any, meas: Any) -> float:
    """Return squared Mahalanobis distance between a track and a measurement.

    Args:
        track: Track with ``x``, ``P``.
        meas: Measurement with ``sensor``.

    Returns:
        Scalar squared Mahalanobis distance.
    """
    # vi: TODO Part F — H = meas.sensor.get_H(track.x);
    # vi: gamma = kalman.innovation(...); S = kalman.innovation_covariance(...);
    # vi: return gamma.T @ inv(S) @ gamma (float scalar).
    # 1. Lấy ma trận Jacobian đo H
    H = meas.sensor.get_H(track.x)

    # 2. Tính innovation gamma và hiệp phương sai S
    gamma = kalman.innovation(track.x, meas)
    S = kalman.innovation_covariance(track.P, meas, H)

    # 3. Tính khoảng cách Mahalanobis bình phương: d^2 = gamma^T @ S^{-1} @ gamma
    inv_S = np.linalg.inv(S)
    mhd_sq = gamma.T @ inv_S @ gamma

    return float(np.asarray(mhd_sq).item())


def chi2_gate(mhd_sq: float, sensor: Any) -> bool:
    """Return True if squared Mahalanobis distance lies inside the chi-square gate.

    Args:
        mhd_sq: Squared Mahalanobis distance.
        sensor: Sensor with ``dim_meas``.

    Returns:
        True if inside gate.
    """
    # vi: TODO Part F — ngưỡng chi2.ppf(gating_threshold, sensor.dim_meas) từ params.
    params = get_tracking_params()
    gating_threshold = getattr(
        params,
        "gating_threshold",
        params.get("gating_threshold", 0.95) if isinstance(params, dict) else 0.95,
    )

    # Ngưỡng chi-square: chi2.ppf(p, df=sensor.dim_meas)
    gate_limit = chi2.ppf(gating_threshold, sensor.dim_meas)

    return bool(mhd_sq <= gate_limit)


def association_cost_matrix(
    track_list: Sequence[Any], meas_list: Sequence[Any]
) -> np.matrix:
    """Build gated costs, checking each sensor's visibility before projection.

    Args:
        track_list: Active tracks.
        meas_list: Measurements for this sensor pass.

    Returns:
        Cost matrix; ``np.inf`` for invisible tracks or rejected chi-square gates.
        Invisible pairs must never call the Mahalanobis/projection helpers.
    """
    # vi: TODO Part F — khởi tạo toàn inf; kiểm tra meas.sensor.in_fov(track.x)
    # vi: trước MHD (camera sau lưng/độ sâu 0 không được chiếu); rồi kiểm tra chi2.
    num_tracks = len(track_list)
    num_meas = len(meas_list)
    cost_matrix = np.full((num_tracks, num_meas), np.inf, dtype=float)

    if num_tracks == 0 or num_meas == 0:
        return np.matrix(cost_matrix)

    for i, track in enumerate(track_list):
        for j, meas in enumerate(meas_list):
            sensor = getattr(meas, "sensor", None)

            # BẮT BUỘC: Kiểm tra FOV trước khi tính Mahalanobis/chiếu camera
            if sensor is not None and not _is_in_fov(track, sensor):
                continue

            # Tính khoảng cách Mahalanobis bình phương
            mhd_sq = mahalanobis_distance(track, meas)

            # Kiểm tra cổng chi2
            if chi2_gate(mhd_sq, sensor):
                cost_matrix[i, j] = mhd_sq

    return np.matrix(cost_matrix)


def pick_next_pair(
    association_matrix: np.matrix,
    unassigned_tracks: Sequence[Any],
    unassigned_meas: Sequence[Any],
) -> tuple[Any, Any, np.matrix, list[Any], list[Any]]:
    """Pick the minimum-cost track/measurement pair and shrink the association problem.

    Args:
        association_matrix: Current cost matrix.
        unassigned_tracks: Track objects still free.
        unassigned_meas: Measurement objects still free.

    Returns:
        Tuple (track, meas, new_matrix, remaining_tracks, remaining_meas).
        If no finite pair exists, return np.nan for track and meas and retain both lists.
    """
    # vi: TODO Part F — chỉ lấy cặp hữu hạn nhỏ nhất rồi xóa hàng/cột tương ứng;
    # vi: ma trận rỗng/toàn inf: trả np.nan, np.nan và giữ các danh sách chưa ghép.
    if (
        association_matrix.size == 0
        or len(unassigned_tracks) == 0
        or len(unassigned_meas) == 0
    ):
        return np.nan, np.nan, association_matrix, list(unassigned_tracks), list(unassigned_meas)

    min_val = np.min(association_matrix)

    # Nếu toàn bộ cost là vô cùng (np.inf) hoặc không có giá trị hữu hạn
    if not np.isfinite(min_val):
        return np.nan, np.nan, association_matrix, list(unassigned_tracks), list(unassigned_meas)

    # Tìm chỉ số (hàng, cột) có cost nhỏ nhất
    row, col = np.unravel_index(np.argmin(association_matrix), association_matrix.shape)
    row = int(row)
    col = int(col)

    track = unassigned_tracks[row]
    meas = unassigned_meas[col]

    # Thu nhỏ ma trận chi phí bằng cách xóa hàng row và cột col đã gán
    new_matrix = np.delete(association_matrix, row, axis=0)
    new_matrix = np.delete(new_matrix, col, axis=1)
    new_matrix = np.matrix(new_matrix)

    remaining_tracks = [t for i, t in enumerate(unassigned_tracks) if i != row]
    remaining_meas = [m for j, m in enumerate(unassigned_meas) if j != col]

    return track, meas, new_matrix, remaining_tracks, remaining_meas


def associate_and_update(
    manager: Any,
    meas_list: Sequence[Any],
    filter_obj: Any,
    sensor: Any,
) -> None:
    """Greedy association loop with EKF updates and track management.

    Args:
        manager: Track manager (``track_list``, ``manage_tracks``, ...).
        meas_list: Lidar or camera measurements for this frame pass.
        filter_obj: Filter with ``predict`` / ``update``.
        sensor: Explicit lidar/camera pass sensor, including empty measurement frames.

    Returns:
        None; updates tracks in place and always finishes the lifecycle pass.
        Visibility is handled in the cost matrix, before pair removal. Camera
        updates refine state only; lidar hits alone increase existence scores.
    """
    # vi: TODO Part F — kể cả meas_list rỗng, vẫn gọi quản lý cuối lượt.
    # vi: Ghép cặp hữu hạn, filter_obj.update rồi handle_updated_track(track, sensor).
    # vi: Không bỏ qua FOV sau khi đã xóa cặp khỏi danh sách chưa ghép.
    # vi: Kết thúc manager.manage_tracks(unassigned_tracks, unassigned_meas, sensor).
    unassigned_tracks = list(manager.track_list)
    unassigned_meas = list(meas_list)

    # Xây dựng ma trận chi phí ban đầu
    cost_matrix = association_cost_matrix(unassigned_tracks, unassigned_meas)

    # Vòng lặp gán Greedy
    while True:
        track, meas, cost_matrix, unassigned_tracks, unassigned_meas = pick_next_pair(
            cost_matrix, unassigned_tracks, unassigned_meas
        )

        # Kết thúc khi không còn cặp hữu hạn nào
        if track is np.nan or (isinstance(track, float) and np.isnan(track)):
            break

        # Cập nhật trạng thái EKF
        filter_obj.update(track, meas)

        # Xử lý track đã được update
        if hasattr(manager, "handle_updated_track"):
            manager.handle_updated_track(track, sensor)
        else:
            try:
                tm = load_workspace_module("track_management")
                if hasattr(tm, "handle_updated_track"):
                    tm.handle_updated_track(track, sensor)
            except Exception:
                pass

    # Luôn gọi quản lý vòng đời track ở cuối lượt (kể cả khi meas_list rỗng)
    manager.manage_tracks(unassigned_tracks, unassigned_meas, sensor)
