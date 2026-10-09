"""Camera field-of-view checks and pinhole measurement modeling.

Part G supplies visibility, projection, and pixel covariance (docs/HUONG_DAN_KY_THUAT.md §2).
The platform differentiates projection using a chain-rule Jacobian.
"""

from __future__ import annotations

from typing import Any
from typing import Sequence

import numpy as np

Matrix = np.matrix | np.ndarray

from fusion_lab.workspace_support import get_tracking_params


def _transform_to_sensor(x: Matrix, sensor: Any) -> np.ndarray:
    """Transform vehicle-frame 3D position [px, py, pz] to sensor frame coordinates [x_s, y_s, z_s].

    Formula: p_s = R @ p + t
    """
    p = np.asarray(x)[:3].reshape(3, 1)

    # Trích xuất R và t từ sensor.veh_to_sens (ma trận biến đổi 4x4) hoặc các thuộc tính R, t
    if hasattr(sensor, "veh_to_sens"):
        v2s = sensor.veh_to_sens
        if callable(v2s):
            return np.asarray(v2s(p)).reshape(3, 1)
        if hasattr(v2s, "shape") and len(v2s.shape) == 2:
            R = np.asarray(v2s[:3, :3])
            t = np.asarray(v2s[:3, 3]).reshape(3, 1)
            return R @ p + t
        if hasattr(v2s, "R") and hasattr(v2s, "t"):
            R = np.asarray(v2s.R)
            t = np.asarray(v2s.t).reshape(3, 1)
            return R @ p + t

    if hasattr(sensor, "R") and hasattr(sensor, "t"):
        R = np.asarray(sensor.R)
        t = np.asarray(sensor.t).reshape(3, 1)
        return R @ p + t

    raise AttributeError("Sensor must have veh_to_sens (4x4 matrix) or R and t attributes")


class MeasurementDict(dict):
    """Dictionary that supports both dict indexing (meas['z']) and attribute access (meas.z)."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'MeasurementDict' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def is_in_field_of_view(x: Matrix, sensor: Any) -> bool:
    """Return True if state x is visible within the sensor horizontal field of view.

    Args:
        x: State vector (6x1) with position in vehicle frame.
        sensor: Lidar or camera adapter with ``veh_to_sens`` and ``fov``
            (radians).

    Returns:
        True if sensor coordinates are finite and the horizontal angle is within
        ``sensor.fov``. A camera additionally requires depth > 1e-6.
    """
    # vi: TODO Part G — p_s = R @ p + t; loại tọa độ không hữu hạn.
    # vi: Camera cần x_s > 1e-6; FOV từ nội tại và bề rộng ảnh.
    # vi: Với cả lidar/camera: kiểm tra atan2(y_s, x_s) nằm trong sensor.fov.
    # 1. Tính tọa độ hệ cảm biến: p_s = R @ p + t
    try:
        p_s = _transform_to_sensor(x, sensor)
    except Exception:
        return False

    # Loại bỏ tọa độ không hữu hạn (NaN / Inf)
    if not np.all(np.isfinite(p_s)):
        return False

    x_s = float(p_s[0, 0])
    y_s = float(p_s[1, 0])
    z_s = float(p_s[2, 0])

    # 2. Kiểm tra nếu là Camera: bắt buộc độ sâu x_s > 1e-6 (nằm trước mặt camera)
    is_camera = (
        hasattr(sensor, "f_i")
        or hasattr(sensor, "c_i")
        or getattr(sensor, "name", "") == "camera"
        or getattr(sensor, "sensor_type", "") == "camera"
        or getattr(sensor, "modality", "") == "camera"
        or "camera" in str(type(sensor).__name__).lower()
    )

    if is_camera and x_s <= 1e-6:
        return False

    # 3. Tính góc ngang (azimuth): angle = atan2(y_s, x_s)
    angle = np.arctan2(y_s, x_s)

    # 4. Kiểm tra góc nằm trong sensor.fov
    fov = getattr(sensor, "fov", None)
    if fov is None and is_camera and hasattr(sensor, "c_i") and hasattr(sensor, "f_i"):
        # Dự phòng tính FOV từ nội tại c_i, f_i và bề rộng ảnh nếu sensor chưa có sẵn fov
        w = getattr(sensor, "width", getattr(sensor, "image_width", 2.0 * sensor.c_i))
        fov_min = np.arctan((sensor.c_i - w) / sensor.f_i)
        fov_max = np.arctan(sensor.c_i / sensor.f_i)
        fov = (min(fov_min, fov_max), max(fov_min, fov_max))

    if fov is not None:
        if hasattr(fov, "__len__") and len(fov) >= 2:
            fov_min = min(fov[0], fov[1])
            fov_max = max(fov[0], fov[1])
            return bool(fov_min <= angle <= fov_max)
        else:
            fov_val = float(fov)
            return bool(-fov_val / 2.0 <= angle <= fov_val / 2.0)

    return True


def camera_measurement_prediction(x: Matrix, sensor: Any) -> Matrix:
    """Predict image-plane measurement h(x) using the pinhole camera model.

    Args:
        x: State vector.
        sensor: Camera with intrinsics ``f_i, f_j, c_i, c_j``.

    Returns:
        2x1 predicted pixel coordinates as ``np.matrix``.

    Raises:
        ValueError: With coordinate context if sensor coordinates are nonfinite
            or depth is at most 1e-6.
    """
    # vi: TODO Part G — tính p_s = R @ p + t; trước phép chia kiểm tra hữu hạn
    # vi: và x_s > 1e-6, ngược lại raise ValueError có tọa độ.
    # vi: u = c_i - f_i * y_s/x_s; v = c_j - f_j * z_s/x_s.
    # 1. Tính p_s = R @ p + t
    p_s = _transform_to_sensor(x, sensor)

    # 2. Trước phép chia, kiểm tra tính hữu hạn và độ sâu x_s > 1e-6
    if not np.all(np.isfinite(p_s)):
        raise ValueError(f"Non-finite sensor coordinates: p_s={p_s.flatten().tolist()}")

    x_s = float(p_s[0, 0])
    y_s = float(p_s[1, 0])
    z_s = float(p_s[2, 0])

    if x_s <= 1e-6:
        raise ValueError(
            f"Camera depth must be > 1e-6, got x_s={x_s} (coordinates: y_s={y_s}, z_s={z_s})"
        )

    # 3. Mô hình Pinhole camera hệ trục Waymo:
    # u = c_i - f_i * (y_s / x_s)
    # v = c_j - f_j * (z_s / x_s)
    u = sensor.c_i - sensor.f_i * (y_s / x_s)
    v = sensor.c_j - sensor.f_j * (z_s / x_s)

    return np.matrix([[float(u)], [float(v)]])


def build_camera_measurement(z: Sequence[float], sensor: Any) -> dict[str, Any]:
    """Build camera measurement vector z and covariance R from pixel coordinates.

    Args:
        z: Sequence ``[u, v]`` pixel coordinates.
        sensor: Camera sensor object.

    Returns:
        Dict with keys ``z``, ``R``, ``sensor``.
    """
    # vi: TODO Part G — z mat 2x1; R diag sigma_cam_i^2, sigma_cam_j^2 từ params.
    p = get_tracking_params()
    sigma_i = getattr(p, "sigma_cam_i", getattr(sensor, "sigma_cam_i", 1.0))
    sigma_j = getattr(p, "sigma_cam_j", getattr(sensor, "sigma_cam_j", 1.0))

    # Vector đo lường pixel 2x1
    z_mat = np.matrix([[float(z[0])], [float(z[1])]])

    # Ma trận hiệp phương sai nhiễu đo R (2x2 đường chéo)
    R = np.matrix([
        [float(sigma_i) ** 2, 0.0],
        [0.0, float(sigma_j) ** 2],
    ])

    return MeasurementDict({
        "z": z_mat,
        "R": R,
        "sensor": sensor,
    })
