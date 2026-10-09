# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Điền file này rồi commit. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Đỗ Hoàng Quân
- MSSV: 2A202603016
- Email: hoangquan11112004@gmail.com
- Link repo (fork): https://github.com/The-Spirit-of-the-Beehive/K4-L2L3-DAY23-DoHoangQuan-2A202603016-SensorFusion
- Commit hash nộp (`git rev-parse HEAD`): faf8ae4c0add0b14ec59eadd261d6fc542e66776

## Tóm tắt kết quả

- `fusion_mode` (bắt buộc `compare`), `frames`, `segment`, `seed`:
  - `fusion_mode`: "compare"
  - `frames`: [0, 198]
  - `segment`: "training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord"
  - `seed`: 0

- `detection.precision`, `detection.recall`, `detection.tp/fp/fn`:
  - `precision`: 0.9700934579439252 (~97.01%)
  - `recall`: 0.7004048582995951 (~70.04%)
  - `tp`: 519, `fp`: 16, `fn`: 222

- `tracking.lidar.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`:
  - `rmse`: 0.1502780969147606 m (~0.1503 m)
  - `matches`: 502
  - `sum_sq_err`: 11.336920218985732 m²
  - `ghost_track_frames`: 0
  - `missed_gt_frames`: 239
  - `mean_confirmed_tracks`: 2.522613065326633

- `tracking.fused.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`:
  - `rmse`: 0.13584776925720826 m (~0.1358 m)
  - `matches`: 502
  - `sum_sq_err`: 9.264217438904167 m²
  - `ghost_track_frames`: 0
  - `missed_gt_frames`: 239
  - `mean_confirmed_tracks`: 2.522613065326633

- **Giải thích khác biệt hai mode, đọc RMSE cùng số ghép và ghost/miss:**
  - **Chất lượng sai số vị trí (RMSE):** Chế độ Fused đạt RMSE là **0.1358 m**, cải thiện rõ rệt so với chế độ LiDAR đơn thuần là **0.1503 m**. Tổng bình phương sai số vị trí 3D (`sum_sq_err`) giảm mạnh từ **11.337 m²** xuống **9.264 m²** (giảm ~18.3% sai số tích lũy). Hiệu số `rmse_fused - rmse_lidar = -0.0144 m <= 0.05 m`, đáp ứng trọn vẹn tiêu chí fusion nhất quán của rubric.
  - **Tính tương đồng về số cặp ghép và vòng đời:** Cả hai chế độ đều có đúng **502 matches**, **0 ghost_track_frames**, **239 missed_gt_frames** và trung bình **2.5226 confirmed tracks/frame**. Điều này phản ánh chính xác kiến trúc **track-then-fuse**: toàn bộ vòng đời track (tạo mới, cộng/trừ score, xác nhận, xóa) chỉ do lượt LiDAR quyết định; lượt camera chỉ thực hiện EKF measurement update để tinh chỉnh vector trạng thái chứ không can thiệp vào vòng đời track.
  - **Độ chính xác và độ phủ track:**
    - `precision_track` = $502 / (502 + 0) = 1.0$ (100% track confirmed đều ghép đúng xe thật, không có ghost track nào).
    - `coverage` = $502 / 519 \approx 0.9672$ (~96.72% detection TP của detector được duy trì thành confirmed track).
    - Cả hai hệ số đều vượt xa ngưỡng tối đa của rubric (`precision_track ≥ 0.75` và `coverage ≥ 0.70`), giúp cả hai mode đạt điểm tối đa $q = 1.0$.
  - **Giới hạn thực nghiệm:** Phép đo camera trong lab được lấy từ tâm hộp 2D ground-truth camera FRONT có thêm nhiễu seeded, không chạy mô hình camera detector độc lập. Do đó, kết quả RMSE fused tốt hơn chứng minh thuật toán EKF kết hợp thông tin góc chiếu 2D hiệu quả để giảm bất định vị trí của track, chứ không phản ánh độ chính xác của một hệ thống perception camera thực tế trên xe.

## Giải thích ngắn (Parts E–H — tự viết)

### 1. Khác biệt đo lidar 3D và camera 2D trong EKF (`z`, `R`)?
- **Không gian đo $z$ và mô hình đo:**
  - LiDAR đo trực tiếp trong không gian Cartesian 3D: $z_{\text{lidar}} \in \mathbb{R}^{3 \times 1} = [x, y, z]^T$. Mô hình đo là tuyến tính $h(x) = H x$ với ma trận Jacobian $H \in \mathbb{R}^{3 \times 6}$ cố định trích xuất 3 tọa độ vị trí đầu tiên.
  - Camera đo trên mặt phẳng ảnh 2D pixel: $z_{\text{cam}} \in \mathbb{R}^{2 \times 1} = [u, v]^T$. Mô hình đo là phi tuyến (pinhole projection Waymo: $u = c_i - f_i \frac{y_s}{x_s}, v = c_j - f_j \frac{z_s}{x_s}$), và ma trận Jacobian $H \in \mathbb{R}^{2 \times 6}$ phải tính đạo hàm phi tuyến phụ thuộc vào trạng thái dự báo $x$.
- **Ma trận hiệp phương sai nhiễu đo $R$:**
  - $R_{\text{lidar}} \in \mathbb{R}^{3 \times 3}$: biểu diễn độ bất định vị trí vật lý theo đơn vị mét vuông ($\text{m}^2$).
  - $R_{\text{cam}} \in \mathbb{R}^{2 \times 2} = \text{diag}(\sigma_{\text{cam\_i}}^2, \sigma_{\text{cam\_j}}^2)$: biểu diễn độ bất định theo đơn vị $\text{pixel}^2$ trên ảnh.

### 2. Vì sao cần gating Mahalanobis trước khi gán?
- Khoảng cách Euclid thông thường chỉ đo khoảng cách hình học đơn thuần, bỏ qua hoàn toàn ma trận hiệp phương sai bất định $P$ của track và $R$ của cảm biến.
- Khoảng cách Mahalanobis $d^2 = \gamma^T S^{-1} \gamma$ (với $S = H P H^T + R$ và innovation $\gamma = z - h(x)$) chuẩn hóa khoảng cách theo elip bất định tổng hợp của cả track và sensor. Khi một track có độ bất định lớn theo trục chuyển động, một measurement ở xa hơn theo trục đó vẫn có thể có khoảng cách Mahalanobis nhỏ hơn một measurement gần hơn nhưng lệch khỏi elip phân bố.
- Cổng Chi-square $\chi^2$ (theo bậc tự do `dim_meas`) loại bỏ các đo lường ngoại lai (outliers/clutter) quá xa vùng tin cậy thống kê, ngăn chặn việc gán nhầm đối tượng và bảo vệ EKF không bị kéo lệch hoặc phân kỳ.

### 3. Pipeline là track-then-fuse hay fuse-then-track? Chỉ ra trên log `fusion-run-lab`.
- Pipeline của bài lab là **track-then-fuse**:
  - Hệ thống duy trì **một danh sách track duy nhất**, không hợp nhất dữ liệu thô (raw sensor) trước khi detection.
  - Ở mỗi frame, tracker thực hiện: EKF predict một lần cho toàn bộ track $\to$ gán và EKF update bằng LiDAR (**AssocL**) $\to$ quản lý vòng đời track (`manage_tracks`) $\to$ gán và EKF update bổ sung bằng Camera (**AssocC**) cho các track nằm trong vùng nhìn FOV camera.
- **Chỉ ra trên log:**
  - Trong `grade_run.log`, ở mỗi frame $k$, số lượng `confirmed` track hoàn toàn trùng khớp giữa mode `lidar` và mode `fused`. Bước camera không tạo thêm track mới hay xóa track hiện có; sự khác biệt duy nhất thể hiện ở giá trị `sum_sq_err` được tinh chỉnh nhỏ hơn nhờ bước update camera sau bước LiDAR.

### 4. Nếu camera lệch calibration, triệu chứng gì trên innovation/residual?
- Nếu camera lệch extrinsics (vị trí/góc đặt) hoặc intrinsics (tiêu cự/điểm chính):
  - Phép chiếu pinhole $h(x)$ sẽ bị lệch có hệ thống (systematic bias) so với tọa độ thực tế trên ảnh.
  - Innovation $\gamma = z - h(x)$ sẽ mang giá trị kỳ vọng khác 0 (non-zero mean bias) thay vì dao động quanh 0 (zero-mean Gaussian white noise) như giả định của Kalman Filter.
  - Khi độ lệch calibration lớn, khoảng cách Mahalanobis $d^2 = \gamma^T S^{-1} \gamma$ sẽ tăng vọt vượt quá ngưỡng cổng $\chi^2$, khiến gating chặn toàn bộ các phép đo camera (không thể update camera). Nếu ngưỡng quá lỏng, EKF sẽ bị kéo lệch theo phép đo sai, dẫn đến tăng đột biến sai số covariance $P$ và RMSE 3D vị trí.

### 5. Vì sao `associate_and_update(..., sensor)` cần sensor tường minh ở frame rỗng? Giải thích vì sao lidar quyết định score/init/delete còn camera chỉ EKF update.
- **Cần sensor tường minh ở frame rỗng:**
  - Ngay cả khi `meas_list` rỗng, tracker vẫn phải gọi `manager.manage_tracks(unassigned_tracks, unassigned_meas, sensor)`. Quản lý track cần biết lượt hiện tại là của cảm biến nào để xử lý: nếu là lượt LiDAR, mọi track trong tầm nhìn không nhận được đo sẽ bị tính là miss (bị trừ điểm $1/\text{window}$); nếu là lượt camera thì việc không có đo lường không được phép làm giảm điểm hay xóa track.
- **LiDAR quyết định vòng đời, Camera chỉ EKF update:**
  - LiDAR cung cấp tọa độ 3D đầy đủ và bao quát 360 độ (hoặc vùng quét lớn), có độ tin cậy chiều sâu cao, phù hợp để tạo và duy trì track trong không gian vật lý 3D.
  - Camera trong lab chỉ có trường nhìn hẹp (FRONT), đo 2D pixel không có thông tin độ sâu trực tiếp (đo camera mô phỏng từ nhãn 2D có nhiễu). Camera không thể tự khởi tạo track 3D và không được xóa track để tránh việc xóa nhầm các xe đi ra khỏi góc nhìn camera phía trước.

### 6. Nêu điều kiện xác nhận, giữ confirmed sau miss, và điều kiện xóa track.
- **Điều kiện xác nhận (`confirmed`):**
  - Track có điểm số $\text{score} > \text{confirmed\_threshold}$ (với mỗi hit LiDAR cộng $1/\text{window}$, tối đa bằng 1.0).
- **Giữ `confirmed` sau miss:**
  - Khi một track đã `confirmed` bị miss LiDAR trong FOV, điểm số bị trừ $1/\text{window}$. Tuy nhiên, trạng thái track **vẫn được giữ nguyên là `confirmed`** chứ không bị hạ cấp về `tentative` hay `initialized`. Track confirmed vẫn tiếp tục được duy trì dự báo và ghép nối ở các frame sau.
- **Điều kiện xóa track (`should_delete_track`):**
  - Xóa track nếu thỏa mãn **bất kỳ** điều kiện nào sau đây (toán tử OR):
    1. Độ bất định vị trí ngang vượt ngưỡng: $P[0, 0] > \text{max\_P}$ hoặc $P[1, 1] > \text{max\_P}$ (xóa ngay bất kể điểm số).
    2. Track đã `confirmed` nhưng điểm số tụt quá thấp: $\text{score} < \text{delete\_threshold}$.
    3. Track chưa `confirmed` (`initialized` hoặc `tentative`) có điểm số $\text{score} \le 0$.

## Bonus (không bắt buộc)

- Không

## Khai báo sử dụng AI (bắt buộc)

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): AI Studio
- Dùng cho phần nào (hàm, câu hỏi, debug): Hỗ trợ triển khai công thức toán học và kiểm tra ma trận trong Part E (kalman.py), Part F (association.py), Part G (camera_fusion.py), Part H (track_management.py); gỡ lỗi môi trường OpenMP và bảng mã UTF-8 trên Windows.
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): Chạy toàn bộ 128 tests với pytest student/tests (pass 100%), đối chiếu công thức với HUONG_DAN_KY_THUAT.md, chạy fusion-run-lab với seed 0 đủ 199 frame và kiểm tra invariant trong log, chạy check_submission.py.

## Checklist nộp

- [x] **Part E–H** trong `workspace/` đã implement; `pytest student/tests -q` không còn `failed`/`xfailed`
- [x] Part A–D: không bắt buộc sửa (hoặc ghi chú nếu bạn đã sửa)
- [x] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [x] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [x] Đã điền đủ file này, gồm khai báo AI
- [x] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [x] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`
- [x] Đã push và nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md))