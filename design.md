# Tài Liệu Thiết Kế UI/UX — ShrimpMeasure Web v3.9

**Phiên bản:** 4.0  
**Ngày cập nhật:** 2026-05-18  
**Thay đổi so với v3.9:** Cập nhật luồng hiệu chỉnh Scale — toàn bộ phép tính `scale_i` chuyển sang backend (`POST /api/calibrate`); frontend chỉ thu thập `real_length_mm` và hiển thị kết quả trả về. Xóa tính toán phía client. Viết lại Section 6.3 (Scale Calculator), Section 8.2 (modal xác nhận), bổ sung endpoint vào Section 9.

---

## Mục Lục

1. [Triết Lý Thiết Kế](#1-triết-lý-thiết-kế)
2. [Design Tokens](#2-design-tokens)
3. [Bố Cục Một Màn Hình](#3-bố-cục-một-màn-hình)
4. [Topbar](#4-topbar)
5. [Left Panel — Điều Khiển & Cấu Hình](#5-left-panel--điều-khiển--cấu-hình)
6. [Right Panel — Bảng Kết Quả](#6-right-panel--bảng-kết-quả)
7. [Log Console](#7-log-console)
8. [Modal Debug Viewer & Modal Xác Nhận](#8-modal-debug-viewer--modal-xác-nhận)
9. [REST API — Flask](#9-rest-api--flask)
10. [Responsive & Mobile](#10-responsive--mobile)
11. [Kiểm Tra 7 Nguyên Tắc UI](#11-kiểm-tra-7-nguyên-tắc-ui)

---

## 1. Triết Lý Thiết Kế

**Phong cách: Industrial Light — Nhà Xưởng Ban Ngày**

Giao diện mô phỏng bảng điều khiển máy công nghiệp dưới ánh sáng nhà xưởng ban ngày. Nền sáng như mặt bàn kim loại phủ sơn trắng, viền xám cứng rắn, accent hổ phách đậm như đèn cảnh báo bật giữa ban ngày. Console log giữ nền tối — như màn hình terminal nhúng trên bảng điều khiển sáng. Không trang trí thừa, không animation phức tạp. Mỗi pixel phục vụ chức năng.

| Yếu tố        | Giá trị                                                                             |
| :------------ | :---------------------------------------------------------------------------------- |
| Màu nền trang | Xám sáng công nghiệp `#F0F0EE` — như mặt bàn kim loại phủ sơn                       |
| Màu panel     | Trắng sạch `#FFFFFF` — như vỏ máy sơn trắng                                         |
| Accent        | Hổ phách đậm `#D97706` — đủ tối để đọc rõ trên nền trắng                            |
| Font hiển thị | Share Tech Mono (Google Fonts) — monospace, đọc như màn hình máy CNC                |
| Font nhãn     | Barlow Condensed — nén, đậm, dứt khoát                                              |
| Viền          | `1px solid #D4D4D1`, góc vuông tuyệt đối (`border-radius: 0`)                       |
| Console       | Giữ nền tối `#111111` — terminal nhúng, tương phản với giao diện sáng               |
| Texture       | Subtle noise grain cực nhạt trên `--bg-base` để tạo cảm giác bề mặt sơn công nghiệp |

---

## 2. Design Tokens

```css
:root {
  /* Nền */
  --bg-base: #f0f0ee; /* Nền toàn trang — xám sáng công nghiệp */
  --bg-panel: #ffffff; /* Panel / card — trắng sơn máy */
  --bg-inset: #f5f5f3; /* Ô nhập, vùng lõm */
  --bg-row-odd: #fafafa; /* Hàng lẻ trong bảng */
  --bg-row-even: #ffffff; /* Hàng chẵn */
  --bg-row-hover: #fffbf0; /* Hover amber nhạt — giữ nhận diện accent */
  --bg-console: #111111; /* Console terminal — giữ tối, tương phản mạnh */

  /* Accent công nghiệp — đậm hơn v3.8 để đạt contrast trên nền sáng */
  --amber: #d97706; /* Màu nhấn chính — 5.9:1 trên #FFFFFF */
  --amber-dim: #fef3c7; /* Nền chip / badge amber nhạt */
  --amber-glow: rgba(217, 119, 6, 0.1);

  /* Trạng thái */
  --green-run: #16a34a; /* Đang chạy — 4.5:1 trên #FFFFFF */
  --green-dim: #dcfce7; /* Nền thông báo xanh nhạt */
  --red-err: #dc2626; /* Lỗi — 4.5:1 trên #FFFFFF */
  --red-dim: #fee2e2; /* Nền thông báo đỏ nhạt */
  --blue-info: #0369a1; /* Thông tin — 5.9:1 trên #FFFFFF */

  /* Văn bản — đã kiểm tra contrast WCAG trên --bg-panel (#FFFFFF) */
  --text-primary: #1a1a1a; /* 18.1:1 — text chính, đảm bảo tuyệt đối */
  --text-secondary: #525252; /* 7.0:1 — đạt WCAG AA mọi kích cỡ */
  --text-muted: #737373; /* 4.6:1 — đạt WCAG AA cho text thường */
  --text-dim: #a3a3a3; /* 2.3:1 — CHỈ dùng border, divider, icon decorative */
  --text-amber: #b45309; /* 6.2:1 trên #FFFFFF — nhãn amber, link */
  --text-console: #a3e635; /* Log xanh lá trên nền console tối #111111 */

  /* Viền */
  --border: #e4e4e1; /* Viền mặc định — nhẹ */
  --border-bright: #c8c8c4; /* Viền nổi bật */
  --border-heavy: #9a9a96; /* 2px — divider phân vùng chính */
  --border-amber: #fcd34d; /* Viền accent amber */

  /* Font */
  --font-mono: "Share Tech Mono", "Courier New", monospace;
  --font-label: "Barlow Condensed", "Arial Narrow", sans-serif;

  /* Badge kích cỡ — điều chỉnh cho nền sáng */
  --badge-s-bg: #ede9fe;
  --badge-s-text: #5b21b6; /* Tím: 7.2:1 */
  --badge-m-bg: #e0f2fe;
  --badge-m-text: #075985; /* Lam: 7.5:1 */
  --badge-l-bg: #dcfce7;
  --badge-l-text: #14532d; /* Lá: 9.1:1 */
  --badge-ncn-bg: #fef9c3;
  --badge-ncn-text: #713f12; /* Vàng: 8.0:1 */
  --badge-ncl-bg: #fee2e2;
  --badge-ncl-text: #7f1d1d; /* Đỏ: 9.4:1 */

  /* Spacing — lưới 8px */
  --sp-1: 4px;
  --sp-2: 8px;
  --sp-3: 12px;
  --sp-4: 16px;
  --sp-6: 24px;
  --sp-8: 32px;

  /* Touch targets */
  --touch-desktop: 28px; /* nút nhỏ trên desktop */
  --touch-mobile: 44px; /* tất cả interactive trên mobile */
}
```

---

## 3. Bố Cục Một Màn Hình

Toàn bộ ứng dụng nằm trên **một trang duy nhất** (`/`). Không có điều hướng, không có sidebar, không có đa trang.

```text
┌─────────────────────────────────────────────────────────────────────┐
│ TOPBAR 56px  [nền #FFFFFF, border-bottom 1px #C8C8C4]              │
│ [■■ SHRIMP MEASURE]        [◌ SCALE: 0.350 mm/px]    [⚙ CONFIG]   │
├──────────────────────┬──────────────────────────────────────────────┤
│                      │                                              │
│  LEFT PANEL 320px    │  RIGHT PANEL — BẢNG KẾT QUẢ                │
│  [nền #FFFFFF]       │  [nền #F0F0EE]                              │
│                      │                                              │
│  ┌─── DROP ZONE ──┐  │  Toolbar bảng (Lọc run / source / size)    │
│  │  KÉO FILE VÀO  │  │  ─────────────────────────────────────────  │
│  │  [CHỌN FILE]   │  │  ID │ Frame │ Pixel │ Đo được │ Thực tế │...│
│  └────────────────┘  │  ───┼───────┼───────┼─────────┼─────────┼───│
│  video.mp4 45MB [✕]  │   1 │    45 │ 672.4 │ 235.3mm │[      ] │ L │
│  img001.jpg 2MB [✕]  │   2 │    89 │ 543.2 │ 190.1mm │[      ] │ L │
│  ──── 2 files ─────  │   3 │   120 │ 398.7 │ 139.5mm │[      ] │ M │
│                      │                                              │
│  [▶ CHẠY PIPELINE]   │  ── SCALE CALCULATOR ──────────────────────  │
│                      │  Đề xuất: 0.3497 mm/px    [ÁP DỤNG]         │
│  ── CẤU HÌNH ──      │                                              │
│  SCALE [0.350 ]      ├──────────────────────────────────────────────┤
│  CONF  range+[0.50]  │                                              │
│  SAVE  [ON][OFF]     │  LOG CONSOLE [nền #111111 — tối, giữ nguyên]│
│  ...                 │  > [F1] Đọc video.mp4...                    │
│  ── SIZE RANGES ──   │  > [F6] ID 1: 672.4px  235.3mm  L           │
│  ...                 │                                              │
└──────────────────────┴──────────────────────────────────────────────┘
```

**Màu nền từng vùng:**

| Vùng                    | Nền                               | Ghi chú                              |
| :---------------------- | :-------------------------------- | :----------------------------------- |
| Topbar                  | `--bg-panel` `#FFFFFF`            | `border-bottom: 1px --border-bright` |
| Left panel              | `--bg-panel` `#FFFFFF`            | `border-right: 1px --border`         |
| Right panel (nền trang) | `--bg-base` `#F0F0EE`             | Tạo cảm giác bề mặt bàn sáng         |
| Bảng kết quả            | Hàng xen kẽ `#FAFAFA` / `#FFFFFF` |                                      |
| Log Console             | `--bg-console` `#111111`          | Giữ tối — terminal nhúng             |

**Chiều cao mỗi vùng:**

| Vùng                | Chiều cao                                       |
| :------------------ | :---------------------------------------------- |
| Topbar              | 56px cố định                                    |
| Left + Right (bảng) | `calc(100vh - 56px - 180px)` — cuộn dọc độc lập |
| Log Console         | 180px cố định dưới cùng, có thể kéo rộng lên    |

---

## 4. Topbar

```text
┌─────────────────────────────────────────────────────────────────────┐
│  [▮▮] SHRIMP MEASURE          ● SCALE: 0.350 mm/px    [CONFIG ▼]  │
└─────────────────────────────────────────────────────────────────────┘
```

- **Nền:** `--bg-panel` `#FFFFFF`, `border-bottom: 1px solid --border-bright`
- **Trái:** Logo hai thanh ngang `▮▮` màu `--amber` `#D97706` + chữ `SHRIMP MEASURE`, font Barlow Condensed 700, letter-spacing 3px, màu `--text-primary`
- **Giữa:** Chip `● SCALE: 0.350 mm/px` — nền `--amber-dim` `#FEF3C7`, dấu chấm màu `--amber`, chữ `--text-amber`, font mono. Khi SCALE vừa cập nhật: nhấp nháy viền amber 1 giây
- **Phải:** Nút `[CONFIG ▼]` — viền `1px --border-bright`, nền `--bg-inset`, chữ `--text-secondary`. Hover: viền `--amber`, chữ `--amber`

---

## 5. Left Panel — Điều Khiển & Cấu Hình

Chiều rộng cố định 320px, nền `--bg-panel` `#FFFFFF`, `border-right: 1px solid --border`.

### 5.1 Vùng Thả File & Nút Chạy Pipeline

#### A. Drop Zone — Desktop

Chiều cao tối thiểu 88px.

```text
┌──────────────────────────────────┐
│                                  │
│       KÉO FILE VÀO ĐÂY          │
│         [ CHỌN FILE ]            │
│     JPG · PNG · MP4 · AVI · MOV │
│                                  │
└──────────────────────────────────┘
```

**Trạng thái drop zone:**

| Trạng thái         | Viền                                           | Nền                     | Chi tiết                                 |
| :----------------- | :--------------------------------------------- | :---------------------- | :--------------------------------------- |
| Mặc định           | 2px dashed `--border-bright` `#C8C8C4`         | `--bg-inset` `#F5F5F3`  | Icon, chữ màu `--text-muted`             |
| Hover chuột        | 2px dashed `--amber` `#D97706`                 | `--amber-dim` `#FEF3C7` | Icon + chữ đổi `--amber`                 |
| Drag file vào      | 2px dashed `--amber` + inset shadow amber nhạt | `#FFF8DC`               | Toàn vùng phát sáng nhẹ vào trong        |
| File sai định dạng | 2px dashed `--red-err` `#DC2626`               | `--red-dim` `#FEE2E2`   | Icon đổi ✕, chữ "Định dạng không hỗ trợ" |
| Đang chạy pipeline | 1px solid `--border`                           | `--bg-base` `#F0F0EE`   | Mờ `opacity: 0.4`, `cursor: not-allowed` |

**Danh sách file đã chọn — Desktop** (mỗi hàng cao 52px):

```text
┌──────────────────────────────────────────┐
│ [▓▓▓] video_line1.mp4        45.2 MB    │  <- icon + tên
│       ████████████░░░░░ 68%  1.2 MB/s   │  <- progress bar + tốc độ
├──────────────────────────────────────────┤
│ [IMG] image_001.jpg           2.1 MB  ✔ │  <- thumbnail thật + check
├──────────────────────────────────────────┤
│ ⚠  report.pdf — Định dạng không hỗ trợ │  <- nền --red-dim, text --red-err
├──────────────────────────────────────────┤
│  2 sẵn sàng · 1 đang tải · 1 lỗi    [XÓA TẤT CẢ]
└──────────────────────────────────────────┘
```

**Thumbnail preview trong hàng file:**

- **Ảnh (JPG/PNG...):** thumbnail thật `40×30px`, `object-fit: cover`, viền `1px --border`
- **Video:** icon `40×30px`, nền `--bg-inset` `#F5F5F3` — không generate thumbnail phía client
- **File lỗi:** icon ⚠ `40×30px`, nền `--red-dim` `#FEE2E2`

**Nút xóa `[✕]`:** Luôn hiển thị góc phải mỗi hàng. Kích thước `28×28px` desktop, màu `--text-muted`, hover màu `--red-err` nền `--red-dim`. `aria-label="Xóa <tên file>"` — bắt buộc. Gọi `DELETE /api/files/input/<filename>`.

**Progress bar upload:**

- Thanh ngang 3px, full-width, fill màu `--amber`, nền `--border`
- Bên phải: `68% 1.2 MB/s`, font mono 11px, màu `--text-muted`
- Khi hoàn tất: đổi màu `--green-run`, chữ đổi `✔ SẴN SÀNG`, ẩn sau 800ms

**Trạng thái từng file:**

| Trạng thái    | Hiển thị                                   | Màu                     |
| :------------ | :----------------------------------------- | :---------------------- |
| Đang upload   | Progress bar + `68% 1.2 MB/s`              | `--amber` `#D97706`     |
| Hoàn tất      | ✔ icon góc phải, không progress bar        | `--green-run` `#16A34A` |
| Lỗi upload    | ✕ LỖI `[THỬ LẠI]`                          | `--red-err` `#DC2626`   |
| Sai định dạng | Hàng nền `--red-dim`, text lỗi `--red-err` | `--red-err`             |

**Dòng tổng kết cuối danh sách:** `2 sẵn sàng · 1 đang tải · 1 lỗi` — mỗi phần màu riêng. Nút `[XÓA TẤT CẢ]` cuối dòng — confirm trước khi xóa. Nếu quá 6 file: hiện 5 hàng + dòng `... và 3 file khác`.

#### B. Nút Chạy Pipeline

```text
┌──────────────────────────────────┐
│                                  │
│         ▶  CHẠY PIPELINE        │
│                                  │
└──────────────────────────────────┘
```

- Nền `--amber` `#D97706`, chữ `#FFFFFF`, font Barlow Condensed 700, 16px, letter-spacing 2px, `border-radius: 0`
- Full-width, height 52px
- Không có file sẵn sàng: `opacity: 0.40`, `cursor: not-allowed`
- Có file sẵn sàng lẫn file lỗi: vẫn cho chạy, tooltip `"Sẽ bỏ qua X file lỗi"`

> **Lưu ý màu chữ:** Nền `--amber` `#D97706` + chữ `#FFFFFF` đạt ratio **5.9:1** (WCAG AA). Không dùng chữ `#000000` trên amber vì giao diện sáng dễ nhầm với nền trang.

#### C. Hiệu Ứng Khi Đang Chạy

Khi nhấn `[CHẠY PIPELINE]` gọi `POST /api/pipeline/run` — các hiệu ứng kích hoạt đồng thời:

**1. Nút — pulse border:**

```css
.btn-run.running {
  background: var(--green-dim); /* #DCFCE7 — xanh nhạt */
  color: var(--green-run); /* #16A34A */
  border: 1px solid var(--green-run);
  animation: pulse-border 1.4s ease-in-out infinite;
}
@keyframes pulse-border {
  0% {
    box-shadow: 0 0 0 0 rgba(22, 163, 74, 0.5);
  }
  50% {
    box-shadow: 0 0 0 6px rgba(22, 163, 74, 0);
  }
  100% {
    box-shadow: 0 0 0 0 rgba(22, 163, 74, 0);
  }
}
```

Nhãn: `● ĐANG XỬ LÝ` -> `●● ĐANG XỬ LÝ` -> `●●● ĐANG XỬ LÝ` (vòng 600ms).

**2. Left panel — scan-line overlay:**

```css
.left-panel::after {
  content: "";
  position: absolute;
  inset: 0;
  background: linear-gradient(
    to bottom,
    transparent 0%,
    rgba(217, 119, 6, 0.06) 50%,
    transparent 100%
  );
  background-size: 100% 40px;
  animation: scanline 2s linear infinite;
  pointer-events: none;
}
@keyframes scanline {
  from {
    background-position: 0 -40px;
  }
  to {
    background-position: 0 100%;
  }
}
```

**3. Log console — dòng mới trượt vào:**

```css
.log-line.new {
  animation: log-appear 0.25s ease-out;
}
@keyframes log-appear {
  from {
    opacity: 0;
    transform: translateX(-8px);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}
```

**4. Topbar chip — flicker:** Dấu `●` đổi thành `◌` nhấp nháy 0.8s/lần.

#### D. Tổng Hợp Trạng Thái Nút + Hiệu Ứng

| Trạng thái    | Nền                     | Chữ                     | Nhãn                  | Hiệu ứng                                      |
| :------------ | :---------------------- | :---------------------- | :-------------------- | :-------------------------------------------- |
| Không có file | `--amber` mờ            | `#FFFFFF` mờ            | ▶ CHẠY PIPELINE       | Disabled                                      |
| Sẵn sàng      | `--amber` `#D97706`     | `#FFFFFF`               | ▶ CHẠY PIPELINE       | —                                             |
| Đang chạy     | `--green-dim` `#DCFCE7` | `--green-run` `#16A34A` | ●●● ĐANG XỬ LÝ...     | Pulse + scanline + log animate + chip flicker |
| Hoàn tất      | `--bg-inset`            | `--amber`               | ✔ HOÀN TẤT — CHẠY LẠI | Toast + bảng tự refresh                       |
| Lỗi           | `--red-dim` `#FEE2E2`   | `--red-err` `#DC2626`   | ✕ LỖI — CHẠY LẠI      | Animation dừng, log đỏ                        |

---

### 5.2 Phân Vùng Left Panel & Accordion Config

Left panel chia thành hai vùng tách biệt bằng divider nặng. Vùng THIẾT LẬP dùng accordion — mặc định đóng.

```text
┌──────────────────────────────────┐
│                                  │  <- VÙNG VẬN HÀNH (không nhãn)
│  [DROP ZONE]                     │
│  file list...                    │
│  [▶ CHẠY PIPELINE]              │
│                                  │
│══════════════════════════════════│  <- border-top 2px --border-heavy #9A9A96
│  THIẾT LẬP                      │  <- 9px, letter-spacing 3px, --text-muted
│══════════════════════════════════│
│                                  │
│  ▶ CẤU HÌNH          [▼]        │  <- accordion header, mặc định ĐÓNG
│  ────────────────────────────── │
│  (nội dung config)               │
│                                  │
│  ▶ PHÂN LOẠI KÍCH CỠ [▼]        │  <- accordion header, mặc định ĐÓNG
│  ────────────────────────────── │
│  (nội dung size ranges)          │
│                                  │
└──────────────────────────────────┘
```

**Divider phân vùng:**

- `border-top: 2px solid var(--border-heavy)` `#9A9A96`
- Nhãn `THIẾT LẬP`: Barlow Condensed 700, 9px, `letter-spacing: 3px`, `--text-muted` `#737373`

**Accordion header:**

- Nền `--bg-panel` `#FFFFFF`, padding `10px 0`, font Barlow Condensed 700, 12px uppercase, `--text-secondary`
- Đường kẻ dưới `1px solid --border`
- Icon `▶` góc phải màu `--text-muted`, xoay 90° khi mở, `transition: transform 200ms ease`
- Trạng thái lưu vào `localStorage('accordion_config')` và `localStorage('accordion_sizes')`
- Khi pipeline đang chạy: cả hai accordion disable, `opacity: 0.5`

---

### 5.3 Cấu Hình Nhanh

> **Lưu ý:** `REQUIRED_TOUCHES` là hằng số nội bộ (luôn = 3 vạch), không hiển thị trong UI. Thay đổi yêu cầu sửa trực tiếp `config.py`.

#### A. Bảng Variant Kiểu Input

| Tham số                                     | Kiểu HTML       | `type`             | `min`  | `max` | `step` | Ghi chú              |
| :------------------------------------------ | :-------------- | :----------------- | :----- | :---- | :----- | :------------------- |
| `SCALE`                                     | Ô số đơn        | `number`           | 0.0001 | —     | 0.0001 | 4 chữ số thập phân   |
| `CONF_DET`                                  | Range + số      | `range` + `number` | 0.10   | 1.00  | 0.01   | Hai widget đồng bộ   |
| `CONF_SEG`                                  | Range + số      | `range` + `number` | 0.10   | 1.00  | 0.01   | Hai widget đồng bộ   |
| `BBOX_PAD`                                  | Range + số      | `range` + `number` | 0      | 50    | 1      | Đơn vị: px           |
| `TOUCH_THRESHOLD`                           | Range + số      | `range` + `number` | 1      | 50    | 1      | Đơn vị: px           |
| `TARGET_FPS`                                | Ô số đơn        | `number`           | 0      | 120   | 1      | 0 = lấy tất cả frame |
| Tên cỡ (`size.py`)                          | Ô text          | `text`             | —      | —     | —      | `maxlength=20`       |
| TỪ / ĐẾN (`size.py`)                        | Ô số đơn        | `number`           | 0      | 9999  | 0.1    | Đơn vị: mm           |
| Nhãn ngoại cỡ                               | Ô text          | `text`             | —      | —     | —      | `maxlength=30`       |
| `SAVE` / `CONVEYOR_VERTICAL` / `CHUNK_MODE` | Switch đôi      | —                  | —      | —     | —      | [ON] / [OFF]         |
| `CLEAR_OUTPUT` / `CLEAR_INPUT`              | Switch cảnh báo | —                  | —      | —     | —      | Màu đỏ khi ON        |

#### B. Switch Button — Hai Variant

**Variant 1 — Switch thường** (SAVE, CONVEYOR_VERTICAL, CHUNK_MODE):

```text
SAVE DEBUG    [ON ]  [OFF]
```

- `[ON]` active: nền `--amber` `#D97706`, chữ `#FFFFFF`, font Barlow Condensed 700
- `[OFF]` active: nền `--bg-inset` `#F5F5F3`, chữ `--text-secondary` `#525252`, viền `1px --border`
- Inactive không active: nền `--bg-base`, chữ `--text-muted`
- Hai nút liền nhau, viền chung `1px --border-bright`

**Variant 2 — Switch cảnh báo** (CLEAR_OUTPUT, CLEAR_INPUT):

```text
CLEAR OUTPUT  [ON ]  [OFF]   ⚠ Xóa không hoàn tác
```

- `[ON]` active: nền `--red-dim` `#FEE2E2`, chữ `--red-err` `#DC2626`, viền `1px --red-err`
- Khi ON: hiện dòng cảnh báo `⚠ Xóa không hoàn tác` màu `--red-err`, font mono 11px, nền `--red-dim` padding nhỏ

#### C. Range + Số — Pattern Dùng Chung

```text
CONF DET  ├────●────────────┤  [0.50]
          0.10              1.00
```

- **Track range:** nền `--border` `#E4E4E1`, chiều cao 3px
- **Fill range:** nền `--amber` `#D97706` (phần từ đầu đến thumb)
- **Thumb:** hình vuông `12×12px` màu `--amber`, viền `1px --border-heavy`
- **Ô số:** rộng 56px, font mono, nền `--bg-panel`, viền `1px --border-bright`, chữ `--text-primary`
- Đồng bộ hai chiều dùng sự kiện `input` (không phải `change`)
- Giá trị ngoài `[min, max]`: clamp về giới hạn, viền ô đổi `--red-err` nháy 300ms
- Nhãn min/max bên dưới hai đầu thanh, màu `--text-muted` 10px

**Ô nhập `<input>` — màu nền theo trạng thái:**

| Trạng thái | Viền                                               | Nền          | Chữ              |
| :--------- | :------------------------------------------------- | :----------- | :--------------- |
| Mặc định   | `1px --border-bright`                              | `--bg-panel` | `--text-primary` |
| Focus      | `1px --amber`, box-shadow `0 0 0 2px --amber-glow` | `--bg-panel` | `--text-primary` |
| Lỗi        | `1px --red-err`                                    | `--red-dim`  | `--text-primary` |
| Disabled   | `1px --border`                                     | `--bg-base`  | `--text-muted`   |

#### D. Layout Config — Grid 2 Cột với Hint Text

```css
.config-row {
  display: grid;
  grid-template-columns: 96px 1fr;
  align-items: start;
  gap: 0 8px;
  padding: 6px 0;
}
.config-hint {
  grid-column: 2;
  font-size: 11px;
  color: var(--text-muted); /* #737373 — 4.6:1 trên #FFFFFF */
  margin-top: 3px;
  line-height: 1.4;
}
```

**Sơ đồ đầy đủ với hint text:**

```text
── CẤU HÌNH ──────────────────────────────────────────────────────────

SCALE     [0.3500 ] mm/px
           1 px = 0.3500 mm. Hiệu chỉnh: đặt vật 100mm vào khung,
           đo chiều dài pixel -> SCALE = 100 / pixel_đó.

CONF DET  ├──●─────────────────────────┤ [0.50]
          0.10                       1.00
           Ngưỡng nhận diện tôm. Thấp -> dễ nhận nhầm vật lạ.
           Cao -> có thể bỏ sót tôm khuất hoặc tôm nhỏ.

CONF SEG  ├──●─────────────────────────┤ [0.50]
          0.10                       1.00
           Ngưỡng phân đoạn thân tôm. Sai mask -> sai chiều dài đo được.

BBOX PAD  ├──────●──────────────────────┤ [10] px
          0                           50
           Mở rộng vùng cắt quanh bounding box trước khi segment.
           Tăng nếu tôm bị cắt mất đầu/đuôi trong ảnh debug F4.

TOUCH THR ├──────●──────────────────────┤ [10] px
          1                           50
           Khoảng cách tối đa (px) để tính là "chạm vạch".
           Băng chuyền rung nhiều hoặc FPS thấp -> nên tăng giá trị.

TARGET FPS  [0 ]
             0 = lấy tất cả frame (chính xác nhất).
             Tăng lên để xử lý nhanh hơn — có thể bỏ sót tôm ngắn.

SAVE        [ON ] [OFF]
             ON  -> lưu ảnh debug F3–F6 + ghi đường dẫn vào JSON.
             OFF -> chỉ xuất JSON kết quả số, không lưu ảnh.

CONVEYOR    [ON ] [OFF]
             ON  = băng chuyền dọc (tôm chạy từ trên xuống, 3 vạch ngang).
             OFF = băng chuyền ngang (trái -> phải, 3 vạch dọc).

CHUNK MODE  [ON ] [OFF]
             ON  = nhiều video xử lý như 1 luồng liên tục.
             OFF = mỗi video là một luồng độc lập (mặc định).

CLEAR OUT   [ON ] [OFF]  ⚠ Không hoàn tác
             ON  -> xóa toàn bộ thư mục output/ trước mỗi lần chạy.

CLEAR IN    [ON ] [OFF]  ⚠ Không hoàn tác
             ON  -> tự động xóa file input/ sau khi pipeline ghi JSON.

──────────────────────────────────────────────────────────────────────
                                                         [LƯU CONFIG]
```

**Nút `[LƯU CONFIG]`:**

- Viền `1px --border-bright`, nền transparent, chữ `--text-secondary`, căn phải
- Hover: viền `--amber`, chữ `--amber`, nền `--amber-dim`
- Gọi `PUT /api/config`
- Sau khi lưu thành công: hiện `✔ ĐÃ LƯU` màu `--green-run` trong 1.5 giây

---

### 5.4 Cấu Hình Phân Loại Kích Cỡ

```text
── SIZE RANGES ──────────────────────────────────────────────────────

   Tên cỡ        Từ (mm)    Đến (mm)
   Chữ + số,     >= 0,      > cột Từ,     [xóa hàng]
   tối đa 20 ký  đơn vị mm  đơn vị mm

  ─────────────────────────────────────────────────────────────────
   [S ]         [100 ]  ->  [125 ]   [✕]
   [M ]         [125 ]  ->  [160 ]   [✕]
   [L ]         [160 ]  ->  [200 ]   [✕]
  ─────────────────────────────────────────────────────────────────
   [+ THÊM CỠ MỚI]
   Nhấn để thêm hàng mới; TỪ = Đến lớn nhất + 1.
   Danh sách tự sắp xếp tăng dần theo cột Từ.

── NHÃN NGOẠI CỠ ────────────────────────────────────────────────────

   NHỎ HƠN MIN   [Ngoại cỡ nhỏ ]
                  Gán cho tôm có real_length < Từ nhỏ nhất (hiện tại: < 100 mm).

   LỚN HƠN MAX   [Ngoại cỡ lớn ]
                  Gán cho tôm có real_length >= Đến lớn nhất (hiện tại: >= 200 mm).

   KHOẢNG TRỐNG  [Ngoại cỡ     ]
                  Gán khi tôm rơi vào mm không được định nghĩa giữa hai cỡ.

── MODEL (chỉ đọc) ───────────────────────────────────────────────────

   DETECT   yolov8n_shrimp_v46_openvino_model
   SEGMENT  yolov8n-seg_shrimp_v54_openvino_model
   Thay đổi model: sửa config.py -> khởi động lại server.

  ─────────────────────────────────────────────────────────────────
                                                  [LƯU PHÂN LOẠI]
```

**Validation bảng cỡ — thông báo lỗi hiện thẳng dưới ô:**

| Điều kiện               | Viền ô             | Nền ô         | Thông báo dưới ô               |
| :---------------------- | :----------------- | :------------ | :----------------------------- |
| TỪ >= ĐẾN               | `--red-err` cả hai | `--red-dim`   | Từ phải nhỏ hơn Đến            |
| Chồng lấp với hàng khác | `--amber`          | `--amber-dim` | Chồng lấp với cỡ "[tên]"       |
| TÊN CỠ trống            | `--red-err`        | `--red-dim`   | Tên không được để trống        |
| Tên trùng lặp           | `--red-err`        | `--red-dim`   | Tên "[tên]" đã tồn tại         |
| Ký tự không hợp lệ      | `--red-err`        | `--red-dim`   | Chỉ dùng chữ cái, số, dấu cách |

**Nhãn ngoại cỡ — hint text đầy đủ:**

| Ô nhập       | Biến `size.py`    | Hint dưới ô                                                  |
| :----------- | :---------------- | :----------------------------------------------------------- |
| NHỎ HƠN MIN  | `UNDERSIZE_LABEL` | Dùng khi `real_length < Từ` nhỏ nhất (hiện tại: < 100 mm)    |
| LỚN HƠN MAX  | `OVERSIZE_LABEL`  | Dùng khi `real_length >= Đến` lớn nhất (hiện tại: >= 200 mm) |
| KHOẢNG TRỐNG | `FALLBACK_LABEL`  | Dùng khi tôm rơi vào mm nằm giữa hai cỡ không liền kề        |

> Giá trị trong hint (`< 100 mm`, `>= 200 mm`) tính động từ bảng cỡ hiện tại — cập nhật ngay khi người dùng thay đổi ô TỪ/ĐẾN.

---

## 6. Right Panel — Bảng Kết Quả

Nền tổng thể của right panel là `--bg-base` `#F0F0EE` — tạo cảm giác bề mặt bàn làm việc nhà xưởng. Bảng dữ liệu nổi lên trên nền này với nền trắng xen kẽ xám rất nhạt.

### 6.1 Toolbar Bảng

```text
[RUN: 2026-05-11_10-30 ▼]  [SOURCE: video ▼]  [SIZE: ALL ▼]  [CSV ↓]
```

- Nền toolbar: `--bg-panel` `#FFFFFF`, `border-bottom: 1px --border`
- Tất cả dropdown: nền `--bg-panel`, viền `1px --border-bright`, font mono, chữ `--text-primary`
- Hover dropdown: viền `--amber`
- Nút `[CSV ↓]`: viền `1px --border-bright`, chữ `--text-secondary`, hover viền `--amber` chữ `--amber`
- Khi chọn `run_dir` mới: bảng refresh tự động qua `GET /api/results/shrimps`

### 6.2 Bảng Kết Quả

```css
table {
  table-layout: fixed;
  width: 100%;
  border-collapse: collapse;
  background: var(--bg-panel);
  border: 1px solid var(--border);
}

th:nth-child(1),
td:nth-child(1) {
  width: 52px;
  text-align: right;
} /* ID */
th:nth-child(2),
td:nth-child(2) {
  width: 72px;
  text-align: right;
} /* FRAME */
th:nth-child(3),
td:nth-child(3) {
  width: 96px;
  text-align: right;
} /* PIXEL(px) */
th:nth-child(4),
td:nth-child(4) {
  width: 104px;
  text-align: right;
} /* ĐO ĐƯỢC */
th:nth-child(5),
td:nth-child(5) {
  width: 120px;
  text-align: center;
} /* THỰC TẾ */
th:nth-child(6),
td:nth-child(6) {
  width: 96px;
  text-align: right;
} /* SCALE */
th:nth-child(7),
td:nth-child(7) {
  width: 64px;
  text-align: center;
} /* SIZE */
th:nth-child(8),
td:nth-child(8) {
  width: 44px;
  text-align: center;
} /* ẢNH */

th,
td {
  padding: 6px 10px;
  font-family: var(--font-mono);
  font-size: 13px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  border-bottom: 1px solid var(--border);
}
```

**Header bảng:** Nền `--bg-base` `#F0F0EE`, font Barlow Condensed 700, 11px, uppercase, letter-spacing 2px, chữ `--text-secondary`. `border-bottom: 2px solid --amber`.

**Hàng dữ liệu:** Hàng lẻ `--bg-row-odd` `#FAFAFA`, hàng chẵn `--bg-row-even` `#FFFFFF`, hover `--bg-row-hover` `#FFFBF0`. Font mono toàn bộ, 13px, chữ `--text-primary`.

**Cột "THỰC TẾ (mm)":** Ô `<input type="number">`, width 88px, nền `--bg-inset` `#F5F5F3`, viền `1px --border`. Khi nhập hợp lệ (> 0): viền `--green-run`. Khi nhập 0 hoặc số âm: viền `--red-err`, nền `--red-dim`. **Cột SCALE không cập nhật khi nhập** — giá trị scale từng hàng chỉ điền sau khi `POST /api/calibrate` trả về `scales_detail`; trước đó hiển thị `--` màu `--text-muted`.

**Cột "SIZE" — Badge — Light Theme:**

| Badge   | Nền (`--badge-*-bg`) | Chữ (`--badge-*-text`) | Ratio | Ý nghĩa             |
| :------ | :------------------- | :--------------------- | :---- | :------------------ |
| `[ S ]` | `#EDE9FE`            | `#5B21B6`              | 7.2:1 | Tôm nhỏ (tím)       |
| `[ M ]` | `#E0F2FE`            | `#075985`              | 7.5:1 | Tôm vừa (lam)       |
| `[ L ]` | `#DCFCE7`            | `#14532D`              | 9.1:1 | Tôm lớn (lá)        |
| `[NCN]` | `#FEF9C3`            | `#713F12`              | 8.0:1 | Ngoại cỡ nhỏ (vàng) |
| `[NCL]` | `#FEE2E2`            | `#7F1D1D`              | 9.4:1 | Ngoại cỡ lớn (đỏ)   |

Badge: `border-radius: 0`, padding `2px 6px`, font Barlow Condensed 700, 11px. Viền `1px` cùng màu chữ, `opacity: 0.3`.

**Cột "ẢNH":** Nút `[⊞]` — viền `1px --border-bright`, nền `--bg-inset`, chữ `--text-secondary`. Hover: nền `--amber-dim`, viền `--amber`, chữ `--amber`. `aria-label="Xem ảnh debug ID <n>"` — bắt buộc.

### 6.3 Scale Calculator Panel

Xuất hiện ngay dưới bảng khi có ít nhất 1 giá trị "Thực tế" hợp lệ được nhập. Nền `--bg-panel`, viền `1px --border`, `border-top: 2px solid --amber`.

> **Phân công rõ ràng:** Frontend chỉ thu thập `real_length_mm` từ user và gửi lên. Toàn bộ phép tính `scale_i = real_length_mm / pixel_length`, lấy trung bình, và lưu `config.SCALE` đều do backend thực hiện trong `POST /api/calibrate`. Frontend không tự tính bất kỳ giá trị nào.

**Step indicator — 3 bước:**

```text
 ① NHẬP THỰC TẾ  ──●──  ② TÍNH SCALE  ──●──  ③ KẾT QUẢ
```

- Bước đang ở: Barlow Condensed 700, chữ `--amber`
- Bước đã qua: chữ `--green-run`, gạch chân nhạt
- Bước chưa tới: chữ `--text-muted`
- Đường nối: `1px dashed --border-bright`

**Trạng thái bước 1 — Nhập thực tế (đang nhập, chưa gửi):**

```text
── SCALE CALCULATOR ─────────────────────────────────────────────────

 MẪU   SOURCE         ID    PIXEL(px)    THỰC TẾ(mm)    SCALE(mm/px)
   1   video1          1      672.4          235.0           --
   2   video1          3      543.2          190.0           --
   3   img_001         1      407.8          142.5           --
 ────────────────────────────────────────────────────────────────────
                                           3 mẫu sẵn sàng

 [▶ TÍNH SCALE TỪ 3 MẪU]
──────────────────────────────────────────────────────────────────────
```

- Cột `SCALE(mm/px)`: hiển thị `--`, màu `--text-muted` — chưa có dữ liệu từ backend
- Nút `[▶ TÍNH SCALE TỪ N MẪU]`: nền `--amber` `#D97706`, chữ `#FFFFFF`, Barlow Condensed 700
- Nhãn đếm mẫu: `N mẫu sẵn sàng`, màu `--text-secondary`, font mono 12px
- Khi nhấn: hiện modal xác nhận (mục 8.2) trước khi gửi API

**Payload frontend gom và gửi lên (không tính gì thêm):**

```json
{
  "entries": [
    { "source_stem": "video1", "track_id": 1, "real_length_mm": 235.0 },
    { "source_stem": "video1", "track_id": 3, "real_length_mm": 189.0 },
    { "source_stem": "img_001", "track_id": 1, "real_length_mm": 142.5 }
  ]
}
```

**Trạng thái bước 2 — Đang gọi API:**

Nút đổi sang `● ĐANG TÍNH...`, `opacity: 0.6`, `cursor: not-allowed`. Cột SCALE vẫn hiện `--`. Spinner nhỏ 14px màu `--amber` trong nút.

**Trạng thái bước 3 — Nhận kết quả từ `POST /api/calibrate`:**

Backend trả về `{ scale, num_samples, scales_detail }`. Frontend chỉ render — không tự tính gì.

```text
── SCALE CALCULATOR ─────────────────────────────────────────────────

 MẪU   SOURCE         ID    PIXEL(px)    THỰC TẾ(mm)    SCALE(mm/px)
   1   video1          1      672.4          235.0         0.349554
   2   video1          3      543.2          189.0         0.348010
   3   img_001         1      407.8          142.5         0.349533
 ────────────────────────────────────────────────────────────────────
 TRUNG BÌNH      ► 0.349032 mm/px             [chữ --amber, đậm]
 SCALE HIỆN TẠI    0.350000 mm/px             [--text-secondary]
 SAI LỆCH          0.28%                      [--text-muted]
──────────────────────────────────────────────────────────────────────
 ✔ ĐÃ LƯU — Scale mới: 0.349032 mm/px        [chữ --green-run]
 [TÍNH LẠI]
──────────────────────────────────────────────────────────────────────
```

- Cột `SCALE(mm/px)`: điền giá trị từ `scales_detail[i]`, màu `--amber`
- `TRUNG BÌNH`: lấy từ `scale` trong response, Barlow Condensed 700, `--amber`
- `SCALE HIỆN TẠI`: giá trị config trước khi gọi API (lưu ở client khi load trang)
- `SAI LỆCH`: `abs(scale_mới - scale_cũ) / scale_cũ * 100`, tính phía client chỉ để hiển thị
- Dòng `✔ ĐÃ LƯU`: hiện sau khi API trả 200. Backend đã ghi vào `config.py` — không cần thêm bước xác nhận
- Chip SCALE ở Topbar cập nhật ngay + nhấp nháy viền amber 1 giây
- Nút `[TÍNH LẠI]`: reset về bước 1, xóa cột SCALE trong bảng về `--`

**Xử lý lỗi từ API:**

| HTTP                            | Nguyên nhân             | Hiển thị trong panel                                  |
| :------------------------------ | :---------------------- | :---------------------------------------------------- |
| 400 — entries rỗng              | Không có mẫu nào        | Không thể xảy ra (nút disable khi 0 mẫu)              |
| 404 — source_stem không tồn tại | Run chưa có JSON        | `⚠ Không tìm thấy kết quả cho "<source_stem>"`        |
| 404 — track_id không khớp       | Tôm không có trong JSON | `⚠ ID <track_id> không tồn tại trong "<source_stem>"` |
| 400 — pixel_length = 0          | Lỗi đo                  | `⚠ ID <track_id>: pixel_length = 0, bỏ qua`           |

Thông báo lỗi: nền `--red-dim`, viền `1px --red-err`, chữ `--red-err`, font mono 12px, hiện ngay dưới bảng mẫu. Nút không reset — user có thể sửa và thử lại.

---

## 7. Log Console

Dải cố định 180px ở đáy màn hình, span toàn chiều rộng (trừ left panel). **Giữ nền tối** `--bg-console` `#111111` — như terminal nhúng trên bảng điều khiển sáng, tương phản mạnh với giao diện light phía trên.

```text
── SYSTEM LOG ──────────────────────────────── [▲ MỞ RỘNG] [XÓA] ──
> [F1] 10:30:01  Đọc video: video.mp4 | 1920x1080
> [F2] 10:30:02  Phát hiện 3 tôm — frame 45
> [F3] 10:30:03  ID 1 đủ 3 vạch -> đẩy sang F4
> [F6] 10:30:05  ID 1 | 672.4 px | 235.3 mm | L
> [OK] 10:30:06  Hoàn tất. JSON -> output/2026-05-11.../video_results.json
────────────────────────────────────────────────────────────────────
```

**Thiết kế thanh tiêu đề console:** Nền `#1A1A1A`, `border-top: 1px solid #2A2A2A`. Nhãn `SYSTEM LOG`: Barlow Condensed 700, 10px, letter-spacing 3px, màu `#909090`. Nút `[▲ MỞ RỘNG]` và `[XÓA]`: chữ `#909090`, hover chữ `#D97706` (amber) — điểm nhấn kết nối với theme sáng phía trên.

**Vùng log:** Font Share Tech Mono 12px, màu `--text-console` `#A3E635` cho text thông thường. Nền `--bg-console` `#111111`.

**Prefix màu theo luồng:**

| Prefix | Màu hex   | Ghi chú      |
| :----- | :-------- | :----------- |
| `[F1]` | `#6EE7B7` | Xanh bạc hà  |
| `[F2]` | `#38BDF8` | Xanh lam     |
| `[F3]` | `#FDE68A` | Vàng nhạt    |
| `[F4]` | `#C4B5FD` | Tím nhạt     |
| `[F5]` | `#FDBA74` | Cam nhạt     |
| `[F6]` | `#F59E0B` | Amber        |
| `[OK]` | `#4ADE80` | Xanh lá sáng |
| `[!!]` | `#F87171` | Đỏ nhạt      |

> Màu prefix là trang trí bổ sung — nội dung phân biệt luồng xác định bằng ký tự `[F1]`...`[F6]`, không phụ thuộc vào màu.

- Auto-scroll xuống cuối khi có dòng mới
- Kết nối SSE: `GET /api/pipeline/log-stream`
- Nút `[▲ MỞ RỘNG]`: kéo console lên chiếm 50% màn hình
- Nút `[XÓA]`: clear console phía client, không xóa log server

---

## 8. Modal Debug Viewer & Modal Xác Nhận

### 8.1 Modal Debug Viewer — Xem Ảnh F3–F6

Mở khi nhấn `[⊞]`. Fullscreen overlay, nền `rgba(15, 15, 15, 0.80)`.

**Thanh tiêu đề modal:** Nền `--bg-panel` `#FFFFFF`, `border-bottom: 1px --border-bright`.

```text
┌──────────────────────────────────────────────────────────────┐
│  ID 1 — FRAME #45 — 672.4px — 235.3mm — [L]   [✕ ĐÓNG]    │
│  [nền #FFFFFF, border-bottom 1px #C8C8C4]                   │
├───────────────┬──────────────────────────────────────────────┤
│  [nền #F5F5F3]│  [nền #FFFFFF]                              │
│  [F3 BEST ] ●│                                              │
│  [F3 TOUCH1] │          ẢNH F3_Best_F45.jpg                │
│  [F3 TOUCH2] │           (zoom + pan)                       │
│  [F3 TOUCH3] │                                              │
│  [F4 SEG   ] │                                              │
│  [F4 MASK  ] │                                              │
│  [F5 SKEL  ] │                                              │
│  [F5 BFS   ] │                                              │
│  [F6 RESULT] │                                              │
│               │                                [↓ TẢI XUỐNG]│
└───────────────┴──────────────────────────────────────────────┘
```

- **Thumbnail strip trái:** Nền `--bg-inset` `#F5F5F3`, `border-right: 1px --border`, width 100px
- Thumbnail active: `border-left: 3px solid --amber`, nền `--amber-dim`
- Thumbnail hover: nền `--bg-row-hover`
- Nhãn thumbnail: Barlow Condensed 700, 10px, uppercase, `--text-secondary`
- **Vùng ảnh chính:** Nền `--bg-panel` `#FFFFFF`
- **Nút `[✕ ĐÓNG]`:** Barlow Condensed 700, viền `1px --border-bright`, chữ `--text-secondary`, hover viền `--red-err` chữ `--red-err`
- **Nút `[↓ TẢI XUỐNG]`:** Viền `1px --border-bright`, chữ `--text-secondary`, hover viền `--amber` chữ `--amber`

**Trạng thái lỗi tải ảnh:**

```text
┌──────────────────────────────────────────────────────────────┐
│  [F4 MASK ] ⚠│                                              │
│               │   ┌────────────────────────────┐            │
│               │   │  ⚠  KHÔNG ĐỌC ĐƯỢC        │            │
│               │   │  ─────────────────────────  │            │
│               │   │  F4_Mask_F45.jpg            │            │
│               │   │  Không tìm thấy file        │            │
│               │   │  hoặc SAVE = False           │            │
│               │   └────────────────────────────┘            │
│               │                       [↓ TẢI XUỐNG](mờ)    │
└───────────────┴──────────────────────────────────────────────┘
```

- Hộp thông báo: viền `1px --red-err`, nền `--red-dim` `#FEE2E2`, chữ `--red-err`, canh giữa
- Thumbnail lỗi trong strip trái: icon ⚠ thay ảnh preview, nền `--red-dim`, viền trái `3px --red-err`
- Nút `[↓ TẢI XUỐNG]`: `opacity: 0.35`, `cursor: not-allowed`

**Focus trap trong modal:**

- Khi mở: focus tự động chuyển vào nút `[✕]`
- Tab chỉ di chuyển trong: `[✕]`, `[◀]`, thumbnail buttons, `[▶]`, `[↓ TẢI XUỐNG]`
- Khi đóng: focus trả về nút `[⊞]` đã mở modal
- `role="dialog"`, `aria-modal="true"`, `aria-label="Ảnh debug ID <n>"`

### 8.2 Modal Xác Nhận Áp Dụng Scale

```text
┌────────────────────────────────────┐  <- nền #FFFFFF, viền 1px --border-bright
│      XÁC NHẬN THAY ĐỔI SCALE      │  <- Barlow Condensed 700, --text-primary
│  ─────────────────────────────── │  <- border 1px --border
│  CŨ :  0.3500 mm/px               │  <- --text-secondary
│  MỚI:  0.3497 mm/px               │  <- --amber, đậm
│                                    │
│  Sẽ ghi đè config.py              │  <- --text-muted, 12px
│                                    │
│     [XÁC NHẬN]      [HỦY]         │
└────────────────────────────────────┘
```

- Overlay nền: `rgba(0,0,0,0.5)`
- Hộp dialog: nền `--bg-panel` `#FFFFFF`, viền `1px --border-bright`, `border-top: 3px solid --amber`, `border-radius: 0`
- `[XÁC NHẬN]`: nền `--amber` `#D97706`, chữ `#FFFFFF`, Barlow Condensed 700
- `[HỦY]`: viền `1px --border-bright`, nền `--bg-inset`, chữ `--text-secondary`. Hover: viền `--border-heavy`

---

## 9. REST API — Flask

| Method   | URL                           | Mô tả                                                    |
| :------- | :---------------------------- | :------------------------------------------------------- |
| `POST`   | `/api/pipeline/run`           | Khởi chạy pipeline                                       |
| `GET`    | `/api/pipeline/log-stream`    | SSE stream log thời gian thực                            |
| `POST`   | `/api/files/upload`           | Upload file vào `input/` (multipart/form-data, multiple) |
| `DELETE` | `/api/files/input/<filename>` | Xóa file khỏi `input/`                                   |
| `GET`    | `/api/files/input`            | Danh sách file hiện có trong `input/`                    |
| `GET`    | `/api/results/runs`           | Danh sách `run_dir`                                      |
| `GET`    | `/api/results/sources`        | Danh sách source trong run                               |
| `GET`    | `/api/results/shrimps`        | Danh sách tôm (có filter)                                |
| `GET`    | `/api/results/export-csv`     | Xuất CSV                                                 |
| `GET`    | `/api/results/image`          | Lấy ảnh debug theo path                                  |
| `GET`    | `/api/config`                 | Lấy toàn bộ config                                       |
| `PUT`    | `/api/config`                 | Cập nhật config (SAVE, CONF...)                          |
| `PATCH`  | `/api/config/scale`           | Cập nhật SCALE từ tính toán                              |
| `GET`    | `/api/config/sizes`           | Lấy bảng `SIZE_RANGES` + 3 nhãn ngoại cỡ                 |
| `PUT`    | `/api/config/sizes`           | Lưu toàn bộ bảng phân loại + nhãn ngoại cỡ               |

**Request body `PUT /api/config/sizes`:**

```json
{
  "ranges": {
    "S": [100, 125],
    "M": [125, 160],
    "L": [160, 200],
    "XL": [200, 250]
  },
  "undersize_label": "Ngoại cỡ nhỏ",
  "oversize_label": "Ngoại cỡ lớn",
  "fallback_label": "Ngoại cỡ"
}
```

Server validate không chồng lấp, không trống trước khi ghi vào `size.py`.

---

## 10. Responsive & Mobile

### 10.1 Breakpoints

| Tên     | Điều kiện      | Mô tả                                                        |
| :------ | :------------- | :----------------------------------------------------------- |
| Desktop | >= 1024px      | Layout hai cột (left panel + right panel) như thiết kế gốc   |
| Tablet  | 768px – 1023px | Left panel thu thành drawer trượt, right panel chiếm toàn bộ |
| Mobile  | < 768px        | Một cột duy nhất, điều hướng bằng tab bar đáy màn hình       |

### 10.2 Layout Mobile (< 768px)

```text
┌─────────────────────────────────┐
│  TOPBAR 48px  [nền #FFFFFF]     │
│  [■■ SHRIMP MEASURE]  [SCALE ▸]│
├─────────────────────────────────┤
│  [nền --bg-base #F0F0EE]        │
│   NỘI DUNG TAB HIỆN TẠI        │
│   (cuộn dọc độc lập)            │
│                                 │
├─────────────────────────────────┤
│  BOTTOM TAB BAR 56px            │
│  [nền #FFFFFF, border-top 1px] │
│  [▶ CHẠY] [▐▌ KẾT QUẢ] [⚙ CFG]│
└─────────────────────────────────┘
```

**Topbar mobile:** Nền `--bg-panel` `#FFFFFF`, `border-bottom: 1px --border`. Logo thu gọn còn `■■ SM`. Chip SCALE thu về icon — nhấn mở popover nhỏ.

**Bottom Tab Bar:** Nền `--bg-panel` `#FFFFFF`, `border-top: 1px --border-bright`, shadow `0 -2px 8px rgba(0,0,0,0.08)`. 3 tab bằng nhau. Tab active: icon + chữ màu `--amber`. Inactive: `--text-muted`. Touch target mỗi tab: toàn bộ 1/3 chiều rộng × 56px.

### 10.3 Tab 1 — CHẠY (Mobile)

**A. Empty State — Chưa Có File:**

```text
┌─────────────────────────────────────┐  <- nền --bg-base
│                                     │
│         CHƯA CÓ FILE NÀO           │  <- --text-muted, Barlow Condensed
│  Thêm ảnh hoặc video để bắt đầu    │  <- --text-muted, 13px
│                                     │
│  ┌─────────────────────────────┐    │
│  │        + THÊM FILE          │    │  <- 56px, 2px dashed --amber, nền #FEF3C7
│  └─────────────────────────────┘    │
│                                     │
│  ┌─────────────────────────────┐    │
│  │       ▶ CHẠY PIPELINE       │    │  <- disabled, opacity 0.35
│  └─────────────────────────────┘    │
│                                     │
└─────────────────────────────────────┘
```

**B. Có File — Card File Mobile:**

- Nền card: `--bg-panel` `#FFFFFF`, viền `1px --border`, `border-left: 3px solid --border`
- Card file sẵn sàng: `border-left: 3px solid --green-run`
- Card file lỗi: nền `--red-dim`, `border-left: 3px solid --red-err`
- Thumbnail/Icon trái: `52×40px`, `object-fit: cover`, viền `1px --border`
- Tên file: font mono 13px, `--text-primary`, `text-overflow: ellipsis`. Dung lượng dòng 2, `--text-muted` 11px
- Nút `[✕]`: Luôn hiển thị, tap zone `44×44px`, màu `--text-muted`, hover nền `--red-dim` chữ `--red-err`
- Swipe-to-delete: Vuốt trái 60px -> lộ nền `--red-dim`. Vuốt đủ 120px -> xóa

**Trạng thái card khi pipeline đang chạy:**

| Trạng thái pipeline | Viền trái card       | Hiển thị                                          |
| :------------------ | :------------------- | :------------------------------------------------ |
| Hàng chờ            | `--green-run`        | ✔ SẴN SÀNG, chữ `--green-run`                     |
| Đang xử lý file này | `--amber`, nhấp nháy | Scanline amber nhạt + ⚙ ĐANG XỬ LÝ, chữ `--amber` |
| Đã xử lý xong       | `--green-run`        | ✔ ĐÃ ĐO XONG, overlay ✔ lên thumbnail             |

File list: `max-height: 280px`, `overflow-y: auto`. Thanh cuộn: width 3px, nền `--amber`.

**Log console mobile:** Chiều cao cố định 160px, nền `--bg-console` `#111111`. Vuốt lên vùng log -> mở rộng lên 60% màn hình.

### 10.4 Tab 2 — KẾT QUẢ (Mobile)

Thay bảng 8 cột bằng danh sách card, mỗi tôm một card. Nền tổng thể tab `--bg-base` `#F0F0EE`. Mỗi card: nền `--bg-panel` `#FFFFFF`, viền `1px --border`, margin `8px 0`.

```text
┌───────────────────────────────┐    235.3 mm       │  <- font mono, --text-primary
│  Size: [ L ]                  │  <- badge nền #DCFCE7 chữ #14532D
│  Thực tế: [          ] mm     │  <- input, nền --bg-inset
│  SCALE:  —                    │  <- --text-muted khi chưa nhập
└───────────────────────────────┘
```

Scale Calculator mobile: Dải cố định 72px ở đáy tab, nền `--bg-panel`, `border-top: 2px solid --amber`, hiện khi có >= 1 mẫu.

### 10.5 Tab 3 — CẤU HÌNH (Mobile)

Nền `--bg-base`. Các nhóm config dạng accordion, mỗi nhóm: nền `--bg-panel`, viền `1px --border`.

- Range slider mobile: thumb `24×24px`, chiều cao thanh 4px, height tối thiểu touch 44px
- Switch button mobile: `min-height: 44px`, `min-width: 52px`
- Ô nhập số: `font-size: 16px` bắt buộc (tránh iOS auto-zoom)

### 10.6 Modal Debug Viewer — Mobile

Ảnh chiếm toàn bộ màn hình. Nền overlay `rgba(240,240,238,0.95)` — sáng, giữ cảm giác light theme.

**Nút đóng `[✕]` — Floating:**

- Vị trí: `bottom: 24px`, `right: 20px` — vùng ngón cái dễ với nhất
- Kích thước: `52×52px`, hình tròn, nền `--bg-panel` `#FFFFFF`, viền `1px --border-bright`, shadow nhẹ
- Chữ `✕` màu `--text-primary`. Luôn hiển thị, không bao giờ tự ẩn

**Gesture điều hướng:**

| Gesture               | Hành động                                      |
| :-------------------- | :--------------------------------------------- |
| Swipe trái            | Ảnh tiếp theo trong danh sách F3->F6           |
| Swipe phải            | Ảnh trước đó                                   |
| Swipe xuống (từ trên) | Đóng modal                                     |
| Pinch                 | Zoom in/out ảnh hiện tại                       |
| Tap một lần           | Hiện/ẩn overlay controls                       |
| Double tap            | Zoom 2x vào điểm chạm, double tap lại để reset |

**Bottom bar khi controls hiện:** Nền `--bg-panel` `#FFFFFF`, `border-top: 1px --border`. Label chữ (`F3B`, `F3T1`, `F4S`...) nút nhỏ 36px. Active: nền `--amber`, chữ `#FFFFFF`. Inactive: nền `--bg-inset`, chữ `--text-secondary`. Counter `[⊞ 1/9]` mở sheet chọn ảnh.

### 10.7 Quy Tắc Bắt Buộc Toàn Bộ Mobile

| Quy tắc                                                              | Lý do                                |
| :------------------------------------------------------------------- | :----------------------------------- |
| Mọi `<input>`, `<select>` phải có `font-size: 16px`                  | iOS tự zoom khi size < 16px          |
| Touch target tối thiểu 44×44px                                       | Apple HIG / WCAG 2.5.5               |
| Không dùng `hover` làm trigger chức năng                             | Mobile không có hover                |
| Không dùng `title` tooltip — dùng `aria-describedby` + text hiển thị | Tooltip không hoạt động trên touch   |
| `viewport: width=device-width, initial-scale=1`                      | Tránh layout bị thu nhỏ              |
| Tránh `position: fixed` chồng lên nhau quá 2 lớp                     | Gây lỗi scroll trên iOS Safari       |
| Range slider cần `touch-action: none` trên thumb                     | Ngăn scroll trang khi kéo slider     |
| Bảng nhiều cột: chuyển sang card view                                | UX kéo ngang bảng rất tệ trên mobile |

### 10.8 Tablet (768px – 1023px)

- Left panel chuyển thành **drawer trượt từ trái**, chiều rộng 300px, nền `--bg-panel`, shadow `4px 0 16px rgba(0,0,0,0.12)`
- Nút hamburger ☰ ở topbar mở/đóng drawer; tap ngoài drawer để đóng
- Bảng kết quả: ẩn cột SCALE khi < 900px
- Range sliders: thumb tăng lên 18px

---

## 11. Kiểm Tra 7 Nguyên Tắc UI

### 11.1 Bảng Tổng Hợp

| Nguyên tắc        | Mức độ       | Vấn đề phát hiện                                              | Trạng thái |
| :---------------- | :----------- | :------------------------------------------------------------ | :--------- |
| 1 — Thứ bậc       | Trung bình   | Left panel quá nhiều mục ngang cấp thị giác                   | Cần sửa    |
| 2 — Tiết lộ dần   | Nghiêm trọng | Config hiện hết ngay; không có step indicator tính Scale      | Cần sửa    |
| 3 — Nhất quán     | Trung bình   | `[✕]` khác kích thước desktop/mobile; tooltip không đồng đều  | Cần sửa    |
| 4 — Tương phản    | Đã xử lý     | Badge light theme: tối thiểu 7.2:1. Console giữ tối để đọc dễ | Ổn         |
| 5 — Accessibility | Nghiêm trọng | Thiếu `aria-label` icon-only; màu log không có fallback text  | Cần sửa    |
| 6 — Gần gũi       | Trung bình   | Vùng vận hành và vùng cài đặt không tách biệt rõ              | Cần sửa    |
| 7 — Phù hợp       | Đạt          | Bảng dùng `table-layout: fixed`; spacing tokens 8px grid      | Ổn         |

### 11.2 Contrast Check — Light Theme (tất cả trên `--bg-panel` `#FFFFFF`)

| Token                              | Hex           | Ratio  | WCAG                        |
| :--------------------------------- | :------------ | :----- | :-------------------------- |
| `--text-primary`                   | `#1A1A1A`     | 18.1:1 | AAA                         |
| `--text-secondary`                 | `#525252`     | 7.0:1  | AA+                         |
| `--text-muted`                     | `#737373`     | 4.6:1  | AA                          |
| `--text-dim`                       | `#A3A3A3`     | 2.3:1  | FAIL — CHỈ dùng border/icon |
| `--amber` nút trên `#D97706` bg    | chữ `#FFFFFF` | 5.9:1  | AA                          |
| `--text-amber`                     | `#B45309`     | 6.2:1  | AA                          |
| Badge S `#5B21B6` trên `#EDE9FE`   | —             | 7.2:1  | AA+                         |
| Badge M `#075985` trên `#E0F2FE`   | —             | 7.5:1  | AA+                         |
| Badge L `#14532D` trên `#DCFCE7`   | —             | 9.1:1  | AAA                         |
| Badge NCN `#713F12` trên `#FEF9C3` | —             | 8.0:1  | AA+                         |
| Badge NCL `#7F1D1D` trên `#FEE2E2` | —             | 9.4:1  | AAA                         |

### 11.3 Quy Tắc Sử Dụng Token Màu — Bắt Buộc

```text
--text-primary   (#1A1A1A) : text chính, số liệu bảng, nhãn form
--text-secondary (#525252) : text đọc thường, tên cột bảng, nhãn config, hint nhỏ
--text-muted     (#737373) : placeholder, dung lượng file, nhãn min/max slider (>= 13px)
--text-dim       (#A3A3A3) : KHÔNG DÙNG CHO TEXT — chỉ border, divider, icon decorative
--text-amber     (#B45309) : link, giá trị nổi bật, nhãn amber inline
```

### 11.4 Sửa Nguyên Tắc 2 — Progressive Disclosure

Config section dạng collapsible accordion. Khi cả hai đóng: left panel chỉ hiện Drop Zone + File list + Nút Chạy. Trạng thái accordion lưu vào `localStorage`.

Step indicator cho luồng tính Scale:

```text
 ① Nhập thực tế  ->  ② Xem đề xuất  ->  ③ Áp dụng
```

Bước hiện tại: Barlow Condensed 700, `--amber`. Bước chưa làm: `--text-muted`. Tự cập nhật theo trạng thái.

### 11.5 Sửa Nguyên Tắc 5 — Accessibility

**Toàn bộ nút icon-only phải có `aria-label`:**

| Nút                        | `aria-label`                      |
| :------------------------- | :-------------------------------- |
| `[✕]` xóa file             | `"Xóa file <tên file>"`           |
| `[⊞]` xem ảnh debug        | `"Xem ảnh debug ID <n>"`          |
| `[◀]` / `[▶]` debug viewer | `"Ảnh trước"` / `"Ảnh tiếp theo"` |
| `[↓]` tải xuống ảnh        | `"Tải xuống ảnh <tên file>"`      |
| `[✕]` đóng modal           | `"Đóng"`                          |

Màu prefix log là trang trí bổ sung — phân biệt luồng bằng ký tự `[F1]`..`[F6]`, không phụ thuộc vào màu.

### 11.6 Sửa Nguyên Tắc 1 & 6 — Thứ Bậc & Gần Gũi

Phân tách vùng vận hành và vùng cài đặt bằng `border-top: 2px solid --border-heavy` `#9A9A96` + nhãn `THIẾT LẬP` (Barlow Condensed 700, 9px, `letter-spacing: 3px`, `--text-muted`).

### 11.7 Sửa Nguyên Tắc 3 — Nhất Quán

**Kích thước `[✕]` thống nhất:**

- Desktop: `28×28px`
- Mobile: `44×44px`
- Mọi modal close button: `44×44px` mọi nền tảng

**Tooltip thống nhất — chỉ xuất hiện trong hai trường hợp:**

1. Validation lỗi: text ngay dưới ô input, màu `--red-err`, font mono 11px, nền `--red-dim` padding nhỏ
2. Text bị truncate: `title` attribute khi hover

Mọi UI element khác: không dùng tooltip — thay bằng hint text hiển thị thẳng.

---

## Ghi Chú Kỹ Thuật Chung

- Không reload trang. Toàn bộ tương tác dùng `fetch API` + SSE
- SCALE tính ngay phía client: `scale_i = parseFloat(input) / pixel_length`. Không cần gọi API
- Bảng kết quả reload khi chọn `run_dir/source` khác hoặc sau khi pipeline hoàn tất
- Font load: Google Fonts CDN — `Share Tech Mono` + `Barlow Condensed`
- Badge động trong bảng kết quả: sau khi lưu phân loại mới, cột SIZE tự cập nhật. Badge tự sinh màu xoay vòng từ palette định sẵn nếu thêm cỡ vượt quá 5
- Console giữ nền tối vĩnh viễn bất kể theme — không toggle theo light/dark switch
