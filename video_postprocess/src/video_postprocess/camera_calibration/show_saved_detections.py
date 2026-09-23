import json
from ctypes import byref, windll
from ctypes.wintypes import RECT
from pathlib import Path
from typing import Annotated, Any

import cv2 as cv
import dearpygui.dearpygui as dpg
import numpy as np
import typer

WINDOW_NAME = "Saved Detections"
BOARD_SIZE = (5, 3)


def draw_detected_points(img: np.ndarray, corners: np.ndarray) -> None:
    for corner in corners:
        cv.circle(
            img,
            (int(corner[0][0]), int(corner[0][1])),
            1,
            (0, 0, 255),
            -1,
        )


def build_labeled_corners(
    points: np.ndarray, size: tuple[int, int]
) -> dict[str, tuple[np.ndarray, tuple[int, int, int]]]:
    cols, rows = size
    return {
        "Left Top": (points[0], (255, 0, 0)),
        "Right Top": (points[cols - 1], (0, 255, 0)),
        "Left Bottom": (
            points[(rows - 1) * cols],
            (0, 165, 255),
        ),
        "Right Bottom": (
            points[rows * cols - 1],
            (255, 0, 255),
        ),
    }


def draw_corner_labels(
    img: np.ndarray,
    points: np.ndarray,
    labeled_corners: dict[str, tuple[np.ndarray, tuple[int, int, int]]],
) -> None:
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    thickness = 2
    img_h, img_w = img.shape[:2]
    center_x = float(np.mean(points[:, 0]))
    center_y = float(np.mean(points[:, 1]))

    for label, (point, color) in labeled_corners.items():
        x, y = int(point[0]), int(point[1])
        cv.circle(img, (x, y), 3, color, -1)

        (text_w, text_h), baseline = cv.getTextSize(
            label, font, font_scale, thickness
        )

        text_x = x - text_w - 8 if x <= center_x else x + 8
        text_y = y - 6 if y <= center_y else y + text_h + 6

        text_x = max(0, min(text_x, img_w - text_w - 1))
        text_y = max(
            text_h + baseline,
            min(text_y, img_h - baseline - 1),
        )
        cv.putText(
            img,
            label,
            (text_x, text_y),
            font,
            font_scale,
            color,
            thickness,
            cv.LINE_AA,
        )


def _get_file_info_entry(
    file_info: list[list[dict[str, Any]]], pose_id: int, camera_id: int
) -> dict[str, Any] | None:
    if pose_id < 0 or pose_id >= len(file_info):
        return None
    row = file_info[pose_id]
    if camera_id < 0 or camera_id >= len(row):
        return None
    return row[camera_id]


def _resolve_image_path(
    file_info: list[list[dict[str, Any]]],
    pose_id: int,
    camera_id: int,
    images_folder: Path | None,
) -> Path | None:
    entry = _get_file_info_entry(file_info, pose_id, camera_id)
    if entry is None:
        return None

    value = entry.get("filePath")
    path = Path(value)

    if images_folder is None:
        if path.exists():
            return path
    else:
        path = images_folder / path.parent.name / path.name
        if path.exists():
            return path
    return None


def _is_included_detection(
    file_info: list[list[dict[str, Any]]], detection: dict[str, Any]
) -> bool:
    pose_id = int(detection.get("poseId", -1))
    camera_id = int(detection.get("cameraId", -1))
    entry = _get_file_info_entry(file_info, pose_id, camera_id)
    return bool(entry is not None and entry.get("included", False))


def _default_canvas_size(
    data: dict[str, Any], camera_id: int
) -> tuple[int, int]:
    default_w, default_h = 1920, 1080
    cameras = data.get("calibration", {}).get("cameras", [])
    if camera_id >= len(cameras):
        return default_w, default_h

    size = (
        cameras[camera_id]
        .get("model", {})
        .get("ptr_wrapper", {})
        .get("data", {})
        .get("CameraModelCRT", {})
        .get("CameraModelBase", {})
        .get("imageSize", {})
    )
    width = int(size.get("width", default_w))
    height = int(size.get("height", default_h))
    return width, height


def _feature_points_to_array(
    feature_points: list[dict[str, Any]],
) -> np.ndarray | None:
    if not feature_points:
        return None

    sorted_points = sorted(
        feature_points,
        key=lambda p: int(p.get("id", 0)),
    )
    points = []
    for point_data in sorted_points:
        pt = point_data.get("point", {})
        x = float(pt.get("x", 0.0))
        y = float(pt.get("y", 0.0))
        points.append([x, y])

    return np.asarray(points, dtype=np.float32)


def _set_included(
    file_info: list[list[dict[str, Any]]],
    pose_id: int,
    camera_id: int,
    included: bool,
) -> None:
    entry = _get_file_info_entry(file_info, pose_id, camera_id)
    if entry is None:
        return
    entry["included"] = included
    entry["status"] = 1 if included else 2


def _rotate_feature_points_90(
    feature_points: list[dict[str, Any]],
    size: tuple[int, int],
) -> tuple[list[dict[str, Any]], tuple[int, int]]:
    cols, rows = size
    sorted_points = sorted(feature_points, key=lambda p: int(p.get("id", 0)))
    if len(sorted_points) < cols * rows:
        return feature_points, size

    points = []
    for point_data in sorted_points[: cols * rows]:
        pt = point_data.get("point", {})
        points.append([float(pt.get("x", 0.0)), float(pt.get("y", 0.0))])

    grid = np.asarray(points, dtype=np.float32).reshape(rows, cols, 2)
    rotated_grid = np.rot90(grid)
    new_rows, new_cols = rotated_grid.shape[:2]

    rotated_points: list[dict[str, Any]] = []
    flat = rotated_grid.reshape(-1, 2)
    for idx, point in enumerate(flat):
        rotated_points.append(
            {
                "point": {
                    "x": float(point[0]),
                    "y": float(point[1]),
                },
                "id": idx,
            }
        )
    return rotated_points, (new_cols, new_rows)


def _write_json(
    data: dict[str, Any],
    file_info: list[list[dict[str, Any]]],
    detections: list[dict[str, Any]],
    output_file: Path,
) -> None:
    data["fileInfo"] = file_info
    data["detections"] = detections
    with output_file.open("w") as f:
        json.dump(data, f, indent=4)


def _get_windows_work_area() -> tuple[int, int, int, int]:
    rect = RECT()
    success = windll.user32.SystemParametersInfoW(
        0x0030,  # SPI_GETWORKAREA
        0,
        byref(rect),
        0,
    )
    if not success:
        screen_w = windll.user32.GetSystemMetrics(0)
        screen_h = windll.user32.GetSystemMetrics(1)
        return 0, 0, int(screen_w), int(screen_h)
    width = rect.right - rect.left
    height = rect.bottom - rect.top
    return int(rect.left), int(rect.top), int(width), int(height)


class DetectionsReviewer:
    TEXTURE_TAG = "review_texture"
    MAIN_WINDOW_TAG = "main_window"
    STATUS_TAG = "status_text"
    IMAGE_TAG = "frame_image"
    CONFIRM_MODAL_TAG = "confirm_overwrite_modal"
    QUIT_MODAL_TAG = "confirm_quit_modal"

    def __init__(
        self,
        data: dict[str, Any],
        file_info: list[list[dict[str, Any]]],
        detections: list[dict[str, Any]],
        size_by_detection: dict[tuple[int, int], tuple[int, int]],
        output_file: Path,
        images_folder: Path | None,
    ) -> None:
        self.data = data
        self.file_info = file_info
        self.detections = detections
        self.size_by_detection = size_by_detection
        self.output_file = output_file
        self.images_folder = images_folder

        self.current = 0
        self.dirty = True
        self.running = True
        self.texture_size: tuple[int, int] | None = None
        self.is_saving = False
        self.exit_after_save = False
        self.pending_save_request = False

    def _frame_to_texture_data(self, frame: np.ndarray) -> np.ndarray:
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        return np.true_divide(
            rgb.astype(np.float32).ravel(),
            255.0,
            dtype=np.float32,
        )

    def _build_frame(self) -> np.ndarray:
        det = self.detections[self.current]
        pose_id = int(det.get("poseId", 0))
        camera_id = int(det.get("cameraId", 0))
        feature_points = det.get("featurePoints", [])
        detection_key = (pose_id, camera_id)
        current_size = self.size_by_detection.get(detection_key, BOARD_SIZE)

        image_path = _resolve_image_path(
            self.file_info,
            pose_id,
            camera_id,
            self.images_folder,
        )
        if image_path is not None and image_path.exists():
            frame = cv.imread(str(image_path))
        else:
            width, height = _default_canvas_size(self.data, camera_id)
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            cv.putText(
                frame,
                "Image not found in fileInfo",
                (12, 64),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv.LINE_AA,
            )

        ordered_points = _feature_points_to_array(feature_points)
        if ordered_points is not None:
            draw_detected_points(frame, ordered_points.reshape(-1, 1, 2))
            expected_points = current_size[0] * current_size[1]
            if ordered_points.shape[0] >= expected_points:
                labeled_corners = build_labeled_corners(
                    ordered_points, current_size
                )
                draw_corner_labels(frame, ordered_points, labeled_corners)

        if self.texture_size is not None:
            width, height = self.texture_size
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv.resize(
                    frame,
                    (width, height),
                    interpolation=cv.INTER_AREA,
                )

        return frame

    def _set_status_text(self) -> None:
        det = self.detections[self.current]
        pose_id = int(det.get("poseId", 0))
        camera_id = int(det.get("cameraId", 0))
        text = (
            f"pose={pose_id} camera={camera_id:02d} "
            f"index={self.current + 1}/{len(self.detections)}"
        )
        dpg.set_value(self.STATUS_TAG, text)

    def _refresh_texture(self) -> None:
        frame = self._build_frame()
        texture_data = self._frame_to_texture_data(frame)
        dpg.set_value(self.TEXTURE_TAG, texture_data)
        self._set_status_text()

    def _close_modal(self, modal_tag: str) -> None:
        if dpg.does_item_exist(modal_tag):
            dpg.configure_item(modal_tag, show=False)

    def _close_app(self) -> None:
        self.running = False
        dpg.stop_dearpygui()

    def _save_without_prompt(self) -> None:
        self.is_saving = True
        _write_json(
            self.data,
            self.file_info,
            self.detections,
            self.output_file,
        )
        typer.echo(f"Saved: {self.output_file}")
        self.is_saving = False

    def _confirm_overwrite(self, overwrite: bool) -> None:
        self._close_modal(self.CONFIRM_MODAL_TAG)
        if overwrite:
            self._save_without_prompt()
            if self.exit_after_save:
                self.exit_after_save = False
                self._close_app()
            return
        self.exit_after_save = False

    def _show_overwrite_modal(self) -> None:
        if dpg.does_item_exist(self.CONFIRM_MODAL_TAG):
            dpg.configure_item(self.CONFIRM_MODAL_TAG, show=True)
            return

        with dpg.window(
            tag=self.CONFIRM_MODAL_TAG,
            label="Confirm Save",
            modal=True,
            no_move=True,
            no_resize=True,
            no_collapse=True,
            width=700,
            height=170,
            pos=(80, 80),
        ):
            dpg.add_text("Output file already exists.")
            dpg.add_text(str(self.output_file))
            dpg.add_spacer(height=8)
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Yes, overwrite",
                    width=160,
                    callback=lambda: self._confirm_overwrite(True),
                )
                dpg.add_button(
                    label="No",
                    width=120,
                    callback=lambda: self._confirm_overwrite(False),
                )

    def _on_quit_choice(self, choice: str) -> None:
        self._close_modal(self.QUIT_MODAL_TAG)
        if choice == "save":
            self.exit_after_save = True
            self.pending_save_request = True
            return
        if choice == "discard":
            self.exit_after_save = False
            self._close_app()
            return
        self.exit_after_save = False

    def _show_quit_modal(self) -> None:
        if dpg.does_item_exist(self.QUIT_MODAL_TAG):
            dpg.configure_item(self.QUIT_MODAL_TAG, show=True)
            return

        with dpg.window(
            tag=self.QUIT_MODAL_TAG,
            label="Quit",
            modal=True,
            no_move=True,
            no_resize=True,
            no_collapse=True,
            width=560,
            height=170,
            pos=(80, 80),
        ):
            dpg.add_text("Save changes before quitting?")
            dpg.add_spacer(height=8)
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Save",
                    width=120,
                    callback=lambda: self._on_quit_choice("save"),
                )
                dpg.add_button(
                    label="Don't Save",
                    width=120,
                    callback=lambda: self._on_quit_choice("discard"),
                )
                dpg.add_button(
                    label="Cancel",
                    width=120,
                    callback=lambda: self._on_quit_choice("cancel"),
                )

    def _request_save(self) -> None:
        if self.output_file.exists():
            self._show_overwrite_modal()
            return
        self._save_without_prompt()
        if self.exit_after_save:
            self.exit_after_save = False
            self._close_app()

    def _next(self) -> None:
        if not self.detections:
            return
        self.current = (self.current + 1) % len(self.detections)
        self.dirty = True

    def _prev(self) -> None:
        if not self.detections:
            return
        self.current = (self.current - 1) % len(self.detections)
        self.dirty = True

    def _keep(self) -> None:
        if not self.detections:
            return
        det = self.detections[self.current]
        pose_id = int(det.get("poseId", 0))
        camera_id = int(det.get("cameraId", 0))
        _set_included(self.file_info, pose_id, camera_id, True)
        self._next()

    def _rotate(self) -> None:
        if not self.detections:
            return
        det = self.detections[self.current]
        pose_id = int(det.get("poseId", 0))
        camera_id = int(det.get("cameraId", 0))
        detection_key = (pose_id, camera_id)
        current_size = self.size_by_detection.get(detection_key, BOARD_SIZE)
        feature_points = det.get("featurePoints", [])
        rotated_points, new_size = _rotate_feature_points_90(
            feature_points,
            current_size,
        )
        det["featurePoints"] = rotated_points
        self.size_by_detection[detection_key] = new_size
        self.dirty = True

    def _discard(self) -> None:
        if not self.detections:
            return
        det = self.detections[self.current]
        pose_id = int(det.get("poseId", 0))
        camera_id = int(det.get("cameraId", 0))
        detection_key = (pose_id, camera_id)

        _set_included(self.file_info, pose_id, camera_id, False)
        self.detections.pop(self.current)
        self.size_by_detection.pop(detection_key, None)

        if not self.detections:
            typer.echo("No detections left after discard.")
            return

        self.current %= len(self.detections)
        self.dirty = True

    def _quit(self) -> None:
        self._show_quit_modal()

    def _on_exit(self) -> None:
        if self.is_saving:
            return
        if self.running:
            self._quit()

    def _on_key(self, sender: int, app_data: int) -> None:
        del sender
        if dpg.does_item_exist(self.CONFIRM_MODAL_TAG) and dpg.is_item_shown(
            self.CONFIRM_MODAL_TAG
        ):
            if app_data == dpg.mvKey_Y:
                self._confirm_overwrite(True)
            elif app_data in (dpg.mvKey_N, dpg.mvKey_Escape):
                self._confirm_overwrite(False)
            return

        if dpg.does_item_exist(self.QUIT_MODAL_TAG) and dpg.is_item_shown(
            self.QUIT_MODAL_TAG
        ):
            if app_data in (dpg.mvKey_Y, dpg.mvKey_S):
                self._on_quit_choice("save")
            elif app_data == dpg.mvKey_N:
                self._on_quit_choice("discard")
            elif app_data == dpg.mvKey_Escape:
                self._on_quit_choice("cancel")
            return

        if app_data in (dpg.mvKey_Right, dpg.mvKey_N):
            self._next()
            return
        if app_data in (dpg.mvKey_Left, dpg.mvKey_P, dpg.mvKey_Back):
            self._prev()
            return
        if app_data == dpg.mvKey_Spacebar:
            self._keep()
            return
        if app_data == dpg.mvKey_R:
            self._rotate()
            return
        if app_data == dpg.mvKey_X:
            self._discard()
            return
        if app_data == dpg.mvKey_S:
            self._request_save()
            return
        if app_data in (dpg.mvKey_Q, dpg.mvKey_Escape):
            self._quit()

    def _setup_ui(self) -> None:
        first_frame = self._build_frame()
        frame_h, frame_w = first_frame.shape[:2]

        work_x, work_y, viewport_w, viewport_h = _get_windows_work_area()

        # Reserve vertical space for header/status/buttons and fit image.
        available_w = max(320, viewport_w - 40)
        available_h = max(240, viewport_h - 140)
        scale = min(
            available_w / frame_w,
            available_h / frame_h,
            1.0,
        )
        render_w = max(1, int(frame_w * scale))
        render_h = max(1, int(frame_h * scale))
        self.texture_size = (render_w, render_h)

        if render_w != frame_w or render_h != frame_h:
            first_frame = cv.resize(
                first_frame,
                (render_w, render_h),
                interpolation=cv.INTER_AREA,
            )

        texture_data = self._frame_to_texture_data(first_frame)

        dpg.create_context()
        dpg.configure_app(manual_callback_management=True)
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                render_w,
                render_h,
                texture_data,
                tag=self.TEXTURE_TAG,
                format=dpg.mvFormat_Float_rgb,
            )

        dpg.create_viewport(
            title=WINDOW_NAME,
            width=viewport_w,
            height=viewport_h,
            x_pos=work_x,
            y_pos=work_y,
            resizable=True,
        )

        with dpg.window(
            tag=self.MAIN_WINDOW_TAG,
            label=WINDOW_NAME,
            no_close=True,
            no_collapse=True,
            no_move=True,
            no_resize=True,
            no_scrollbar=True,
            no_scroll_with_mouse=True,
            width=viewport_w,
            height=viewport_h,
            pos=(0, 0),
        ):
            dpg.add_text(
                "left/p=prev, right/n=next, space=keep, r=rotate, x=discard, s=save, q=quit"
            )
            dpg.add_text("", tag=self.STATUS_TAG)
            dpg.add_image(self.TEXTURE_TAG, tag=self.IMAGE_TAG)
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Prev", width=90, callback=lambda: self._prev()
                )
                dpg.add_button(
                    label="Next", width=90, callback=lambda: self._next()
                )
                dpg.add_button(
                    label="Keep", width=90, callback=lambda: self._keep()
                )
                dpg.add_button(
                    label="Rotate", width=90, callback=lambda: self._rotate()
                )
                dpg.add_button(
                    label="Discard", width=90, callback=lambda: self._discard()
                )
                dpg.add_button(
                    label="Save",
                    width=90,
                    callback=lambda: self._request_save(),
                )
                dpg.add_button(
                    label="Quit", width=90, callback=lambda: self._quit()
                )

        with dpg.handler_registry():
            dpg.add_key_press_handler(callback=self._on_key)

        self._set_status_text()
        dpg.setup_dearpygui()
        dpg.set_exit_callback(self._on_exit)
        dpg.show_viewport()
        dpg.set_primary_window(self.MAIN_WINDOW_TAG, True)

    def run(self) -> None:
        self._setup_ui()
        while dpg.is_dearpygui_running() and self.running:
            if self.pending_save_request:
                self.pending_save_request = False
                self._request_save()

            jobs = dpg.get_callback_queue()  # retrieves and clears queue
            dpg.run_callbacks(jobs)

            if self.dirty:
                self._refresh_texture()
                self.dirty = False
            dpg.render_dearpygui_frame()
        dpg.destroy_context()


def main(
    calib_file: Path,
    output_file: Path,
    images_folder: Annotated[
        Path | None,
        typer.Option(
            "--images-folder",
            "-i",
            help="Optional root folder containing images referenced by fileInfo.",
        ),
    ] = None,
    only_included: Annotated[
        bool,
        typer.Option(
            "--only-included",
            help="Show only detections marked as included in fileInfo.",
        ),
    ] = True,
) -> None:
    with calib_file.open() as f:
        data = json.load(f)

    detections: list[dict[str, Any]] = data.get("detections", [])
    if not detections:
        typer.echo("No detections found in file.")
        return

    file_info: list[list[dict[str, Any]]] = data.get("fileInfo", [])
    if only_included:
        detections = [
            det for det in detections if _is_included_detection(file_info, det)
        ]
        if not detections:
            typer.echo("No included detections found in file.")
            return

    detections = sorted(
        detections,
        key=lambda d: (int(d.get("poseId", 0)), int(d.get("cameraId", 0))),
    )

    size_by_detection: dict[tuple[int, int], tuple[int, int]] = {
        (int(d.get("poseId", 0)), int(d.get("cameraId", 0))): BOARD_SIZE
        for d in detections
    }

    reviewer = DetectionsReviewer(
        data=data,
        file_info=file_info,
        detections=detections,
        size_by_detection=size_by_detection,
        output_file=output_file,
        images_folder=images_folder,
    )
    reviewer.run()


if __name__ == "__main__":
    typer.run(main)
