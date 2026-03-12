"""OpenCV Tools - Capture images and videos from webcam."""

import time
from typing import Any, Dict, Optional
from uuid import uuid4

from omnicoreagent.core.tools.local_tools_registry import Tool
from omnicoreagent.core.utils import log_error, log_info

try:
    import cv2
except ImportError:
    cv2 = None


class OpenCVCaptureImage:
    def __init__(self, show_preview: bool = False):
        if cv2 is None:
            raise ImportError(
                "`opencv-python` package not found. Please install with `pip install opencv-python`"
            )
        self.show_preview = show_preview

    def get_tool(self) -> Tool:
        return Tool(
            name="opencv_capture_image",
            description="Capture an image from the webcam.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "default": "Webcam capture"},
                    "output_path": {
                        "type": "string",
                        "description": "Path to save the image",
                    },
                },
            },
            function=self._capture_image,
        )

    async def _capture_image(
        self, prompt: str = "Webcam capture", output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return {
                    "status": "error",
                    "data": None,
                    "message": "Could not open webcam",
                }

            time.sleep(1)  # Allow camera to warm up

            ret, frame = cap.read()
            cap.release()

            if not ret:
                return {
                    "status": "error",
                    "data": None,
                    "message": "Failed to capture frame",
                }

            if not output_path:
                output_path = f"capture_{uuid4().hex[:8]}.jpg"

            cv2.imwrite(output_path, frame)
            log_info(f"Image captured: {output_path}")
            return {
                "status": "success",
                "data": output_path,
                "message": f"Image captured: {output_path}",
            }
        except Exception as e:
            log_error(f"Error capturing image: {e}")
            return {"status": "error", "data": None, "message": str(e)}


class OpenCVCaptureVideo:
    def __init__(self, show_preview: bool = False):
        self.show_preview = show_preview

    def get_tool(self) -> Tool:
        return Tool(
            name="opencv_capture_video",
            description="Capture a video from the webcam.",
            inputSchema={
                "type": "object",
                "properties": {
                    "duration": {
                        "type": "integer",
                        "default": 10,
                        "description": "Duration in seconds",
                    },
                    "prompt": {"type": "string", "default": "Webcam video capture"},
                    "output_path": {"type": "string"},
                },
            },
            function=self._capture_video,
        )

    async def _capture_video(
        self,
        duration: int = 10,
        prompt: str = "Webcam video capture",
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return {
                    "status": "error",
                    "data": None,
                    "message": "Could not open webcam",
                }

            if not output_path:
                output_path = f"video_{uuid4().hex[:8]}.mp4"

            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            log_info(f"Recording {duration}s of video...")
            start_time = time.time()
            frame_count = 0

            while (time.time() - start_time) < duration:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                frame_count += 1

                if self.show_preview:
                    cv2.imshow("Recording", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            cap.release()
            out.release()
            if self.show_preview:
                cv2.destroyAllWindows()

            log_info(f"Video captured: {output_path} ({frame_count} frames)")
            return {
                "status": "success",
                "data": {
                    "path": output_path,
                    "frames": frame_count,
                    "duration": duration,
                },
                "message": f"Video captured: {output_path}",
            }
        except Exception as e:
            log_error(f"Error capturing video: {e}")
            return {"status": "error", "data": None, "message": str(e)}
