"""Video: רצף frames עם זמן תצוגה לכל frame, וייצוא ל-mp4.

התקנה:
    pip install numpy "imageio[ffmpeg]"
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

import numpy as np


# ======================================================================
# Frame
# ======================================================================
@dataclass(frozen=True)
class Frame:
    """frame בודד: התוכן וזמן התצוגה שלו."""

    data: Any          # np.ndarray, נתיב לקובץ תמונה, או כל אובייקט אחר
    duration: float    # זמן תצוגה בשניות

    def __post_init__(self):
        if self.duration <= 0:
            raise ValueError(f"duration must be positive, got {self.duration}")


# ======================================================================
# Video
# ======================================================================
class Video:
    """וידאו כציר זמן: כל frame מוצג לפי ה-duration שלו."""

    def __init__(self, fps: float | None = None):
        """fps (אופציונלי): קובע זמן תצוגה ברירת מחדל ל-frame (1/fps)."""
        self._default_duration = 1.0 / fps if fps else None
        self._frames: list[Frame] = []
        self._starts: list[float] = []   # זמן התחלה של כל frame
        self._total: float = 0.0

    @classmethod
    def from_frames(
        cls, frames: Iterable[Any], duration: float | None = None, fps: float | None = None
    ) -> "Video":
        """בונה Video מרשימת frames עם אותו זמן תצוגה לכולם."""
        video = cls(fps=fps)
        return video.extend(frames, duration)

    # ------------------------------------------------------------------
    # בנייה
    # ------------------------------------------------------------------
    def add_frame(self, data: Any, duration: float | None = None) -> "Video":
        duration = duration if duration is not None else self._default_duration
        if duration is None:
            raise ValueError("pass duration, or set fps in the constructor")

        frame = Frame(data, duration)
        self._frames.append(frame)
        self._starts.append(self._total)
        self._total += frame.duration
        return self  # מאפשר שרשור

    def extend(self, frames: Iterable[Any], duration: float | None = None) -> "Video":
        for data in frames:
            self.add_frame(data, duration)
        return self

    # ------------------------------------------------------------------
    # ציר הזמן
    # ------------------------------------------------------------------
    @property
    def duration(self) -> float:
        """אורך הווידאו הכולל בשניות."""
        return self._total

    @property
    def start_times(self) -> np.ndarray:
        return np.array(self._starts)

    @property
    def end_times(self) -> np.ndarray:
        return self.start_times + np.array([f.duration for f in self._frames])

    @property
    def timeline(self) -> np.ndarray:
        """מערך Nx2 של [התחלה, סיום] לכל frame."""
        return np.stack([self.start_times, self.end_times], axis=1)

    # ------------------------------------------------------------------
    # גישה לפי זמן
    # ------------------------------------------------------------------
    def frame_index_at(self, t: float) -> int:
        """האינדקס של ה-frame שמוצג בזמן t (חיפוש בינארי)."""
        if not self._frames:
            raise IndexError("video is empty")
        if not 0 <= t < self._total:
            raise ValueError(f"t={t} out of range [0, {self._total})")
        return bisect_right(self._starts, t) - 1

    def frame_at(self, t: float) -> Any:
        return self._frames[self.frame_index_at(t)].data

    def sample(self, fps: float) -> Iterator[tuple[int, Any]]:
        """דגימה בקצב קבוע. מחזיר (אינדקס frame מקורי, data) לכל צעד."""
        for i in range(max(1, round(self._total * fps))):
            idx = self.frame_index_at(min(i / fps, self._total - 1e-9))
            yield idx, self._frames[idx].data

    # ------------------------------------------------------------------
    # ייצוא
    # ------------------------------------------------------------------
    def to_mp4(self, path: str, fps: float = 30, quality: int = 8) -> None:
        """שומר כקובץ mp4 בקצב fps קבוע (הזמנים המקוריים נשמרים).

        quality: 0-10, גבוה יותר = איכות טובה יותר.
        """
        import imageio.v2 as imageio

        if not self._frames:
            raise ValueError("video is empty")

        last_idx, last_img, shape = None, None, None

        with imageio.get_writer(
            path, fps=fps, codec="libx264", quality=quality,
            pixelformat="yuv420p", macro_block_size=2,
        ) as writer:
            for idx, data in self.sample(fps):
                if idx != last_idx:  # ממיר רק כשה-frame מתחלף
                    last_img = self._to_rgb_uint8(data)
                    if shape is None:
                        shape = last_img.shape
                    elif last_img.shape != shape:
                        raise ValueError(
                            f"frame {idx} has shape {last_img.shape}, expected {shape}"
                        )
                    last_idx = idx
                writer.append_data(last_img)

    @staticmethod
    def _to_rgb_uint8(data: Any) -> np.ndarray:
        """ממיר frame ל-RGB uint8 עם מידות זוגיות (דרישת h264)."""
        # נתיב לקובץ תמונה
        if isinstance(data, (str, bytes)) or hasattr(data, "__fspath__"):
            import imageio.v2 as imageio
            data = imageio.imread(data)

        img = np.asarray(data)

        # סוג הנתונים -> uint8 (float מוגדר בטווח [0,1])
        if img.dtype != np.uint8:
            if np.issubdtype(img.dtype, np.floating):
                img = img * 255
            img = np.clip(img, 0, 255).astype(np.uint8)

        # מספר ערוצים -> 3
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = img[..., :3]  # זורק alpha
        elif img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"unsupported frame shape: {img.shape}")

        # מידות זוגיות
        h, w = img.shape[:2]
        return np.ascontiguousarray(img[: h // 2 * 2, : w // 2 * 2])

    # ------------------------------------------------------------------
    # פרוטוקולים של Python
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, i: int) -> Frame:
        return self._frames[i]

    def __iter__(self) -> Iterator[Frame]:
        return iter(self._frames)

    def __repr__(self) -> str:
        return f"Video(frames={len(self)}, duration={self._total:.3f}s)"


# ======================================================================
# דוגמה
# ======================================================================
if __name__ == "__main__":
    v = Video()
    v.add_frame(np.random.randint(0, 255, (480, 640, 3), np.uint8), 0.5)
    v.add_frame(np.zeros((480, 640, 3), np.uint8), 2.0)
    v.add_frame(np.full((480, 640), 0.5), 0.25)   # גווני אפור, float

    print(v)                 # Video(frames=3, duration=2.750s)
    print(v.timeline)
    v.to_mp4("out.mp4", fps=30)