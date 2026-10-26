"""Tests for video.py (Frame and Video). video.py must be in the project root."""
import dataclasses

import numpy as np
import pytest

from Algorithem.video import Frame, Video

# The mp4 tests need the optional dependency:  pip install "imageio[ffmpeg]"
imageio = pytest.importorskip("imageio", reason="imageio not installed")
pytest.importorskip("imageio_ffmpeg", reason="imageio-ffmpeg not installed")


def img(h=4, w=6, value=0):
    """A small RGB uint8 image filled with one value."""
    return np.full((h, w, 3), value, dtype=np.uint8)


# ------------------------------------------------------------------ Frame
def test_frame_keeps_data_and_duration():
    f = Frame("data", 0.5)
    assert f.data == "data"
    assert f.duration == 0.5


@pytest.mark.parametrize("bad", [0, -1, -0.001])
def test_frame_rejects_non_positive_duration(bad):
    with pytest.raises(ValueError, match="positive"):
        Frame("x", bad)


def test_frame_is_immutable():
    f = Frame("x", 1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(f, "duration", 2.0)   # setattr: the IDE warns about a direct assignment


# ------------------------------------------------------------ building
def test_empty_video():
    v = Video()
    assert len(v) == 0
    assert v.duration == 0.0
    assert v.timeline.shape == (0, 2)


def test_add_frame_with_explicit_duration():
    v = Video()
    v.add_frame("a", 0.5).add_frame("b", 2.0)        # chaining works
    assert len(v) == 2
    assert v.duration == pytest.approx(2.5)
    assert v[0].data == "a" and v[1].duration == 2.0


def test_add_frame_uses_default_duration_from_fps():
    v = Video(fps=4)
    v.add_frame("a")
    assert v[0].duration == pytest.approx(0.25)


def test_explicit_duration_overrides_fps():
    v = Video(fps=4)
    v.add_frame("a", 1.0)
    assert v[0].duration == 1.0


def test_add_frame_without_duration_or_fps_fails():
    with pytest.raises(ValueError, match="duration"):
        Video().add_frame("a")


def test_failed_add_does_not_change_the_video():
    v = Video().add_frame("a", 1.0)
    with pytest.raises(ValueError):
        v.add_frame("b", -1)
    assert len(v) == 1
    assert v.duration == pytest.approx(1.0)


def test_from_frames_with_duration():
    v = Video.from_frames(["a", "b", "c"], duration=0.5)
    assert len(v) == 3
    assert v.duration == pytest.approx(1.5)


def test_from_frames_with_fps():
    v = Video.from_frames(["a", "b"], fps=10)
    assert v.duration == pytest.approx(0.2)


def test_extend_and_iteration():
    v = Video().extend(["a", "b"], 1.0)
    assert [f.data for f in v] == ["a", "b"]


def test_repr():
    v = Video().add_frame("a", 2.75)
    assert repr(v) == "Video(frames=1, duration=2.750s)"


# ------------------------------------------------------------- timeline
def test_timeline_values():
    v = Video().add_frame("a", 0.5).add_frame("b", 2.0).add_frame("c", 0.25)
    assert np.allclose(v.start_times, [0.0, 0.5, 2.5])
    assert np.allclose(v.end_times, [0.5, 2.5, 2.75])
    assert v.timeline.shape == (3, 2)
    assert np.allclose(v.timeline[1], [0.5, 2.5])
    assert v.duration == pytest.approx(2.75)


def test_timeline_has_no_gaps():
    v = Video.from_frames(range(20), duration=0.1)
    t = v.timeline
    assert np.allclose(t[1:, 0], t[:-1, 1])


# ------------------------------------------------------ access by time
@pytest.fixture
def abc():
    return Video().add_frame("a", 0.5).add_frame("b", 2.0).add_frame("c", 0.25)


def test_frame_index_at_inside_frames(abc):
    assert abc.frame_index_at(0.0) == 0
    assert abc.frame_index_at(0.25) == 0
    assert abc.frame_index_at(1.0) == 1
    assert abc.frame_index_at(2.6) == 2


def test_frame_index_at_boundaries(abc):
    assert abc.frame_index_at(0.5) == 1          # start of frame b belongs to b
    assert abc.frame_index_at(0.4999999) == 0
    assert abc.frame_index_at(2.5) == 2
    assert abc.frame_index_at(2.7499999) == 2


@pytest.mark.parametrize("t", [-0.1, 2.75, 10])
def test_frame_index_at_out_of_range(abc, t):
    with pytest.raises(ValueError, match="out of range"):
        abc.frame_index_at(t)


def test_frame_index_at_empty_video():
    with pytest.raises(IndexError):
        Video().frame_index_at(0)


def test_frame_at_returns_data(abc):
    assert abc.frame_at(0.1) == "a"
    assert abc.frame_at(1.0) == "b"


# --------------------------------------------------------------- sample
def test_sample_count_and_order(abc):
    out = list(abc.sample(4))                      # 2.75 s * 4 fps = 11 steps
    assert len(out) == 11
    idx = [i for i, _ in out]
    assert idx == sorted(idx)                      # never goes back
    assert idx == [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2]


def test_sample_returns_the_frame_data(abc):
    assert [d for _, d in abc.sample(4)][:3] == ["a", "a", "b"]


def test_sample_one_step_per_frame_when_fps_matches():
    v = Video.from_frames(["a", "b", "c"], fps=10)
    assert [i for i, _ in v.sample(10)] == [0, 1, 2]


def test_sample_very_short_video_gives_at_least_one_step():
    v = Video().add_frame("a", 0.001)
    assert list(v.sample(30)) == [(0, "a")]


# ----------------------------------------------------- _to_rgb_uint8
conv = Video._to_rgb_uint8


def test_gray_uint8_becomes_rgb():
    out = conv(np.full((4, 6), 7, np.uint8))
    assert out.shape == (4, 6, 3) and out.dtype == np.uint8
    assert (out == 7).all()


def test_float_range_0_1_is_scaled():
    out = conv(np.array([[0.0, 0.5, 1.0, 1.0]]).repeat(2, axis=0))
    assert out[0, :, 0].tolist() == [0, 127, 255, 255]


def test_float_out_of_range_is_clipped():
    out = conv(np.array([[-1.0, 2.0], [0.0, 0.0]]))
    assert out[0, 0, 0] == 0 and out[0, 1, 0] == 255


def test_integer_values_are_clipped_not_scaled():
    # NOTE: uint16 / int data is clipped to 0..255, NOT scaled. Window 16/32-bit
    # volume data to 8 bits before building frames.
    out = conv(np.array([[-5, 100], [300, 70000]], dtype=np.int64))
    assert out[:, :, 0].tolist() == [[0, 100], [255, 255]]


def test_single_channel_is_repeated():
    out = conv(np.full((4, 4, 1), 9, np.uint8))
    assert out.shape == (4, 4, 3) and (out == 9).all()


def test_alpha_channel_is_dropped():
    data = np.zeros((4, 4, 4), np.uint8)
    data[..., 3] = 255
    out = conv(data)
    assert out.shape == (4, 4, 3) and (out == 0).all()


@pytest.mark.parametrize("shape", [(4, 4, 2), (4, 4, 5), (4,), (2, 2, 3, 3)])
def test_unsupported_shapes_rejected(shape):
    with pytest.raises(ValueError, match="unsupported"):
        conv(np.zeros(shape, np.uint8))


def test_odd_sizes_are_cropped_to_even():
    out = conv(img(5, 7))
    assert out.shape == (4, 6, 3)


def test_result_is_contiguous_and_does_not_change_input():
    src = img(5, 7, 3)
    out = conv(src)
    assert out.flags["C_CONTIGUOUS"]
    assert src.shape == (5, 7, 3)


def test_image_file_path_is_loaded(tmp_path):
    p = tmp_path / "f.png"
    imageio.imwrite(p, img(4, 6, 200))
    out = conv(str(p))
    assert out.shape == (4, 6, 3) and (out == 200).all()


# ------------------------------------------------------------- to_mp4
def read_mp4(path):
    reader = imageio.get_reader(str(path))
    frames = [f for f in reader]
    reader.close()
    return frames


def test_to_mp4_writes_playable_file(tmp_path):
    v = Video.from_frames([img(8, 10, 0), img(8, 10, 255)], duration=0.5)
    out = tmp_path / "out.mp4"
    v.to_mp4(str(out), fps=10)
    assert out.exists() and out.stat().st_size > 0
    frames = read_mp4(out)
    assert len(frames) == 10                       # 1 s * 10 fps
    assert frames[0].shape == (8, 10, 3)


def test_to_mp4_keeps_frame_order_and_timing(tmp_path):
    # black for 0.5 s, then white for 0.5 s
    v = Video().add_frame(img(8, 8, 0), 0.5).add_frame(img(8, 8, 255), 0.5)
    out = tmp_path / "bw.mp4"
    v.to_mp4(str(out), fps=10)
    frames = read_mp4(out)
    means = [f.mean() for f in frames]
    assert all(m < 40 for m in means[:5])
    assert all(m > 215 for m in means[5:])


def test_to_mp4_accepts_gray_and_float_frames(tmp_path):
    v = Video.from_frames([np.zeros((6, 8), np.uint8), np.full((6, 8), 0.5)], duration=0.2)
    out = tmp_path / "g.mp4"
    v.to_mp4(str(out), fps=10)
    assert len(read_mp4(out)) == 4


def test_to_mp4_odd_size_frames_work(tmp_path):
    v = Video.from_frames([img(7, 9, 50)], duration=0.3)
    out = tmp_path / "odd.mp4"
    v.to_mp4(str(out), fps=10)
    assert read_mp4(out)[0].shape == (6, 8, 3)


def test_to_mp4_empty_video_fails(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        Video().to_mp4(str(tmp_path / "x.mp4"))


def test_to_mp4_frames_of_different_size_fail(tmp_path):
    v = Video.from_frames([img(8, 8), img(10, 10)], duration=0.2)
    with pytest.raises(ValueError, match="shape"):
        v.to_mp4(str(tmp_path / "x.mp4"), fps=10)