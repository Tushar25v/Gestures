"""Tests for the refactored pure functions in app.py and app_main.py."""
import sys
import os
from collections import deque
from unittest.mock import MagicMock

import numpy as np
import pytest

# We can't directly import app.py because it requires mediapipe/tensorflow at
# module level.  Instead, load only the pure helper functions we want to test
# by reading the source and exec-ing them in an isolated namespace.

_APP_DIR = os.path.join(os.path.dirname(__file__), '..')
_APP_PY = os.path.join(_APP_DIR, 'app.py')


def _load_helpers():
    """Return a namespace containing just the helper functions."""
    import cv2 as cv

    ns = {"np": np, "cv": cv, "__builtins__": __builtins__}

    with open(_APP_PY) as f:
        source = f.read()

    # Execute only from 'def select_mode' onward (pure helper functions)
    func_start = source.index("\ndef select_mode")
    exec(compile(source[func_start:], _APP_PY, "exec"), ns)
    return ns


_helpers = _load_helpers()

pre_process_landmark = _helpers["pre_process_landmark"]
pre_process_point_history = _helpers["pre_process_point_history"]
select_mode = _helpers["select_mode"]
calc_bounding_rect = _helpers["calc_bounding_rect"]
draw_landmarks = _helpers["draw_landmarks"]


class TestPreProcessLandmark:
    """Tests for the vectorized pre_process_landmark function."""

    def test_basic_landmarks(self):
        landmarks = [[100, 200], [110, 210], [120, 220]]
        result = pre_process_landmark(landmarks)
        assert result[0] == 0.0
        assert result[1] == 0.0
        assert len(result) == 6

    def test_normalization(self):
        landmarks = [[0, 0], [10, 20], [5, 15]]
        result = pre_process_landmark(landmarks)
        max_abs = max(abs(v) for v in result)
        assert max_abs == pytest.approx(1.0)

    def test_single_point(self):
        landmarks = [[50, 50]]
        result = pre_process_landmark(landmarks)
        assert result == [0.0, 0.0]

    def test_21_landmarks(self):
        landmarks = [[i * 10, i * 5] for i in range(21)]
        result = pre_process_landmark(landmarks)
        assert len(result) == 42

    def test_does_not_modify_input(self):
        landmarks = [[100, 200], [110, 210]]
        original = [row[:] for row in landmarks]
        pre_process_landmark(landmarks)
        assert landmarks == original


class TestPreProcessPointHistory:
    """Tests for the vectorized pre_process_point_history function."""

    def test_basic_history(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        history = deque([[320, 240], [330, 250], [340, 260]], maxlen=16)
        result = pre_process_point_history(image, history)
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.0)
        assert len(result) == 6

    def test_empty_history(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        history = deque(maxlen=16)
        result = pre_process_point_history(image, history)
        assert result == []

    def test_normalization_by_image_size(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        history = deque([[0, 0], [200, 100]], maxlen=16)
        result = pre_process_point_history(image, history)
        assert result[2] == pytest.approx(1.0)
        assert result[3] == pytest.approx(1.0)


class TestSelectMode:
    """Tests for the select_mode function."""

    def test_digit_keys(self):
        for digit in range(10):
            number, mode = select_mode(48 + digit, 0)
            assert number == digit

    def test_mode_switch_n(self):
        _, mode = select_mode(110, 1)
        assert mode == 0

    def test_mode_switch_k(self):
        _, mode = select_mode(107, 0)
        assert mode == 1

    def test_mode_switch_h(self):
        _, mode = select_mode(104, 0)
        assert mode == 2

    def test_non_special_key(self):
        number, mode = select_mode(65, 1)
        assert number == -1
        assert mode == 1


class TestCalcBoundingRect:
    """Tests for the vectorized calc_bounding_rect function."""

    def _make_landmarks(self, points, image_shape):
        image_height, image_width = image_shape[:2]
        mock = MagicMock()
        mock_landmarks = []
        for x, y in points:
            lm = MagicMock()
            lm.x = x / image_width
            lm.y = y / image_height
            mock_landmarks.append(lm)
        mock.landmark = mock_landmarks
        return mock

    def test_simple_rect(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        landmarks = self._make_landmarks(
            [(100, 100), (200, 200), (150, 150)], image.shape
        )
        result = calc_bounding_rect(image, landmarks)
        assert result[0] <= 100
        assert result[1] <= 100
        assert result[2] >= 200
        assert result[3] >= 200

    def test_returns_list_of_four(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        landmarks = self._make_landmarks([(50, 50)], image.shape)
        result = calc_bounding_rect(image, landmarks)
        assert len(result) == 4


class TestDrawLandmarks:
    """Tests for the refactored draw_landmarks function."""

    def test_returns_image(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        landmark_point = [[i * 10, i * 10] for i in range(21)]
        result = draw_landmarks(image, landmark_point)
        assert result is image

    def test_empty_landmarks(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = draw_landmarks(image, [])
        assert result is image

    def test_draws_on_image(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        landmark_point = [[100 + i * 10, 100 + i * 5] for i in range(21)]
        result = draw_landmarks(image, landmark_point)
        assert np.any(result > 0)
