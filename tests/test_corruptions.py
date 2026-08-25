import numpy as np
import pytest

from gpu_corruptnet.corruptions import apply, available
from gpu_corruptnet.corruptions.base import ARTIFACT_CLASSES
from gpu_corruptnet.data import make_demo_frame


@pytest.fixture
def frame() -> np.ndarray:
    return make_demo_frame(h=120, w=200, rng=0)


def test_at_least_six_injectors_registered():
    assert len(available()) >= 6


def test_all_registered_names_are_valid_classes():
    assert set(available()).issubset(set(ARTIFACT_CLASSES))


def test_all_ten_classes_registered_when_opencv_present():
    pytest.importorskip("cv2")
    assert set(available()) == set(ARTIFACT_CLASSES)


@pytest.mark.parametrize("name", available())
@pytest.mark.parametrize("severity", [1, 3, 5])
def test_injector_preserves_shape_and_dtype(frame, name, severity):
    out = apply(name, frame, severity=severity, rng=0)
    assert out.shape == frame.shape
    assert out.dtype == np.uint8


@pytest.mark.parametrize("name", available())
def test_injector_actually_changes_the_image(frame, name):
    out = apply(name, frame, severity=5, rng=0)
    assert np.any(out != frame), f"{name} left the frame unchanged"


@pytest.mark.parametrize("name", available())
def test_injector_does_not_mutate_input(frame, name):
    before = frame.copy()
    apply(name, frame, severity=4, rng=0)
    assert np.array_equal(frame, before), f"{name} mutated its input"


@pytest.mark.parametrize("name", available())
def test_injector_is_deterministic_given_seed(frame, name):
    a = apply(name, frame, severity=3, rng=42)
    b = apply(name, frame, severity=3, rng=42)
    assert np.array_equal(a, b)


def test_invalid_severity_rejected(frame):
    with pytest.raises(ValueError):
        apply(available()[0], frame, severity=9)


def test_unknown_corruption_raises(frame):
    with pytest.raises(KeyError):
        apply("does_not_exist", frame)
