# tests/test_core.py
import pytest

from gradient.ai.diffusion.guidance import is_constant, parse_guidance_schedule
from gradient.core import config


def test_env_overrides_paths(workspace):
    assert config.get_model_search_dirs()[0] == (workspace / "models").resolve()
    assert config.get_outputs_dir() == (workspace / "experiments").resolve()
    assert config.get_data_dir() == (workspace / "data").resolve()


def test_get_setting_defaults():
    assert config.get_setting("app.port") == 8000
    assert config.get_setting("does.not.exist", "x") == "x"
    assert config.get_setting("performance.memory_mode") == "auto"


@pytest.mark.parametrize(
    "spec,expected",
    [
        (7.5, [7.5] * 4),
        ("6", [6.0] * 4),
        ("linear(9, 3)", [9.0, 7.0, 5.0, 3.0]),
        ("9@0,3@3", [9.0, 7.0, 5.0, 3.0]),
        ([[0, 9.0], [3, 3.0]], [9.0, 7.0, 5.0, 3.0]),
        ("9@0, 5@2", [9.0, 7.0, 5.0, 5.0]),
    ],
)
def test_guidance_schedules(spec, expected):
    assert parse_guidance_schedule(spec, 4) == pytest.approx(expected)


def test_cosine_schedule_endpoints():
    vals = parse_guidance_schedule("cosine(10, 2)", 5)
    assert vals[0] == pytest.approx(10) and vals[-1] == pytest.approx(2)
    assert not is_constant(vals)


@pytest.mark.parametrize("bad", ["zigzag(1,2)", "7@x", [[1]], "linear(1)"])
def test_guidance_schedule_errors(bad):
    with pytest.raises(ValueError):
        parse_guidance_schedule(bad, 4)


async def test_gpu_tasks_are_serialized():
    import threading
    import time

    from gradient.core.scheduling import run_gpu_task

    active = []
    peak = []
    lock = threading.Lock()

    def work():
        with lock:
            active.append(1)
            peak.append(len(active))
        time.sleep(0.05)
        with lock:
            active.pop()
        return threading.current_thread().name

    import asyncio

    names = await asyncio.gather(*(run_gpu_task(work) for _ in range(4)))
    assert max(peak) == 1
    assert all(n != threading.main_thread().name for n in names)
