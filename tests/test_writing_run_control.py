from pathlib import Path

from veriwrite_agent.services.writing_run_control import WritingRunControlStore


def test_pause_is_shared_and_blocks_new_executor(tmp_path: Path) -> None:
    first = WritingRunControlStore(tmp_path, project_key="paper_alpha")
    second = WritingRunControlStore(tmp_path, project_key="paper_alpha")

    first.pause()

    assert second.is_paused() is True
    assert second.try_acquire("second-window") is None

    second.resume()
    lease = first.try_acquire("first-window")

    assert lease is not None
    assert second.try_acquire("second-window") is None
    assert lease.continue_allowed() is True
    lease.release()
    assert second.try_acquire("second-window") is not None


def test_pause_interrupts_an_existing_lease_at_next_boundary(tmp_path: Path) -> None:
    first = WritingRunControlStore(tmp_path, project_key="paper_alpha")
    second = WritingRunControlStore(tmp_path, project_key="paper_alpha")
    lease = first.try_acquire("first-window")

    assert lease is not None
    second.pause()

    assert lease.continue_allowed() is False
    lease.release()
