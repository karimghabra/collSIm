"""Job bookkeeping: a window listed again with a longer run is an extension."""
import json

from aamd import umbrella


def test_extension_is_unfinished_until_the_longer_run_is_done(tmp_path, monkeypatch):
    monkeypatch.setattr(umbrella, "workdir", lambda name: tmp_path / name)
    short, longer = ["window", "sys", "0"], ["window", "sys", "0", "16"]
    contact = ["contact", "sys", "1.300", "4.0"]
    assert umbrella.job_target_ns(short) == umbrella.PROTOCOL.prod_ns
    assert umbrella.job_target_ns(longer) == 16.0
    assert umbrella.job_target_ns(contact) == 4.0
    d = umbrella.job_dir(short)
    d.mkdir(parents=True)
    assert not umbrella.job_finished(short)
    (d / "done.json").write_text(json.dumps({"prod_ns": 4.0}))
    assert umbrella.job_finished(short)
    assert not umbrella.job_finished(longer)
    (d / "done.json").write_text(json.dumps({"prod_ns": 16.0}))
    assert umbrella.job_finished(longer)
    (d / "done.json").unlink()
    (d / "failed.json").write_text("{}")
    assert umbrella.job_finished(longer)          # a failed job is not retried
