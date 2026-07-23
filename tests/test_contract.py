import numpy as np
import pytest

from injectr import contract


def test_load_base_valid(base_npz):
    sample = contract.load_base(base_npz)
    assert sample["label"] == "null"
    assert sample["tic_id"] == 123456
    assert sample["time"].shape == sample["flux"].shape == sample["flux_err"].shape


def test_load_base_missing_file(tmp_path):
    with pytest.raises(contract.ContractError):
        contract.load_base(tmp_path / "nope.npz")


def test_load_base_missing_required_array(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, time=np.arange(10.0), flux=np.zeros(10),
              tic_id=1, label="null", sector=1)  # missing flux_err
    with pytest.raises(contract.ContractError):
        contract.load_base(path)


def test_load_base_invalid_label(tmp_path):
    path = tmp_path / "bad_label.npz"
    np.savez(path, time=np.arange(10.0), flux=np.zeros(10), flux_err=np.ones(10),
              tic_id=1, label="not_a_real_label", sector=1)
    with pytest.raises(contract.ContractError):
        contract.load_base(path)


def test_write_injected_round_trip(base_npz, tmp_path):
    sample = contract.load_base(base_npz)
    out_path = tmp_path / "out.npz"
    contract.write_injected(
        out_path, time=sample["time"], flux=sample["flux"], flux_err=sample["flux_err"],
        label="planet", injection_params={"class": "planet", "period": 1.0},
        base_meta=sample,
    )
    reloaded = contract.load_base(out_path)
    assert reloaded["label"] == "planet"
    assert reloaded["tic_id"] == sample["tic_id"]
    assert reloaded["augmented"] == True  # noqa: E712
    assert reloaded["injection_params"] == {"class": "planet", "period": 1.0}


def test_write_injected_writes_exactly_the_requested_path(base_npz, tmp_path):
    # np.savez(path, ...) silently appends ".npz" to any path that doesn't
    # already end in it -- a real risk here since write_injected stages
    # through a ".tmp"-suffixed tmp file before the atomic rename. Confirm
    # the final file lands at exactly out_path, with nothing else left over.
    sample = contract.load_base(base_npz)
    out_path = tmp_path / "out.npz"
    result_path = contract.write_injected(
        out_path, time=sample["time"], flux=sample["flux"], flux_err=sample["flux_err"],
        label="planet", injection_params={"class": "planet", "period": 1.0},
        base_meta=sample,
    )
    assert result_path == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0
    assert not (tmp_path / "out.npz.npz").exists()  # not the wrong .npz-appended path
    assert list(tmp_path.glob("*.tmp")) == []


def test_write_injected_overwrites_a_corrupt_existing_file(base_npz, tmp_path):
    # write_injected must overwrite atomically even when a previous crash
    # left a corrupt/truncated file at the destination -- not fail, and
    # not leave the corrupt bytes in place.
    sample = contract.load_base(base_npz)
    out_path = tmp_path / "out.npz"
    out_path.write_bytes(b"not a real npz file")

    contract.write_injected(
        out_path, time=sample["time"], flux=sample["flux"], flux_err=sample["flux_err"],
        label="planet", injection_params={"class": "planet", "period": 1.0},
        base_meta=sample,
    )
    reloaded = contract.load_base(out_path)
    assert reloaded["label"] == "planet"
    assert list(tmp_path.glob("*.tmp")) == []  # no leftover tmp file
