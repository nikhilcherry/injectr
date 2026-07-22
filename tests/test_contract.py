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
