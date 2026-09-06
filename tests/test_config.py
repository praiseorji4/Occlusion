from pathlib import Path

import pytest

from occlusion.config import AUG_STRATEGIES, FUSION_TYPES, load_config

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def _all_experiments():
    return sorted((CONFIGS / "experiment").glob("*.yaml"))


@pytest.mark.parametrize("path", _all_experiments())
def test_experiment_loads(path):
    cfg = load_config(path)
    assert cfg.model in {"yolov8x", "yolov8n"}
    assert cfg.dataset in {"kitti", "citypersons"}
    assert cfg.fusion_type in FUSION_TYPES
    assert cfg.aug_strategy in AUG_STRATEGIES
    assert cfg.epochs > 0
    assert cfg.batch > 0
    assert cfg.nc == len(cfg.names)


def test_base_inheritance_kitti_hyperparams():
    cfg = load_config(CONFIGS / "experiment" / "m0_kitti.yaml")
    # KITTI W&B-sweep values must propagate from dataset/kitti.yaml.
    assert cfg.lr0 == pytest.approx(0.0013018964408270848)
    assert cfg.optimizer == "SGD"
    assert cfg.nc == 3
    assert cfg.names[1] == "Pedestrian"


def test_base_inheritance_citypersons():
    cfg = load_config(CONFIGS / "experiment" / "m0_citypersons.yaml")
    assert cfg.dataset == "citypersons"
    assert cfg.nc == 1
    assert cfg.batch == 4
    assert cfg.lr0 == pytest.approx(0.000842484306662393)


def test_overrides_win():
    cfg = load_config(CONFIGS / "experiment" / "m0_kitti.yaml", {"seed": 123, "epochs": 1})
    assert cfg.seed == 123
    assert cfg.epochs == 1


def test_uses_depth_flag():
    assert load_config(CONFIGS / "experiment" / "m0_kitti.yaml").uses_depth is False
    assert load_config(CONFIGS / "experiment" / "latefusion_kitti.yaml").uses_depth is True
    assert load_config(CONFIGS / "experiment" / "full_system_kitti.yaml").uses_depth is True


def test_invalid_fusion_rejected():
    with pytest.raises(ValueError):
        load_config(CONFIGS / "experiment" / "m0_kitti.yaml", {"fusion_type": "bogus"})


def test_run_name_stable():
    cfg = load_config(CONFIGS / "experiment" / "m0_kitti.yaml")
    assert cfg.run_name == f"m0_baseline_kitti_yolov8x_seed{cfg.seed}"
