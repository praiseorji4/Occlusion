"""Split-protocol tests: KITTI chronological and CityPersons city-level non-overlap."""
from __future__ import annotations

from occlusion.data.kitti_labels import KITTI_TRAIN_END, KITTI_VAL_END, split_for_id


def test_kitti_split_boundaries():
    assert KITTI_TRAIN_END < KITTI_VAL_END
    assert split_for_id(0) == "train"
    assert split_for_id(KITTI_TRAIN_END - 1) == "train"
    assert split_for_id(KITTI_TRAIN_END) == "val"
    assert split_for_id(KITTI_VAL_END - 1) == "val"
    assert split_for_id(KITTI_VAL_END) == "test"


def test_kitti_splits_disjoint():
    train = {i for i in range(0, KITTI_TRAIN_END)}
    val = {i for i in range(KITTI_TRAIN_END, KITTI_VAL_END)}
    test = {i for i in range(KITTI_VAL_END, 7481)}
    assert not (train & val) and not (train & test) and not (val & test)


def test_citypersons_city_sets_disjoint():
    import data.split_verification as sv

    sets = [sv.CP_TRAIN_CITIES, sv.CP_VAL_CITIES, sv.CP_TEST_CITIES]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            assert not (sets[i] & sets[j]), f"CP split overlap between {i} and {j}"
    assert len(sv.CP_TRAIN_CITIES) == 18
    assert len(sv.CP_VAL_CITIES) == 3
    assert len(sv.CP_TEST_CITIES) == 6
