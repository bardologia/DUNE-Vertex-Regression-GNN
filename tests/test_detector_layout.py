"""Tests for the documented module layout, the frame conversion and the mapping of the raw bins onto the modules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipelines.dataset.detector_layout import ApexModuleLayout, ChannelMapping, CoordinateTransform, RawBinMapping


@pytest.fixture(scope="module")
def layout() -> ApexModuleLayout:
    """Returns the documented module layout."""
    return ApexModuleLayout()


@pytest.fixture(scope="module")
def walls(layout) -> pd.DataFrame:
    """Returns the 5,760 documented wall modules."""
    return layout.wall_block()


@pytest.fixture(scope="module")
def endcaps(layout) -> pd.DataFrame:
    """Returns the 1,152 documented endcap modules."""
    return layout.endcap_block()


def _position(table: pd.DataFrame, module: int) -> tuple:
    """Returns the sensor-frame position of one module of a block table, rounded to 0.1 mm."""
    row = table.loc[table["module_index"] == module].iloc[0]
    return tuple(round(float(row[axis]), 4) for axis in ("x", "y", "z"))


def _face(table: pd.DataFrame, module: int) -> str:
    """Returns the face name of one module of a block table."""
    return str(table.loc[table["module_index"] == module, "face"].iloc[0])


def test_vertex_transform_maps_the_file_name_frame_to_the_sensor_frame():
    """Verifies the file-name vertex (x, y, z) becomes (y, -x, z) in the sensor frame."""
    assert CoordinateTransform().apply_to_target(1.0, 2.0, 3.0) == (2.0, -1.0, 3.0)


def test_module_positions_take_the_vertex_transform():
    """Verifies the module transform is the vertex transform, so the document's right side (X = +6.673 m) is the wall at y = -6.673 m."""
    simulation = np.array([[6.673, -6.0275, -29.75], [1.0, 2.0, 3.0]])
    sensor     = CoordinateTransform().apply_to_positions(simulation)

    assert np.allclose(sensor, [[-6.0275, -6.673, -29.75], [2.0, -1.0, 3.0]])
    assert tuple(sensor[1]) == CoordinateTransform().apply_to_target(1.0, 2.0, 3.0)


def test_drift_climbs_the_lower_volume_then_descends_the_upper_one(layout):
    """Verifies d(0) = -6.0275, d(11) = -0.4725, the wrap to d(12) = +6.0275 and d(23) = +0.4725, in steps of 0.505 m inside each half."""
    drift = layout.drift(np.arange(24))

    assert drift[0] == pytest.approx(-6.0275) and drift[11] == pytest.approx(-0.4725)
    assert drift[12] == pytest.approx(6.0275) and drift[23] == pytest.approx(0.4725)
    assert np.allclose(np.diff(drift[:12]), 0.505) and np.allclose(np.diff(drift[12:]), -0.505)
    assert drift[12] - drift[11] == pytest.approx(6.5)


@pytest.mark.parametrize("module, position, face", [
    (0,    (-6.0275, -6.673, -29.75), "right_wall"),
    (1,    (-6.0275,  6.673, -29.75), "left_wall"),
    (2,    (-5.5225, -6.673, -29.75), "right_wall"),
    (3,    (-5.5225,  6.673, -29.75), "left_wall"),
    (22,   (-0.4725, -6.673, -29.75), "right_wall"),
    (23,   (-0.4725,  6.673, -29.75), "left_wall"),
    (24,   ( 6.0275, -6.673, -29.75), "right_wall"),
    (25,   ( 6.0275,  6.673, -29.75), "left_wall"),
    (26,   ( 5.5225, -6.673, -29.75), "right_wall"),
    (46,   ( 0.4725, -6.673, -29.75), "right_wall"),
    (47,   ( 0.4725,  6.673, -29.75), "left_wall"),
    (48,   (-6.0275, -6.673, -29.25), "right_wall"),
    (5759, ( 0.4725,  6.673,  29.75), "left_wall"),
])
def test_wall_modules_sit_where_the_document_places_them(walls, module, position, face):
    """Verifies hand-picked wall modules of the document's text and figure, document number = index + 1: module 1 at the lowest drift and z on the right side, module 25 at the highest drift, 23 and 47 next to the cathode, 49 one ring up."""
    assert _position(walls, module) == pytest.approx(position)
    assert _face(walls, module) == face


@pytest.mark.parametrize("module, position, face", [
    (0,    (-6.0275,  5.8075,  29.9265), "front_endcap"),
    (1,    (-6.0275,  5.8075, -29.9265), "back_endcap"),
    (2,    (-5.5225,  5.8075,  29.9265), "front_endcap"),
    (24,   ( 6.0275,  5.8075,  29.9265), "front_endcap"),
    (48,   (-6.0275,  5.3025,  29.9265), "front_endcap"),
    (1151, ( 0.4725, -5.8075, -29.9265), "back_endcap"),
])
def test_endcap_modules_sit_where_the_document_places_them(endcaps, module, position, face):
    """Verifies hand-picked endcap modules: module 1 at the lowest drift, the leftmost X = -5.8075 m (sensor y = +5.8075 m) and the front face; the next batch at X = -5.3025 m."""
    assert _position(endcaps, module) == pytest.approx(position)
    assert _face(endcaps, module) == face


def test_module_table_is_6912_distinct_mirror_symmetric_positions(layout):
    """Verifies the documented layout holds 5,760 wall and 1,152 endcap modules at distinct positions, a set closed under the three reflections."""
    table     = layout.module_table()
    positions = np.round(table[["x", "y", "z"]].to_numpy(), 4)
    present   = {tuple(row) for row in positions.tolist()}

    assert (table["block"] == "wall").sum() == 5760 and (table["block"] == "endcap").sum() == 1152
    assert len(present) == 6912

    for signs in ((-1, 1, 1), (1, -1, 1), (1, 1, -1)):
        assert {tuple(row) for row in np.round(positions * np.array(signs), 4).tolist()} == present


def test_raw_bins_are_the_real_bins():
    """Verifies the expected raw bins are 12, 13, 15 to 5772 and 5774 to 6925, 6,912 of them."""
    bins = RawBinMapping().raw_bins()

    assert bins.shape == (6912,)
    assert bins[:3].tolist() == [12, 13, 15]
    assert 14 not in bins and 5773 not in bins
    assert 5772 in bins and 5774 in bins and bins[-1] == 6925


@pytest.mark.parametrize("raw_bin, block, module", [
    (12,   "wall",       1),
    (15,   "wall",       3),
    (5771, "wall",       5759),
    (5774, "endcap",     1),
    (6924, "endcap",     1151),
    (13,   "face_total", -1),
    (5772, "face_total", -1),
    (6925, "face_total", -1),
])
def test_bins_map_onto_modules(raw_bin, block, module):
    """Verifies the wall offset 12, the endcap offset 5773, the bin 12 exception and the three face totals."""
    blocks, modules = RawBinMapping().assign(np.array([raw_bin]))

    assert blocks[0] == block
    assert modules[0] == module


@pytest.mark.parametrize("raw_bin", [11, 14, 5773, 6926])
def test_bins_outside_the_numbering_are_refused(raw_bin):
    """Verifies a bin that is neither a module, the lead bin nor a face total raises."""
    with pytest.raises(ValueError, match="belong to no module block"):
        RawBinMapping().assign(np.array([raw_bin]))


def test_bin_12_holds_the_far_wall_module_against_the_document(walls):
    """Verifies bin 12 is placed on wall module 1, the far wall at y = +6.673 m, where the offset rule of the document would give module 0 on the near wall."""
    blocks, modules = RawBinMapping().assign(np.array([12]))

    assert (blocks[0], modules[0]) == ("wall", 1)
    assert _position(walls, 1) == pytest.approx((-6.0275, 6.673, -29.75))
    assert _position(walls, 12 - RawBinMapping.WALL_OFFSET) == pytest.approx((-6.0275, -6.673, -29.75))


def test_mapping_rebuilds_the_light_derived_geometry(raw_frame, light_derived, quiet_logger):
    """Verifies the mapping of a real raw file places every kept raw row where the light-derived rule of the sweep puts it: even rows at P(k), odd rows at P(k + 2)."""
    geometry = ChannelMapping(quiet_logger).build(raw_frame)

    assert tuple(geometry.columns) == ChannelMapping.GEOMETRY_COLUMNS
    assert np.array_equal(geometry["channel_index"].to_numpy(), np.arange(6909))
    assert np.array_equal(geometry["raw_channel_index"].to_numpy(), light_derived.kept_rows)
    assert np.array_equal(geometry["bin"].to_numpy(), light_derived.bins)
    assert np.abs(geometry[["x", "y", "z"]].to_numpy() - light_derived.positions).max() < 1e-9


def test_mapping_removes_the_face_totals_and_keeps_raw_order(raw_frame, quiet_logger):
    """Verifies the face-total bins and their raw rows 1, 5759 and 6911 are absent and the kept rows stay in raw order."""
    geometry = ChannelMapping(quiet_logger).build(raw_frame)
    rows     = geometry["raw_channel_index"].to_numpy()

    assert not set(geometry["bin"].tolist()) & {13, 5772, 6925}
    assert not set(rows.tolist()) & {1, 5759, 6911}
    assert (np.diff(rows) > 0).all()


def test_mapping_leaves_wall_modules_0_and_2_and_endcap_module_0_empty(raw_frame, layout, quiet_logger):
    """Verifies the modules without a channel are wall 0 and 2 and endcap 0, the positions P(1), P(3) and P(5761) of the 2024 table, and every channel sits on its own module."""
    geometry = ChannelMapping(quiet_logger).build(raw_frame)
    on_wall  = geometry["face"].isin(ApexModuleLayout.WALL_FACES).to_numpy()

    assert sorted(set(range(5760)) - set(geometry.loc[on_wall, "module_index"].tolist())) == [0, 2]
    assert sorted(set(range(1152)) - set(geometry.loc[~on_wall, "module_index"].tolist())) == [0]
    assert np.bincount(on_wall.astype(np.int64)).tolist() == [1151, 5758]

    table   = layout.module_table()
    keys    = table.set_index(["block", "module_index"])
    blocks  = np.where(on_wall, "wall", "endcap")
    located = keys.loc[list(zip(blocks, geometry["module_index"].to_numpy()))]
    assert np.allclose(located[["x", "y", "z"]].to_numpy(), geometry[["x", "y", "z"]].to_numpy())

    empty = np.round(np.array([_position(table[table["block"] == "wall"], 0), _position(table[table["block"] == "wall"], 2), _position(table[table["block"] == "endcap"], 0)]), 4)
    assert empty.tolist() == [[-6.0275, -6.673, -29.75], [-5.5225, -6.673, -29.75], [-6.0275, 5.8075, 29.9265]]


def test_mapping_ignores_the_raw_positions(raw_frame, light_derived, quiet_logger):
    """Verifies two raw layouts that carry the same position set in another row order give the same geometry, since positions come from the bins."""
    order                     = np.random.default_rng(3).permutation(len(raw_frame))
    permuted                  = raw_frame.copy()
    permuted[["x", "y", "z"]] = raw_frame[["x", "y", "z"]].to_numpy()[order]

    geometry = ChannelMapping(quiet_logger).build(permuted)

    assert np.abs(geometry[["x", "y", "z"]].to_numpy() - light_derived.positions).max() < 1e-9


def test_mapping_refuses_a_missing_or_extra_bin(raw_frame, quiet_logger):
    """Verifies bins other than exactly 12, 13, 15..5772 and 5774..6925 raise, whether one is replaced, added or removed."""
    replaced               = raw_frame.copy()
    replaced.loc[2, "bin"] = 14.0
    removed                = raw_frame.iloc[1:].reset_index(drop=True)
    added                  = pd.concat([raw_frame, raw_frame.iloc[[0]]], ignore_index=True)

    for frame in (replaced, removed, added):
        with pytest.raises(ValueError, match="raw bins are not exactly"):
            ChannelMapping(quiet_logger).build(frame)


def test_mapping_refuses_a_fractional_bin(raw_frame, quiet_logger):
    """Verifies a non-integer bin raises."""
    broken                = raw_frame.copy()
    broken.loc[5, "bin"] += 0.5

    with pytest.raises(ValueError, match="non-integer"):
        ChannelMapping(quiet_logger).build(broken)


def test_mapping_refuses_raw_positions_outside_the_documented_set(raw_frame, quiet_logger):
    """Verifies a raw layout whose position set differs from the 6,912 documented modules raises."""
    broken              = raw_frame.copy()
    broken.loc[40, "x"] = broken.loc[40, "x"] + 0.1

    with pytest.raises(ValueError, match="set of raw positions differs"):
        ChannelMapping(quiet_logger).build(broken)


def test_mapping_refuses_a_channel_count_other_than_6909(raw_frame, quiet_logger, monkeypatch):
    """Verifies the count check fires when the expected channel count disagrees with the mapping."""
    monkeypatch.setattr(ChannelMapping, "CHANNEL_COUNT", 6910)

    with pytest.raises(ValueError, match="keeps 6909 channels, expected 6910"):
        ChannelMapping(quiet_logger).build(raw_frame)


def test_mapping_refuses_two_channels_on_one_position(raw_frame, quiet_logger, monkeypatch):
    """Verifies the distinct-position check fires when the lead bin is sent to wall module 3, which bin 15 already holds."""
    monkeypatch.setattr(RawBinMapping, "LEAD_MODULE", 3)

    with pytest.raises(ValueError, match="only 6908 distinct positions"):
        ChannelMapping(quiet_logger).build(raw_frame)


def test_mapping_refuses_another_set_of_empty_modules(raw_frame, quiet_logger, monkeypatch):
    """Verifies the empty-module check fires when the expected empty modules disagree with the mapping."""
    monkeypatch.setattr(RawBinMapping, "EMPTY_WALL_MODULES", (0,))

    with pytest.raises(ValueError, match="modules without a channel are wall"):
        ChannelMapping(quiet_logger).build(raw_frame)
