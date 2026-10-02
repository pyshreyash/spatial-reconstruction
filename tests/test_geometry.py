import numpy as np

from roomscan.geometry.frame import manhattan_yaw, yaw_rotation
from roomscan.geometry.freespace import FreeSpaceGrid
from roomscan.geometry.polygon import _mincut, _trace_loops, build_complex
from roomscan.geometry.rooms import build_rooms, find_openings, line_profiles, segment_cells
from roomscan.geometry.walls import WallPlane


def test_manhattan_yaw_recovers_rotation():
    theta = np.radians(23.0)
    phis = np.concatenate([np.full(100, a) for a in (0, np.pi / 2, np.pi, -np.pi / 2)]) + theta
    n = np.stack([np.cos(phis), np.zeros_like(phis), np.sin(phis)], 1)
    est = manhattan_yaw(n)
    assert abs(est - theta) < 1e-6
    aligned = n @ yaw_rotation(est).T
    assert np.allclose(np.abs(aligned).max(1), 1.0)


def test_mincut_respects_unaries_and_smoothing():
    inside = _mincut(np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 1.0]), [])
    assert inside.tolist() == [True, False, True]
    # Strong smoothing pulls the weak middle cell inside.
    inside = _mincut(np.array([0.0, 0.2, 0.0]), np.array([1.0, 0.0, 1.0]), [(0, 1, 1.0), (1, 2, 1.0)])
    assert inside.all()


def test_trace_l_shape():
    mask = np.array([[1, 1], [1, 0]], bool)
    (loop,) = _trace_loops(mask)
    assert len(loop) == 6


def _wall(axis, facing, offset, lo, hi, gap=None):
    along = np.arange(lo, hi, 0.01)
    if gap:
        along = along[(along < gap[0]) | (along > gap[1])]
    along = np.repeat(along, 3)
    return WallPlane(axis, facing, offset, 0.005, len(along), (lo, hi), along)


def _grid(x1, z1):
    res = 0.02
    free = np.zeros((int((x1 + 1) / res), int((z1 + 1) / res)), np.int64)
    free[25:25 + int(x1 / res), 25:25 + int(z1 / res)] = 10  # origin at -0.5 m
    return FreeSpaceGrid(-0.5, -0.5, res, free, np.zeros_like(free))


def _pipeline(walls, grid):
    cx = build_complex(walls, grid)
    prof = line_profiles(cx)
    lab = segment_cells(cx, prof)
    rooms = build_rooms(cx, lab)
    return rooms, find_openings(cx, lab, rooms, prof)


def test_rectangle_room_dimensions():
    walls = [
        _wall("x", 1, 0.0, 0.0, 3.0), _wall("x", -1, 4.0, 0.0, 3.0),
        _wall("z", 1, 0.0, 0.0, 4.0), _wall("z", -1, 3.0, 0.0, 4.0),
    ]
    (room,), (openings, _) = _pipeline(walls, _grid(4.0, 3.0))
    assert abs(room.area - 12.0) < 1e-9
    assert sorted(round(w.length, 6) for w in room.walls) == [3.0, 3.0, 4.0, 4.0]
    assert openings == []


def test_partition_with_door_splits_rooms():
    # 4 x 3 m with a partition at x = 2.5 and a 0.8 m door at z 1.0..1.8 (seen from both sides).
    walls = [
        _wall("x", 1, 0.0, 0.0, 3.0), _wall("x", -1, 4.0, 0.0, 3.0),
        _wall("z", 1, 0.0, 0.0, 4.0), _wall("z", -1, 3.0, 0.0, 4.0),
        _wall("x", -1, 2.5, 0.0, 3.0, gap=(1.0, 1.8)),
    ]
    rooms, (openings, adjacency) = _pipeline(walls, _grid(4.0, 3.0))
    assert sorted(round(r.area, 6) for r in rooms) == [4.5, 7.5]
    (door,) = openings
    assert door.kind == "door" and abs(door.width - 0.8) <= 0.04
    assert [a.via for a in adjacency] == ["door"]
