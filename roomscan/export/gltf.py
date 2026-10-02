"""Minimal binary glTF 2.0 (.glb) writer: one mesh, one flat-coloured primitive per material."""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963
FLOAT = 5126
UNSIGNED_INT = 5125


@dataclass
class Primitive:
    name: str
    positions: np.ndarray  # (N,3) metres, +y up
    triangles: np.ndarray  # (M,3) vertex indices
    rgba: tuple[float, float, float, float]


def write_glb(path: Path, primitives: list[Primitive]) -> None:
    binary = bytearray()
    views, accessors, materials, prims = [], [], [], []

    def add_view(data: bytes, target: int) -> int:
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(data), "target": target})
        binary.extend(data)
        return len(views) - 1

    for p in primitives:
        if len(p.triangles) == 0:
            continue
        pos = np.ascontiguousarray(p.positions, dtype="<f4")
        idx = np.ascontiguousarray(p.triangles, dtype="<u4").ravel()
        accessors.append({
            "bufferView": add_view(pos.tobytes(), ARRAY_BUFFER), "componentType": FLOAT,
            "count": len(pos), "type": "VEC3", "min": pos.min(0).tolist(), "max": pos.max(0).tolist(),
        })
        accessors.append({
            "bufferView": add_view(idx.tobytes(), ELEMENT_ARRAY_BUFFER), "componentType": UNSIGNED_INT,
            "count": len(idx), "type": "SCALAR",
        })
        materials.append({
            "name": p.name,
            "pbrMetallicRoughness": {"baseColorFactor": list(p.rgba), "metallicFactor": 0.0, "roughnessFactor": 1.0},
            "doubleSided": True,
            **({"alphaMode": "BLEND"} if p.rgba[3] < 1 else {}),
        })
        prims.append({"attributes": {"POSITION": len(accessors) - 2}, "indices": len(accessors) - 1,
                      "material": len(materials) - 1})

    doc = {
        "asset": {"version": "2.0", "generator": "roomscan"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "floorplan"}],
        "meshes": [{"primitives": prims}],
        "materials": materials,
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": views,
        "accessors": accessors,
    }
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    binary.extend(b"\0" * (-len(binary) % 4))
    total = 12 + 8 + len(js) + 8 + len(binary)
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total))
        f.write(struct.pack("<I4s", len(js), b"JSON") + js)
        f.write(struct.pack("<I4s", len(binary), b"BIN\0") + bytes(binary))
