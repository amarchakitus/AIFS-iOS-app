"""Generate the Liquid Glass app icon (`AIFS-iOS-app/AppIcon.icon`, an Icon Composer document).

Two flat layers on a gradient, with the glass rendering left to the system:

  * a geodesic sphere -- a frequency-2 icosahedron, the kind of mesh the model's graph
    network runs on -- as a faceted glass globe;
  * the meteorological tropical-cyclone symbol in front of it.

Run with any Python 3 (stdlib only):  python3 tools/make_icon.py
Preview:  ictool AIFS-iOS-app/AppIcon.icon --export-image --output-file icon.png \\
            --platform iOS --rendition Default --width 1024 --height 1024 --scale 1
"""

import itertools
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "AIFS-iOS-app" / "AppIcon.icon"
SIZE = 1024
CX, CY, R = 512, 520, 360  # globe centre and radius, SVG coordinates (y down)

GLOBE = "#EAF2FD"
MESH = "#1C5CAB"
MESH_DARK = "#9EC5F4"
CYCLONE = "#F0705F"


# --- geodesic sphere ---------------------------------------------------------------------


def normalize(p):
    n = math.sqrt(sum(c * c for c in p))
    return tuple(c / n for c in p)


def icosphere():
    """Frequency-2 geodesic sphere: icosahedron with each face split in four."""
    t = (1 + 5**0.5) / 2
    verts = [normalize(v) for v in [
        (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
        (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
        (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1),
    ]]
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]
    cache = {}

    def mid(a, b):
        key = (min(a, b), max(a, b))
        if key not in cache:
            verts.append(normalize(tuple((x + y) / 2 for x, y in zip(verts[a], verts[b], strict=True))))
            cache[key] = len(verts) - 1
        return cache[key]

    out = []
    for a, b, c in faces:
        ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
        out += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
    return verts, out


def rotate(p, yaw, pitch):
    x, y, z = p
    x, z = x * math.cos(yaw) + z * math.sin(yaw), -x * math.sin(yaw) + z * math.cos(yaw)
    y, z = y * math.cos(pitch) - z * math.sin(pitch), y * math.sin(pitch) + z * math.cos(pitch)
    return x, y, z


def globe_svgs():
    verts, faces = icosphere()
    verts = [rotate(v, 0.35, 0.45) for v in verts]

    def xy(v):
        return CX + v[0] * R, CY - v[1] * R

    edges = set()
    for f in faces:
        # Only faces turned toward the viewer, so the wireframe reads as a solid object.
        if sum(verts[i][2] for i in f) / 3 <= 0.05:
            continue
        for a, b in itertools.combinations(f, 2):
            edges.add((min(a, b), max(a, b)))
    lines = "\n".join(
        '<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}"/>'.format(*xy(verts[a]), *xy(verts[b]))
        for a, b in sorted(edges)
    )
    globe = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{SIZE}" height="{SIZE}" viewBox="0 0 {SIZE} {SIZE}">
<circle cx="{CX}" cy="{CY}" r="{R}" fill="{GLOBE}"/>
</svg>
"""
    mesh = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{SIZE}" height="{SIZE}" viewBox="0 0 {SIZE} {SIZE}">
<defs><clipPath id="disc"><circle cx="{CX}" cy="{CY}" r="{R - 1}"/></clipPath></defs>
<g clip-path="url(#disc)" stroke="{MESH}" stroke-width="11" stroke-linecap="round" fill="none">
{lines}
</g>
</svg>
"""
    return globe, mesh


# --- cyclone symbol ----------------------------------------------------------------------


def cyclone_svg():
    """Ring with two hooked arms, as on a weather map, curling counter-clockwise."""
    ex, ey = CX + 30, CY - 20  # eye, slightly off-centre for depth against the globe
    ring_out, ring_in = 122, 62
    parts = [
        # ring as one path; the inner circle runs the other way, so nonzero fill cuts it out
        f"M {ex + ring_out} {ey} A {ring_out} {ring_out} 0 1 0 {ex - ring_out} {ey} "
        f"A {ring_out} {ring_out} 0 1 0 {ex + ring_out} {ey} Z "
        f"M {ex + ring_in} {ey} A {ring_in} {ring_in} 0 1 1 {ex - ring_in} {ey} "
        f"A {ring_in} {ring_in} 0 1 1 {ex + ring_in} {ey} Z"
    ]
    # Two hooks leaving the ring from top and bottom, each sweeping a quarter-turn
    # counter-clockwise as it widens out -- the map symbol for a tropical cyclone.
    mid_r, half0 = (ring_out + ring_in) / 2, (ring_out - ring_in) / 2
    sweep, grow = 0.62 * math.pi, 190
    for arm in range(2):
        phase = math.pi / 2 + arm * math.pi
        left, right = [], []
        steps = 80
        for s in range(steps + 1):
            t = s / steps
            theta = phase + t * sweep
            rad = mid_r + grow * t**1.3
            # SVG y is down: negate sin so the sweep is counter-clockwise on screen
            c = (ex + math.cos(theta) * rad, ey - math.sin(theta) * rad)
            drad = grow * 1.3 * max(t, 1e-3) ** 0.3
            tx = drad * math.cos(theta) - rad * sweep * math.sin(theta)
            ty = -(drad * math.sin(theta) + rad * sweep * math.cos(theta))
            n = math.hypot(tx, ty)
            half = half0 * (1 - t) ** 0.85 + 0.5
            nx, ny = -ty / n * half, tx / n * half
            left.append((c[0] + nx, c[1] + ny))
            right.append((c[0] - nx, c[1] - ny))
        pts = left + right[::-1]
        parts.append("M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts) + " Z")
    d = " ".join(parts)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{SIZE}" height="{SIZE}" viewBox="0 0 {SIZE} {SIZE}">
<path d="{d}" fill="{CYCLONE}" fill-rule="nonzero"/>
</svg>
"""


# --- document ----------------------------------------------------------------------------


def srgb(hex_color, alpha=1.0):
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return f"srgb:{r:.5f},{g:.5f},{b:.5f},{alpha:.5f}"


def main():
    globe, mesh = globe_svgs()
    assets = OUT / "Assets"
    assets.mkdir(parents=True, exist_ok=True)
    (assets / "globe.svg").write_text(globe)
    (assets / "mesh.svg").write_text(mesh)
    (assets / "cyclone.svg").write_text(cyclone_svg())

    doc = {
        "fill": {"linear-gradient": [srgb("#2A78D6"), srgb("#0B2A55")]},
        # Groups and layers are listed front to back.
        "groups": [
            {
                "name": "cyclone",
                "layers": [
                    {
                        "name": "cyclone",
                        "image-name": "cyclone.svg",
                        "glass": True,
                        # Tinted icons are coloured from luminance; the coral is too close
                        # to the globe's there, so the cyclone goes white to stay legible.
                        "fill-specializations": [
                            {"value": {"solid": srgb(CYCLONE)}},
                            {"appearance": "tinted", "value": {"solid": srgb("#FFFFFF")}},
                        ],
                    }
                ],
                "shadow": {"kind": "layer-color", "opacity": 0.5},
                "translucency": {"enabled": True, "value": 0.25},
                "specular": True,
            },
            {
                "name": "globe",
                "layers": [
                    {
                        "name": "mesh",
                        "image-name": "mesh.svg",
                        "glass": False,
                        # In dark mode the system turns the globe deep blue, where the
                        # light-mode mesh blue vanishes; switch to a pale blue there.
                        "fill-specializations": [
                            {"value": {"solid": srgb(MESH)}},
                            {"appearance": "dark", "value": {"solid": srgb(MESH_DARK)}},
                            # Tinted keeps only luminance and dims it further in dark mode, so the
                            # mesh needs a wide brightness gap to the globe beneath it.
                            {"appearance": "tinted", "value": {"solid": srgb("#D6D6D6")}},
                        ],
                        "opacity-specializations": [
                            {"value": 0.55},
                            {"appearance": "dark", "value": 0.8},
                            {"appearance": "tinted", "value": 0.9},
                        ],
                    },
                    {
                        "name": "globe",
                        "image-name": "globe.svg",
                        "glass": True,
                        # Tinted icons darken every layer in dark mode; dimming the globe
                        # keeps the cyclone the brightest shape in both tinted looks.
                        "fill-specializations": [
                            {"value": {"solid": srgb(GLOBE)}},
                            {"appearance": "tinted", "value": {"solid": srgb("#3C3C3C")}},
                        ],
                    },
                ],
                "shadow": {"kind": "neutral", "opacity": 0.5},
                "translucency": {"enabled": True, "value": 0.45},
                "specular": True,
            },
        ],
        "supported-platforms": {"squares": "shared"},
    }
    (OUT / "icon.json").write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
