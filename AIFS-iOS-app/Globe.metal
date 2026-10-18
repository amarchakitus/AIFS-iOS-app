// The globe: an orthographic sphere raycast per pixel, after AIFS-VIZ's
// (~/code/AIFS-VIZ/src/aifs_viz/macos/AIFSViz/Shaders.metal) -- same projection, halo,
// limb shaping, terminator, derivative-width coastline and wind-particle trails, with the
// same knobs. Fields arrive as values in display units, two lead times at once, blended
// in value space so playback is smooth; the colour scale is applied here.
//
// Every grid texture is 0.25-degree equirectangular: 1440 x 721, columns from 180W, rows
// from 90N, texel centres on the grid nodes.

#include <metal_stdlib>
using namespace metal;

struct GlobeUniforms {
    float3x3 invRotation;   // camera space -> earth space (columns: east, north, out)
    float2 center;          // disc centre, px, y-up
    float2 resolution;      // drawable px
    float4 sunDir;          // earth space, unit
    float radius;           // disc radius, px
    float blend;            // 0..1 from field0 to field1
    float hasField;
    float scaleKind;        // 0 sequential, 1 diverging, 2 gray
    float low;
    float mid;
    float high;
    float gamma;
    float night;            // basemap strengths, as in AIFS-VIZ's panel
    float relief;
    float coast;
    float graticule;
    float atmosphere;
    float stars;
    float opacity;          // field opacity over the land/ocean basemap
    float hasElevation;
};

struct VOut {
    float4 position [[position]];
    float2 px;              // px, y-up
};

static inline float2 gridUV(float lat, float lon) {
    return float2(((lon + 180.0) / 0.25 + 0.5) / 1440.0, ((90.0 - lat) / 0.25 + 0.5) / 721.0);
}

static inline float3 earth(float lat, float lon) {
    float p = lat * 0.017453292519943295, l = lon * 0.017453292519943295;
    return float3(cos(p) * cos(l), sin(p), cos(p) * sin(l));
}

static inline float hash12(float2 p) {
    float3 p3 = fract(float3(p.x, p.y, p.x) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

/// Mirrors ColorScale.position in AIFSKit (Display/Maps.swift).
static inline float scalePosition(float v, constant GlobeUniforms &u) {
    if (u.scaleKind > 1.5) return clamp((v - u.low) / (u.high - u.low), 0.0, 1.0);
    if (u.scaleKind > 0.5) {
        if (v < u.mid) return 0.5 * clamp((v - u.low) / (u.mid - u.low), 0.0, 1.0);
        return 0.5 + 0.5 * clamp((v - u.mid) / (u.high - u.mid), 0.0, 1.0);
    }
    return pow(clamp((v - u.low) / (u.high - u.low), 0.0, 1.0), u.gamma);
}

constexpr sampler gridSampler(filter::linear, s_address::repeat, t_address::clamp_to_edge);
constexpr sampler nearestSampler(filter::nearest, s_address::repeat, t_address::clamp_to_edge);
constexpr sampler lutSampler(filter::linear, s_address::clamp_to_edge, t_address::clamp_to_edge);

constant float3 OCEAN = float3(0.043, 0.071, 0.129);
constant float3 LAND = float3(0.129, 0.145, 0.161);
constant float3 COAST = float3(0.74, 0.83, 0.94);
constant float3 SKY = float3(0.36, 0.60, 1.00);
constant float3 TERMINATOR = float3(1.00, 0.56, 0.26);

// ---------------------------------------------------------------------------------------
// Globe
// ---------------------------------------------------------------------------------------

vertex VOut globeVertex(uint vid [[vertex_id]], constant GlobeUniforms &u [[buffer(0)]]) {
    float2 ndc = float2(vid == 1 ? 3.0 : -1.0, vid == 2 ? 3.0 : -1.0);  // one big triangle
    VOut out;
    out.position = float4(ndc, 0.0, 1.0);
    out.px = (ndc * 0.5 + 0.5) * u.resolution;
    return out;
}

fragment float4 globeFragment(VOut in [[stage_in]], constant GlobeUniforms &u [[buffer(0)]],
                              texture2d<float> field0 [[texture(0)]],
                              texture2d<float> field1 [[texture(1)]],
                              texture2d<float> land [[texture(2)]],
                              texture2d<float> lut [[texture(3)]],
                              texture2d<float> elevation [[texture(4)]]) {
    float2 p = (in.px - u.center) / u.radius;
    float r2 = dot(p, p);

    // --- background: off-neutral gradient and stars ---------------------------------
    float2 uv = in.px / u.resolution;
    float3 col = mix(float3(0.028, 0.028, 0.038), float3(0.016, 0.024, 0.043), smoothstep(0.0, 1.0, uv.y));
    if (u.stars > 0.0) {
        float cell = 46.0;
        float2 gid = floor(in.px / cell);
        float2 f = fract(in.px / cell);
        if (hash12(gid) > 0.55) {
            float2 c = float2(hash12(gid + 11.3), hash12(gid + 71.7));
            float d = length(f - c) * cell;
            float mag = hash12(gid + 3.1);
            float star = exp(-pow(d / (0.55 + 0.85 * mag * mag), 1.7)) * (0.18 + 0.82 * mag * mag * mag);
            col += mix(float3(0.72, 0.80, 1.0), float3(1.0, 0.92, 0.80), hash12(gid + 5.9)) * star * u.stars;
        }
    }

    // --- outside the disc: atmosphere halo ---------------------------------------------
    if (r2 > 1.0) {
        float r = sqrt(r2);
        float halo = exp(-(r - 1.0) * 22.0);
        float3 limb = normalize(u.invRotation * float3(normalize(p), 0.0));
        float lit = smoothstep(-0.35, 0.45, dot(limb, u.sunDir.xyz));
        float amount = halo * u.atmosphere * 0.34 * mix(0.16, 1.0, mix(1.0, lit, u.night));
        return float4(col + SKY * amount * 1.1, 1.0);
    }

    // --- the sphere ----------------------------------------------------------------------
    float3 e = normalize(u.invRotation * float3(p, sqrt(max(0.0, 1.0 - r2))));
    float lat = asin(clamp(e.y, -1.0, 1.0)) * 57.29577951308232;
    float lon = atan2(e.z, e.x) * 57.29577951308232;
    float2 tuv = gridUV(lat, lon);

    float landFrac = land.sample(gridSampler, tuv).r;
    float3 surface = mix(OCEAN, LAND, landFrac);
    if (u.hasField > 0.5) {
        // NaN (undefined, e.g. waves over land) would poison bilinear filtering, so it is
        // tested on the nearest texel and the basemap shows through there.
        float n0 = field0.sample(nearestSampler, tuv).r, n1 = field1.sample(nearestSampler, tuv).r;
        if (!isnan(n0) && !isnan(n1)) {
            float v = mix(field0.sample(gridSampler, tuv).r, field1.sample(gridSampler, tuv).r, u.blend);
            surface = mix(surface, lut.sample(lutSampler, float2(scalePosition(v, u), 0.5)).rgb, u.opacity);
        }
    }

    // Shaded relief from the model's orography, over land, lit from the north-west.
    if (u.hasElevation > 0.5 && u.relief > 0.0) {
        float2 dx = float2(1.0 / 1440.0, 0.0), dy = float2(0.0, 1.0 / 721.0);
        float metresPerTexel = 27800.0;  // 0.25 degrees of latitude
        float zx = (elevation.sample(gridSampler, tuv + dx).r - elevation.sample(gridSampler, tuv - dx).r)
            / (2.0 * metresPerTexel * max(cos(lat * 0.0174533), 0.05));
        float zy = (elevation.sample(gridSampler, tuv - dy).r - elevation.sample(gridSampler, tuv + dy).r)
            / (2.0 * metresPerTexel);
        float3 n = normalize(float3(-zx * 18.0, -zy * 18.0, 1.0));  // exaggerated
        float shade = dot(n, normalize(float3(-0.6, 0.6, 0.55)));
        float amount = u.relief * smoothstep(0.2, 0.6, landFrac);
        surface *= mix(1.0, 0.55 + 0.6 * shade, amount);
    }

    // Coastline at constant screen width, from the land fraction's own derivative.
    if (u.coast > 0.0) {
        float d = landFrac - 0.5;
        float w = max(fwidth(d), 1e-6);
        float line = 1.0 - smoothstep(0.0, 1.35, abs(d) / w);
        surface = mix(surface, COAST, line * u.coast * 0.55);
    }

    if (u.graticule > 0.0) {
        float latLine = abs(fract(lat / 30.0 + 0.5) - 0.5) * 30.0 / max(fwidth(lat), 1e-6);
        float lonLine = abs(fract(lon / 30.0 + 0.5) - 0.5) * 30.0 / max(min(fwidth(lon), 4.0), 1e-6);
        float grat = max(1.0 - smoothstep(0.0, 1.1, latLine), 1.0 - smoothstep(0.0, 1.1, lonLine));
        float eq = 1.0 - smoothstep(0.0, 1.4, abs(lat) / max(fwidth(lat), 1e-6));
        surface = mix(surface, float3(1.0), (grat * 0.055 + eq * 0.07) * u.graticule);
    }

    // Day and night at the frame's valid time.
    float sun = dot(e, u.sunDir.xyz);
    float day = smoothstep(-0.09, 0.15, sun);
    float3 nightSide = mix(surface * 0.55, surface * float3(0.30, 0.36, 0.55) * 1.05, 0.8);
    float3 lit = mix(nightSide, surface, day) + TERMINATOR * exp(-pow(sun / 0.075, 2.0)) * 0.11;
    surface = mix(surface, lit, u.night);

    // Limb darkening and a thin rim of atmosphere.
    float rim = sqrt(r2);
    surface *= 1.0 - 0.32 * pow(rim, 5.0);
    surface += SKY * pow(rim, 16.0) * 0.13 * u.atmosphere * mix(0.35, 1.0, day);
    surface += (hash12(in.px) - 0.5) / 255.0;  // dither against 8-bit banding
    return float4(surface, 1.0);
}

// ---------------------------------------------------------------------------------------
// Wind particles: advected in geographic coordinates every frame by the frame's own
// duration (smooth at any display rate), positions remembered in a ring at a fixed spacing
// in *time* (so trails look the same at 30, 60 or 120 Hz), trails re-projected every frame
// (so rotating the globe never smears them -- the reason AIFS-VIZ gave up the
// fading-framebuffer trick).
// ---------------------------------------------------------------------------------------

struct ParticleUniforms {
    float3x3 invRotation;   // camera -> earth, to spawn where the camera looks
    float2 extent;          // the cell's half-size in disc radii: what is on screen
    float blend;            // between wind0 and wind1
    float dt;               // seconds this step (the frame's duration)
    float advect;           // degrees of arc per second per (m/s)
    float lifeBase;         // seconds
    uint count;             // live particles
    uint stride;            // allocated particles: the history ring's row length
    uint trail;             // history slots
    uint head;              // slot to stamp into
    uint stamp;             // 1 = remember this step's positions
    uint frame;
};

static inline float2 latLon(float3 e) {
    return float2(asin(clamp(e.y, -1.0, 1.0)), atan2(e.z, e.x)) * 57.29577951308232;
}

/// A birthplace on screen: uniform over the visible part of the disc, so what is in view
/// stays stocked at any zoom and none is wasted on the far side (after AIFS-VIZ's spawn).
static inline float2 spawnInView(float2 seed, constant ParticleUniforms &u) {
    float2 box = min(u.extent, float2(1.0));
    for (int t = 0; t < 4; t++) {
        float2 h = float2(hash12(seed + float(t) * 7.31), hash12(seed + 3.7 + float(t) * 5.17));
        float2 p = (h * 2.0 - 1.0) * box;
        float r2 = dot(p, p);
        if (r2 < 1.0) return latLon(u.invRotation * float3(p, sqrt(1.0 - r2)));
    }
    // Unlucky four times: the visible cap plus a margin behind the limb.
    float zc = 1.0 - 1.6 * hash12(seed + 11.0);
    float rr = sqrt(max(0.0, 1.0 - zc * zc));
    float phi = 6.2831853 * hash12(seed + 23.0);
    return latLon(u.invRotation * float3(rr * cos(phi), rr * sin(phi), zc));
}

/// x = lat, y = lon, z = age (s), w = lifetime (s)
kernel void particleUpdate(device float4 *state [[buffer(0)]],
                           device float2 *history [[buffer(1)]],
                           constant ParticleUniforms &u [[buffer(2)]],
                           texture2d<float> wind0 [[texture(0)]],
                           texture2d<float> wind1 [[texture(1)]],
                           uint i [[thread_position_in_grid]]) {
    if (i >= u.count) return;
    float4 s = state[i];
    float2 seed = float2(float(i) * 0.6180339 + float(u.frame) * 0.013, float(u.frame) * 0.7071 + float(i));
    bool reborn = s.z >= s.w || abs(s.x) > 88.0;
    if (reborn) {
        bool first = s.w <= 0.0;  // never lived: zeroed on allocation or on going live
        s.xy = spawnInView(seed, u);
        s.w = u.lifeBase * (0.5 + hash12(seed + 31.0));
        // A first generation starts part-way through its life, so the population never
        // dies and respawns in lockstep -- a pulse across the globe.
        s.z = first ? hash12(seed + 47.0) * s.w * 0.8 : 0.0;
    }
    float2 tuv = gridUV(s.x, s.y);
    float2 w = mix(wind0.sample(gridSampler, tuv).rg, wind1.sample(gridSampler, tuv).rg, u.blend);
    if (any(isnan(w))) w = float2(0.0);
    // Off screen (behind the limb, or outside the cell when zoomed in), age ten times as
    // fast: after a drag or a pinch the population moves into the new view within a
    // second or two instead of waiting out its life where nobody can see it.
    float3 c = transpose(u.invRotation) * earth(s.x, s.y);
    bool away = c.z < 0.05 || any(abs(c.xy) > u.extent * 1.1);
    s.z += away ? 9.0 * u.dt : 0.0;
    // Parked in a calm, a particle is a dead pixel; and particles drift into
    // convergence, emptying the calm. Both are answered by moving it on.
    if (!reborn && length(w) < 0.12) s.z = s.w;
    s.x += w.y * u.advect * u.dt;
    s.y += w.x * u.advect * u.dt / max(cos(s.x * 0.0174533), 0.08);
    s.y = s.y - 360.0 * floor((s.y + 180.0) / 360.0);
    s.z += u.dt;
    state[i] = s;
    if (reborn) {
        // A new particle has no past: every slot starts at its birthplace, so no trail
        // ever spans the jump from where the old one died.
        for (uint k = 0; k < u.trail; k++) history[k * u.stride + i] = s.xy;
    } else if (u.stamp == 1) {
        history[u.head * u.stride + i] = s.xy;
    }
}

struct TrailUniforms {
    float3x3 rotation;      // earth space -> camera space
    float2 center;          // px, y-up
    float2 resolution;
    float radius;
    float width;            // px
    float fade;
    float intensity;
    float whiteness;
    float slotSeconds;      // seconds between remembered positions
    float blend;            // between wind0 and wind1
    float speedRange;       // m/s at the top of the level's usual range
    uint count;             // live particles
    uint trail;
    uint head;
    uint stride;            // allocated particles: the history ring's row length
};

struct TrailOut {
    float4 position [[position]];
    float alpha;
    float side;             // -1..1 across the trail, for an anti-aliased edge
    float halfWidth;        // px, after the taper
    float3 color;
};

vertex TrailOut trailVertex(uint vid [[vertex_id]], uint iid [[instance_id]],
                            device const float4 *state [[buffer(0)]],
                            device const float2 *history [[buffer(1)]],
                            constant TrailUniforms &u [[buffer(2)]],
                            texture2d<float> wind0 [[texture(0)]],
                            texture2d<float> wind1 [[texture(1)]]) {
    // Segment 0 runs from the particle's live position to the newest remembered one, so
    // the head moves every frame whatever the display rate; segment k >= 1 joins two
    // remembered positions. A trail is `trail` segments.
    uint i = iid % u.count, k = iid / u.count;
    float2 pa = k == 0 ? state[i].xy : history[((u.head + u.trail - k + 1) % u.trail) * u.stride + i];
    float2 pb = history[((u.head + u.trail - k) % u.trail) * u.stride + i];
    float3 ca = u.rotation * earth(pa.x, pa.y), cb = u.rotation * earth(pb.x, pb.y);

    TrailOut out;
    out.position = float4(2.0, 2.0, 0.0, 1.0);  // off-screen: a culled, zero-area segment
    out.alpha = 0.0;
    out.side = 0.0;
    out.halfWidth = 0.0;
    out.color = float3(0.0);
    bool young = float(k) * u.slotSeconds > state[i].z + 1e-3;
    if (young || ca.z < 0.0 || cb.z < 0.0 || abs(pa.y - pb.y) > 90.0) return out;

    float2 sa = u.center + ca.xy * u.radius, sb = u.center + cb.xy * u.radius;
    float2 d = sb - sa;
    float len = length(d);
    if (len < 1e-3) return out;
    float2 n = float2(-d.y, d.x) / len;
    // Two triangles: (a-, a+, b-), (b-, a+, b+).
    uint corner = vid % 6;
    float t = (corner == 2 || corner == 3 || corner == 5) ? 1.0 : 0.0;
    float side = (corner == 1 || corner == 4 || corner == 5) ? 1.0 : -1.0;
    // Per vertex, so width and brightness flow along the streak, which narrows to a point.
    float age = (float(k) + t) / float(max(u.trail, 1u));
    float halfW = max(u.width * 0.5 * mix(0.42, 1.0, 1.0 - age), 0.25);
    float feather = 0.9;
    float2 px = mix(sa, sb, t) + n * side * (halfW + feather);
    out.position = float4(px / u.resolution * 2.0 - 1.0, 0.0, 1.0);

    // Calm air recedes and fast air stands out, lifted towards white more at the fast
    // end, as in AIFS-VIZ: otherwise the trades and the jet read as the same weight.
    float2 tuv = gridUV(pa.x, pa.y);
    float2 w = mix(wind0.sample(gridSampler, tuv, level(0)).rg, wind1.sample(gridSampler, tuv, level(0)).rg, u.blend);
    float speedT = clamp(length(w) / max(u.speedRange, 1.0), 0.0, 1.0);
    if (any(isnan(w))) speedT = 0.0;

    float limb = smoothstep(0.0, 0.25, mix(ca.z, cb.z, t));
    out.alpha = u.intensity * pow(max(1.0 - age, 0.0), u.fade) * limb * mix(0.3, 1.0, sqrt(speedT));
    out.side = side * (halfW + feather);
    out.halfWidth = halfW;
    out.color = mix(float3(0.55, 0.72, 1.0), float3(1.0), clamp(u.whiteness * mix(0.58, 1.0, speedT), 0.0, 1.0));
    return out;
}

fragment float4 trailFragment(TrailOut in [[stage_in]], constant TrailUniforms &u [[buffer(0)]]) {
    float coverage = 1.0 - smoothstep(in.halfWidth - 0.55, in.halfWidth + 0.9, abs(in.side));
    float a = in.alpha * coverage;
    if (a <= 0.002) discard_fragment();
    return float4(in.color * a, a);  // premultiplied
}
