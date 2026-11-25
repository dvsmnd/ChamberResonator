#!/usr/bin/env python3
"""ChamberResonator - prototype.
Draw rooms (rectangle / tapered / circle, flat / angled / domed ceiling), set acoustic
parameters, optionally import an IR per room, place path points and render an input WAV:
  Cascade - the sound passes through every room of the path in series.
  Timed   - the source walks along the path (keyframes on the waveform) and is heard from its
            own position: the room it is in processes the audio, with crossfades between rooms;
            the room it just left keeps ringing, heard through the open/door/wall passage.
            Early reflections are ray-cast from the walker's position against the room's real
            outline (rectangle / trapezoid / circle) and ceiling, and change as it walks.
Points: ordinary points are traversal; the first point in a new room is an entry point and
carries the passage type (open / door / window / wall / thick wall).
Requires: numpy, scipy (tkinter ships with Python). Optional: sounddevice for Preview.
"""
import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve

BANDS = [125, 250, 500, 1000, 2000, 4000]
MATERIALS = {  # absorption coefficients per octave band (typical values)
    "concrete":      [0.01, 0.01, 0.02, 0.02, 0.02, 0.03],
    "plaster":       [0.01, 0.02, 0.02, 0.03, 0.04, 0.05],
    "glass":         [0.35, 0.25, 0.18, 0.12, 0.07, 0.04],
    "wood panel":    [0.28, 0.22, 0.17, 0.09, 0.10, 0.11],
    "carpet":        [0.08, 0.24, 0.57, 0.69, 0.71, 0.73],
    "acoustic tile": [0.50, 0.70, 0.60, 0.70, 0.70, 0.50],
    "brick":         [0.03, 0.03, 0.03, 0.04, 0.05, 0.07],
    "curtains":      [0.07, 0.31, 0.49, 0.75, 0.70, 0.60],
    "rubber (experimental)": [0.04, 0.04, 0.08, 0.12, 0.03, 0.10],
    "water (experimental)":  [0.008, 0.008, 0.013, 0.015, 0.020, 0.025],
    "flesh (experimental)":  [0.25, 0.35, 0.45, 0.55, 0.60, 0.65],
}
MAT_FILTER = {  # optional "material filter" on top of a room: (gain, low-pass cutoff Hz)
    "plaster": (0.90, 14000), "brick": (0.87, 12000), "concrete": (0.94, 15000), "wood panel": (0.86, 10500),
    "glass": (0.95, 17000), "carpet": (0.72, 5500), "curtains": (0.70, 4800), "acoustic tile": (0.75, 6000),
    "rubber (experimental)": (0.68, 4200), "water (experimental)": (0.80, 8500), "flesh (experimental)": (0.60, 3000),
}
TRANSMISSION = {"open": (None, 0), "door": (2500, -6), "window": (5000, -12),
                "wall": (400, -20), "thick wall": (200, -30)}  # (cutoff Hz, dB)
REF_T, REF_RH, REF_IR_DIST = 20.0, 50.0, 10.0


def c_air(T):
    return 331.3 + 0.606 * T


def alpha_db(f, T=20.0, rh=50.0, p=101.325):
    """Air absorption in dB/m, ISO 9613-1."""
    f = np.asarray(f, float)
    Tk, T0, T01, pr = T + 273.15, 293.15, 273.16, 101.325
    psat = pr * 10 ** (-6.8346 * (T01 / Tk) ** 1.261 + 4.6151)
    h = rh * psat / p
    frO = (p / pr) * (24 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    frN = (p / pr) * (Tk / T0) ** -0.5 * (9 + 280 * h * np.exp(-4.170 * ((Tk / T0) ** (-1 / 3) - 1)))
    f2 = f * f
    return 8.686 * f2 * (1.84e-11 * (pr / p) * (Tk / T0) ** 0.5 + (Tk / T0) ** -2.5 * (
        0.01275 * np.exp(-2239.1 / Tk) / (frO + f2 / frO) + 0.1068 * np.exp(-3352 / Tk) / (frN + f2 / frN)))


def fft_filter(x, sr, gain_fn):
    n = x.shape[-1]
    N = int(2 ** np.ceil(np.log2(n + 8192)))
    X = np.fft.rfft(x, N, axis=-1) * gain_fn(np.fft.rfftfreq(N, 1 / sr))
    return np.fft.irfft(X, N, axis=-1)[..., :n]


def air_absorb(x, sr, T, rh, dist):
    return fft_filter(x, sr, lambda f: 10 ** (-alpha_db(f, T, rh) * dist / 20))


def transmit(x, sr, kind):
    fc, db = TRANSMISSION[kind]
    if fc is None:
        return x
    return fft_filter(x, sr, lambda f: 10 ** (db / 20) / np.sqrt(1 + (f / fc) ** 4))


def room_geometry(r):
    """Volume and surface area (m3, m2) from plan shape, ceiling type and height."""
    w, d, shape = r["x1"] - r["x0"], r["y1"] - r["y0"], r.get("shape", "rectangle")
    tp = r.get("taper", 1.0)
    if shape == "circle":
        D = min(w, d)
        area, per = np.pi * (D / 2) ** 2, np.pi * D
    elif shape == "tapered":
        area, per = d * w * (1 + tp) / 2, w * (1 + tp) + 2 * np.hypot(d, w * (1 - tp) / 2)
    else:
        area, per = w * d, 2 * (w + d)
    h, ceil = r["h"], r.get("ceiling", "flat")
    if ceil == "angled":
        h2 = r.get("h_end", h)
        hm, ceil_area = (h + h2) / 2, area * np.hypot(1, (h2 - h) / max(d, 1e-6))
    elif ceil == "domed":
        rise = r.get("dome_rise", 1.0)
        hm, ceil_area = h + rise / 2, area * (1 + 0.5 * (rise / max(min(w, d) / 2, 1e-6)) ** 2)
    else:
        hm, ceil_area = h, area
    return area * hm, area + ceil_area + per * hm


def room_contains(r, x, y):
    """Point-in-room test (metres) respecting the plan shape."""
    w, d = r["x1"] - r["x0"], r["y1"] - r["y0"]
    lx, ly = x - r["x0"], y - r["y0"]
    if not (0 <= lx <= w and 0 <= ly <= d):
        return False
    if r.get("shape") == "circle":
        return (lx - w / 2) ** 2 + (ly - d / 2) ** 2 <= (min(w, d) / 2) ** 2
    if r.get("shape") == "tapered":
        cur = w * r["taper"] + (w - w * r["taper"]) * ly / d
        return (w - cur) / 2 <= lx <= (w + cur) / 2
    return True


def new_room(n, x0, y0, x1, y1, shape="rectangle"):
    return dict(name=f"Room {n}", x0=x0, y0=y0, x1=x1, y1=y1, shape=shape, taper=0.6 if shape == "tapered" else 1.0,
                ceiling="flat", h=3.0, h_end=3.5, dome_rise=1.0, T=20.0, rh=50.0, mat="plaster", engine="Statistical",
                matfilt=False, er=True, trim=1.0, ir=None, irsr=None, irname="")


def rt60_bands(r):
    """Sabine RT60 per octave band incl. air absorption, times the room's decay multiplier."""
    V, S = room_geometry(r)
    a = np.array(MATERIALS[r["mat"]])
    m = alpha_db(BANDS, r["T"], r["rh"]) / 4.343
    return 0.161 * V / (S * a + 4 * m * V) * r["trim"]


def synth_ir(r, sr, seed=1):
    """Statistical IR: noise with per-octave-band exponential decay."""
    rt = rt60_bands(r)
    n = int(sr * min(rt.max(), 8.0)) + 1
    t = np.arange(n) / sr
    N = int(2 ** np.ceil(np.log2(n)))
    spec = np.fft.rfft(np.random.default_rng(seed).standard_normal(N))
    f = np.fft.rfftfreq(N, 1 / sr)
    tail = np.zeros(n)
    for i, fc in enumerate(BANDS):
        lo = 0 if i == 0 else fc / 2 ** 0.5
        hi = np.inf if i == len(BANDS) - 1 else fc * 2 ** 0.5
        band = np.fft.irfft(spec * ((f >= lo) & (f < hi)), N)[:n]
        tail += band * np.exp(-6.91 * t / rt[i])
    tail *= 0.7 / np.sqrt(np.sum(tail ** 2))
    tail[0] += 1.0  # direct sound
    return tail


def estimate_rt60(ir, sr):
    edc = np.cumsum(ir[::-1] ** 2)[::-1]
    db = 10 * np.log10(edc / edc[0] + 1e-12)
    i5, i25 = np.argmax(db < -5), np.argmax(db < -25)
    return max(3 * (i25 - i5) / sr, 0.05) if i25 > i5 else 0.5


def resample_lin(x, ratio):
    return np.interp(np.arange(int(len(x) * ratio)) / ratio, np.arange(len(x)), x)


def modify_ir(r, ir, irsr, sr):
    """Imported IR treated as recorded at 20C/50%RH; apply the differences."""
    if irsr != sr:
        ir = resample_lin(ir, sr / irsr)
    ir = resample_lin(ir, REF_T and c_air(REF_T) / c_air(r["T"]))  # faster sound -> shorter time axis
    if abs(r["trim"] - 1) > 1e-3:
        rt = estimate_rt60(ir, sr)
        t = np.arange(len(ir)) / sr
        ir = ir * np.exp(-6.91 * t * (1 / (rt * r["trim"]) - 1 / rt))
    dA = alpha_db(np.fft.rfftfreq(2 ** 16, 1 / sr), r["T"], r["rh"]) - alpha_db(np.fft.rfftfreq(2 ** 16, 1 / sr), REF_T, REF_RH)
    return fft_filter(ir, sr, lambda f: np.minimum(10 ** (-np.interp(f, np.fft.rfftfreq(2 ** 16, 1 / sr), dA) * REF_IR_DIST / 20), 2.0))


# 8.1.7 layout: ch1-8 ear level (45 deg steps, ch1 = front/up on canvas, clockwise), ch9 LFE,
# ch10-15 height ring (30, 90 ... 330 deg), ch16 zenith.
N_OUT = 16
EAR_AZ = np.arange(8) * 45.0
HGT_AZ = 30.0 + np.arange(6) * 60.0


def center(r):
    return (r["x0"] + r["x1"]) / 2, (r["y0"] + r["y1"]) / 2


def bearing(a, b):
    """Azimuth (deg, clockwise, canvas-up = 0) of room b as seen from room a."""
    (xa, ya), (xb, yb) = center(a), center(b)
    return np.degrees(np.arctan2(xb - xa, ya - yb)) % 360


def next_prime(n):
    while any(n % q == 0 for q in range(2, int(n ** 0.5) + 1)):
        n += 1
    return n


def fdn_tail(r, sr, seed=2):
    """16-line FDN (Hadamard feedback, one-pole damping per line), block-vectorised. Returns (16, n)."""
    from scipy.linalg import hadamard
    from scipy.signal import lfilter
    N = 16
    rt = rt60_bands(r)
    V, _ = room_geometry(r)
    L = V ** (1 / 3) / c_air(r["T"])
    D = np.array([next_prime(max(int(sr * L * f), 211)) for f in np.linspace(0.6, 1.7, N)])
    B, M = int(D.min()), int(D.max())
    n = int(sr * min(rt.max(), 8.0)) // B * B + B
    H = hadamard(N) / np.sqrt(N)
    g_lo, g_hi = 10 ** (-3 * D / (sr * rt[1])), 10 ** (-3 * D / (sr * rt[5]))
    rho = np.minimum(g_hi / g_lo, 1.0)
    p = (1 - rho) / (1 + rho)  # one-pole: DC gain g_lo, Nyquist gain g_hi
    zi = [np.zeros(1) for _ in range(N)]
    s = np.zeros((N, M + n))
    out = np.zeros((N, n))
    sign = np.random.default_rng(seed).choice([-1.0, 1.0], N)
    for n0 in range(0, n, B):
        f = np.empty((N, B))
        for j in range(N):
            a = M + n0 - D[j]
            f[j], zi[j] = lfilter([g_lo[j] * (1 - p[j])], [1, -p[j]], s[j, a:a + B], zi=zi[j])
        out[:, n0:n0 + B] = f
        blk = H @ f
        if n0 == 0:
            blk[:, 0] += 0.25 * sign
        s[:, M + n0:M + n0 + B] = blk
    return out * np.sqrt(0.49 / np.sum(out ** 2))


def pan_gains(az):
    """Cosine panning toward az (deg); az=None -> non-directional (own sound, centred in the head)."""
    g = np.zeros(N_OUT)
    w = (lambda a: np.ones_like(a)) if az is None else (lambda a: np.maximum(0, np.cos(np.radians(az - a))) ** 3)
    g[:8], g[9:15] = w(EAR_AZ), 0.35 * w(HGT_AZ)
    return g / max(np.sqrt((g ** 2).sum()), 1e-9)


def spatialize(ir, sr, az, seed=3):
    """Mono IR -> 16 ch: direct+early (first 50 ms) panned toward az, late part decorrelated per channel."""
    ne, rmp = int(0.05 * sr), int(0.005 * sr)
    fade = np.clip((ne + rmp - np.arange(len(ir))) / rmp, 0, 1)
    early, late = ir * fade, ir * (1 - fade)
    N = int(2 ** np.ceil(np.log2(2 * len(ir))))
    spec = np.fft.rfft(late, N)
    rng = np.random.default_rng(seed)
    out = np.zeros((N_OUT, len(ir)))
    for k in [c for c in range(N_OUT) if c != 8]:
        out[k] = np.fft.irfft(spec * np.exp(1j * rng.uniform(-np.pi, np.pi, len(spec))), N)[:len(ir)] / np.sqrt(15)
    return out + np.outer(pan_gains(az), early)


def _room_ir(r, sr, nch, az, er=False):
    if r.get("ir") is not None:
        base = modify_ir(r, r["ir"], r["irsr"], sr)
    elif r.get("engine") == "Bypass":
        base = np.array([1.0])
    elif r.get("engine") == "FDN":
        tail = fdn_tail(r, sr)
        if nch == 1:
            t = tail.sum(0)
            t *= 0.7 / np.sqrt((t ** 2).sum())
            t[0] += 1.0
            return t[None]
        mc = np.zeros((N_OUT, tail.shape[1]))
        mc[:8], mc[9:] = tail[:8], tail[8:15]
        mc[:, 0] += pan_gains(az)  # direct sound
        return mc
    else:
        base = synth_ir(r, sr)
        if er:  # the early part is supplied by the position-dependent reflections instead
            base[1:] *= np.clip((np.arange(1, len(base)) / sr - 0.005) / 0.04, 0, 1)
    return base[None] if nch == 1 else spatialize(base, sr, az)


def room_ir(r, sr, nch, az, er=False):
    """(nch, m) impulse response for one room: imported IR > Bypass > FDN > statistical (+ optional material filter)."""
    ir = _room_ir(r, sr, nch, az, er)
    if r.get("matfilt"):
        g, fc = MAT_FILTER[r["mat"]]
        ir = fft_filter(ir, sr, lambda f: g / np.sqrt(1 + (f / fc) ** 4))
    return ir


def bearing_xy(a, b):
    return np.degrees(np.arctan2(b[0] - a[0], a[1] - b[1])) % 360


def finish(y, sr, nch):
    if nch == N_OUT:
        y[8] = fft_filter(y[:8].sum(0), sr, lambda f: 1 / np.sqrt(1 + (f / 120.0) ** 8)) * 0.5
    return y / max(np.abs(y).max(), 1e-9) * 0.9


def stages_from_points(points):
    """Consecutive points in the same room form one stage; kind = connection used to enter it."""
    out = []
    for p in points:
        if out and out[-1]["room"] is p["room"]:
            out[-1]["exit"] = (p["x"], p["y"])
        else:
            out.append(dict(room=p["room"], kind=p["kind"], enter=(p["x"], p["y"]), exit=(p["x"], p["y"])))
    return out


def render(x, sr, stages, nch=1, log=print):
    """Cascade model: the whole signal passes through every stage (room) in series."""
    y = np.asarray(x, float)[None]
    for i, s in enumerate(stages):
        r = s["room"]
        if i > 0:
            q = stages[i - 1]
            dist = np.hypot(s["enter"][0] - q["exit"][0], s["enter"][1] - q["exit"][1])
            T, rh = (q["room"]["T"] + r["T"]) / 2, (q["room"]["rh"] + r["rh"]) / 2
            y = np.concatenate([np.zeros((y.shape[0], int(dist / c_air(T) * sr))), y], axis=1)
            y = transmit(air_absorb(y, sr, T, rh, dist), sr, s["kind"]) / max(dist, 1.0)
            log(f"  {q['room']['name']} -> {r['name']}: {dist:.1f} m, {dist / c_air(T) * 1000:.0f} ms, {s['kind']}")
        az = (bearing_xy(s["exit"], stages[i + 1]["enter"]) if i + 1 < len(stages)
              else bearing_xy(s["exit"], stages[i - 1]["exit"]) if i else 0.0)
        y = fftconvolve(y, room_ir(r, sr, nch, az), axes=-1)
        log(f"{r['name']}: " + ("imported IR" if r.get("ir") is not None else r.get("engine", "Statistical"))
            + (f", early sound toward {az:.0f} deg" if nch > 1 else ""))
    return finish(y, sr, nch)


# ---- early reflections from the room's real shape, for a walker (source = listener) standing at p ----
SRC_H = 1.6                    # source/ear height above the floor (m)
N_RAYS = 360
RAY_AZ = np.arange(N_RAYS) * 360.0 / N_RAYS                                     # compass azimuth, 0 = canvas up
RAY_DIR = np.stack([np.sin(np.radians(RAY_AZ)), -np.cos(np.radians(RAY_AZ))], 1)  # (x right, y down)
ER_GAIN, ER_MAX_ENERGY = 18.0 / N_RAYS, 0.8
_PAN = {}


def room_polygon(r):
    x0, y0, x1, y1 = r["x0"], r["y0"], r["x1"], r["y1"]
    i = (x1 - x0) * (1 - r["taper"]) / 2 if r.get("shape") == "tapered" else 0.0
    return np.array([[x0 + i, y0], [x1 - i, y0], [x1, y1], [x0, y1]])


def cast_rays(r, p):
    """Distance to the wall and cosine of the incidence angle along each of the N_RAYS directions from p."""
    u, p = RAY_DIR, np.asarray(p, float)
    if r.get("shape") == "circle":
        c, R = np.array(center(r)), min(r["x1"] - r["x0"], r["y1"] - r["y0"]) / 2
        q = p - c
        b = u @ q
        t = -b + np.sqrt(np.maximum(b * b - (q @ q - R * R), 0))
        return t, np.sum(u * (q + t[:, None] * u), 1) / R
    V = room_polygon(r)
    E = np.roll(V, -1, 0) - V
    nrm = np.stack([E[:, 1], -E[:, 0]], 1) / np.linalg.norm(E, axis=1)[:, None]
    nrm *= np.where(np.sum(nrm * ((V + np.roll(V, -1, 0)) / 2 - V.mean(0)), 1) < 0, -1, 1)[:, None]  # outward
    w = V - p
    den = u[:, 0, None] * E[None, :, 1] - u[:, 1, None] * E[None, :, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (w[None, :, 0] * E[None, :, 1] - w[None, :, 1] * E[None, :, 0]) / den
        sp = (w[None, :, 0] * u[:, 1, None] - w[None, :, 1] * u[:, 0, None]) / den
    t = np.where((t > 1e-9) & (sp >= 0) & (sp <= 1), t, np.inf)
    j = t.argmin(1)
    k = np.arange(N_RAYS)
    return t[k, j], np.sum(u * nrm[j], 1)


def ceiling_height(r, p):
    w, d, h = r["x1"] - r["x0"], r["y1"] - r["y0"], r["h"]
    if r.get("ceiling") == "angled":
        return h + (r.get("h_end", h) - h) * min(max((p[1] - r["y0"]) / d, 0), 1)
    if r.get("ceiling") == "domed":
        cx, cy = center(r)
        rho = min(np.hypot(p[0] - cx, p[1] - cy) / max(min(w, d) / 2, 1e-6), 1)
        return h + r.get("dome_rise", 1.0) * (1 - rho * rho)
    return h


def inside_pos(r, x, y):
    cx, cy = center(r)
    for k in range(21):
        px, py = x + (cx - x) * k / 20, y + (cy - y) * k / 20
        if room_contains(r, px, py): return px, py
    return cx, cy


def er_ir(r, p, sr, nch):
    """(nch, n) early-reflection response at position p: wall echoes per ray (specular lobe, arriving from the
    wall direction), ceiling echo (local ceiling height) and floor echo, coloured by the wall material."""
    c, Lmax = c_air(r["T"]), int(0.25 * sr)
    t, cph = cast_rays(r, p)
    L = 2 * t
    amp = ER_GAIN * np.maximum(cph, 0) ** 12 / np.maximum(L, 1.0)
    if nch > 1:
        if "g" not in _PAN: _PAN["g"] = np.array([pan_gains(a) for a in RAY_AZ])
        G = _PAN["g"]
    else:
        G = np.ones((N_RAYS, 1))
    er = np.zeros((Lmax + 2, nch))

    def put(delay, a):
        keep = delay < Lmax - 1
        i, f = np.floor(delay[keep]).astype(int), (delay[keep] - np.floor(delay[keep]))[:, None]
        np.add.at(er, i, a[keep] * (1 - f)); np.add.at(er, i + 1, a[keep] * f)

    put(L / c * sr, amp[:, None] * G)
    gc, gf = np.zeros(nch), np.zeros(nch)
    if nch == 1: gc[:] = gf[:] = 1
    else: gc[15], gc[9:15], gf[:8] = 1.0, 0.3, 1.0
    Lc, Lf = 2 * max(ceiling_height(r, p) - SRC_H, 0.3), 2 * SRC_H
    put(np.array([Lc / c * sr, Lf / c * sr]), np.stack([0.7 / max(Lc, 1) * gc / np.linalg.norm(gc), 0.7 / max(Lf, 1) * gf / np.linalg.norm(gf)]))
    e = (er ** 2).sum()
    if e > ER_MAX_ENERGY: er *= np.sqrt(ER_MAX_ENERGY / e)
    refl = np.sqrt(1 - np.clip(MATERIALS[r["mat"]], 0, 0.99))
    return fft_filter(er.T, sr, lambda f: np.interp(np.log10(np.maximum(f, 1.0)), np.log10(BANDS), refl))


def apply_er(seg, sr, r, pos_fn, nch):
    """Time-varying convolution: 50 %-overlap Hann blocks, each with the reflections for the walker's position."""
    hop, n = int(0.023 * sr), len(seg)
    win = np.sin(np.pi * np.arange(2 * hop) / (2 * hop)) ** 2
    out = np.zeros((nch, n + int(0.25 * sr) + 2))
    for k in range(n // hop + 2):
        a = (k - 1) * hop
        lo, hi = max(a, 0), min(a + 2 * hop, n)
        if hi <= lo: continue
        chunk = seg[lo:hi] * win[lo - a:hi - a]
        if not chunk.any(): continue
        blk = fftconvolve(chunk[None], er_ir(r, pos_fn(k * hop / sr), sr, nch), axes=-1)
        out[:, lo:lo + blk.shape[1]] += blk
    return out


# ---- timed path: keyframes map path points to times in the source audio ----
def make_timeline(points, keyframes, duration):
    """keyframes: [{time, pt}] -> knots [(time, point_index)]; skipped points are timed by path distance."""
    if not points: raise ValueError("Add at least one path point.")
    if not duration: raise ValueError("Load an audio file.")
    if not keyframes: raise ValueError("Add at least one keyframe (click the waveform).")
    idx = {id(p): i for i, p in enumerate(points)}
    frames = []
    for k in sorted(keyframes, key=lambda k: k["time"]):
        if id(k["pt"]) not in idx: raise ValueError("A keyframe refers to a missing point.")
        if not 0 <= k["time"] <= duration: raise ValueError("Keyframe times must lie within the source audio.")
        frames.append((k["time"], idx[id(k["pt"])]))
    for (t0, i0), (t1, i1) in zip(frames, frames[1:]):
        if t1 - t0 < 0.001: raise ValueError("Keyframes must be at least 0.001 s apart.")
        if i1 < i0: raise ValueError("Keyframes must follow numbered path order.")
    anchors = list(frames)
    if anchors[0][0] > 0: anchors.insert(0, (0.0, anchors[0][1]))
    if anchors[-1][0] < duration: anchors.append((duration, anchors[-1][1]))
    if len(anchors) == 1: anchors.append((duration, anchors[0][1]))
    knots = [anchors[0]]
    for (ta, ia), (tb, ib) in zip(anchors, anchors[1:]):
        if ia == ib:
            knots.append((tb, ib))
            continue
        seg = [np.hypot(points[j]["x"] - points[j - 1]["x"], points[j]["y"] - points[j - 1]["y"]) for j in range(ia + 1, ib + 1)]
        total, run = sum(seg), 0.0
        for n, j in enumerate(range(ia + 1, ib + 1)):
            run += seg[n]
            knots.append((ta + (tb - ta) * (run / total if total > 0 else (j - ia) / (ib - ia)), j))
    return knots


def position_at(points, knots, t):
    P = lambda i: (points[i]["x"], points[i]["y"])
    if t < knots[0][0]: return P(knots[0][1])
    for (ta, ia), (tb, ib) in zip(knots, knots[1:]):
        if t < tb:
            f = min(max((t - ta) / (tb - ta), 0), 1) if tb > ta else 1
            (xa, ya), (xb, yb) = P(ia), P(ib)
            return xa + (xb - xa) * f, ya + (yb - ya) * f
    return P(knots[-1][1])


def make_visits(points, knots, duration):
    """Room visits; the room changes halfway in time between two knots lying in different rooms."""
    v = [dict(room=points[knots[0][1]]["room"], start=0.0, end=duration, pt=knots[0][1])]
    for (t0, _), (t1, i1) in zip(knots, knots[1:]):
        if v[-1]["room"] is not points[i1]["room"]:
            v[-1]["end"] = (t0 + t1) / 2
            v.append(dict(room=points[i1]["room"], start=(t0 + t1) / 2, end=duration, pt=i1))
    return v


def render_timed(x, sr, points, keyframes, nch=1, log=print):
    """Walker model: the source hears itself in whichever room it is in. Audio is gated into room branches
    (30 ms crossfades), each convolved with its room; the room being left keeps ringing through the passage."""
    n = len(x)
    knots = make_timeline(points, keyframes, n / sr)
    visits = make_visits(points, knots, n / sr)
    fades = [min(0.03, (a["end"] - a["start"]) / 2, (b["end"] - b["start"]) / 2) for a, b in zip(visits, visits[1:])]
    outs = []
    for i, v in enumerate(visits):
        g = np.zeros(n)
        a, b = int(v["start"] * sr), int(v["end"] * sr)
        g[a:b] = 1
        fi = int(fades[i - 1] * sr) if i > 0 else 0
        fo = int(fades[i] * sr) if i < len(visits) - 1 else 0
        if fi: g[a - fi:a + fi] = np.linspace(0, 1, 2 * fi)
        if fo:
            sl = slice(b - fo, b + fo)
            g[sl] = np.linspace(1, 0, 2 * fo)[:len(g[sl])]
        room, seg = v["room"], np.asarray(x, float) * g
        use_er = room.get("er", True) and room.get("ir") is None and room.get("engine") != "Bypass"
        out = fftconvolve(seg[None], room_ir(room, sr, nch, None, er=use_er), axes=-1)
        if use_er:  # reflections for the walker's actual position, kept inside this room's outline
            eo = apply_er(seg, sr, room, lambda tc: inside_pos(room, *position_at(points, knots, tc)), nch)
            m = max(out.shape[1], eo.shape[1])
            out = np.pad(out, ((0, 0), (0, m - out.shape[1]))) + np.pad(eo, ((0, 0), (0, m - eo.shape[1])))
        if i < len(visits) - 1:  # leaving: what keeps ringing here is heard through the passage into the next room
            kind = points[visits[i + 1]["pt"]]["kind"]
            if kind != "open":
                w = np.clip((np.arange(out.shape[1]) - b) / int(0.05 * sr), 0, 1)
                out = (1 - w) * out + w * transmit(out, sr, kind)
        outs.append(out)
        log(f"{v['room']['name']}: {v['start']:.2f}-{v['end']:.2f} s"
            + (f", left through {points[visits[i + 1]['pt']]['kind']}" if i < len(visits) - 1 else ""))
    y = np.zeros((outs[0].shape[0], max(o.shape[1] for o in outs)))
    for o in outs:
        y[:, :o.shape[1]] += o
    return finish(y, sr, nch)


def to_float_mono(d):
    if d.dtype == np.int16: d = d / 32768.0
    elif d.dtype == np.int32: d = d / 2 ** 31
    elif d.dtype == np.uint8: d = (d - 128) / 128.0
    d = d.astype(float)
    return d.mean(axis=1) if d.ndim > 1 else d


def main():
    import time
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    S, MW, MH, WH = 20, 820, 560, 110
    SHAPES, CEILS, ENGINES = ["rectangle", "tapered", "circle"], ["flat", "angled", "domed"], ["Statistical", "FDN", "Bypass"]
    root = tk.Tk(); root.title("ChamberResonator - prototype")
    rooms, points, keys = [], [], []
    st = dict(room=None, pt=None, key=None, klist=[], x=None, sr=None, start=None, head=None, play=None, n=0)
    msg = tk.StringVar(value="Draw rooms, add path points, load an input WAV.")
    say = msg.set
    cv = tk.Canvas(root, width=MW, height=MH, bg="#1c1c1e", highlightthickness=0)
    wv = tk.Canvas(root, width=MW, height=WH, bg="#141416", highlightthickness=0)
    cv.grid(row=0, column=0, padx=6, pady=(6, 0)); wv.grid(row=1, column=0, padx=6, pady=3)
    nb = ttk.Notebook(root); nb.grid(row=0, column=1, rowspan=2, sticky="n", padx=8, pady=6)
    tr, tp = ttk.Frame(nb, padding=6), ttk.Frame(nb, padding=6)
    nb.add(tr, text="Room"); nb.add(tp, text="Path / keyframes")
    mode, shape_new, model, outm = tk.StringVar(value="draw"), tk.StringVar(value="rectangle"), tk.StringVar(value="Timed (moving source)"), tk.StringVar(value="mono")

    # ---- room panel ----
    RV = {k: tk.StringVar() for k in ("name", "shape", "w", "d", "taper", "ceiling", "h", "h_end", "dome_rise", "T", "rh", "mat", "trim", "engine")}
    mf, erv = tk.BooleanVar(), tk.BooleanVar()
    for k, lab, vals in [("name", "Name", None), ("shape", "Shape", SHAPES), ("w", "Width (m)", None), ("d", "Length (m)", None),
                         ("taper", "Top width ratio (tapered)", None), ("ceiling", "Ceiling", CEILS),
                         ("h", "Height / start height (m)", None), ("h_end", "End height (angled)", None),
                         ("dome_rise", "Dome rise (domed)", None), ("T", "Temp (C)", None), ("rh", "Humidity (%)", None),
                         ("mat", "Material (statistical / FDN)", list(MATERIALS)), ("trim", "Decay x (0.5-1.5)", None),
                         ("engine", "Reverb engine (no IR loaded)", ENGINES)]:
        ttk.Label(tr, text=lab).pack(anchor="w")
        w = (ttk.Combobox(tr, textvariable=RV[k], values=vals, state="readonly", width=22) if vals
             else ttk.Entry(tr, textvariable=RV[k], width=24))
        w.pack(anchor="w"); w.bind("<<ComboboxSelected>>" if vals else "<Return>", lambda e: apply_room())
    ttk.Checkbutton(tr, text="Material filter on top\n(may double-count in an IR)", variable=mf, command=lambda: apply_room()).pack(anchor="w", pady=3)
    ir_lbl = tk.StringVar()
    ttk.Checkbutton(tr, text="Early reflections from room shape\n(Timed model; not used with an IR)", variable=erv, command=lambda: apply_room()).pack(anchor="w", pady=3)
    ttk.Label(tr, textvariable=ir_lbl, wraplength=190).pack(anchor="w")
    ttk.Button(tr, text="Load IR for room...", command=lambda: load_ir()).pack(anchor="w")
    ttk.Button(tr, text="Clear IR", command=lambda: clear_ir()).pack(anchor="w")
    ttk.Button(tr, text="Apply", command=lambda: apply_room()).pack(anchor="w", pady=(6, 0))
    ttk.Button(tr, text="Delete room", command=lambda: del_room()).pack(anchor="w")

    # ---- path / keyframe panel ----
    PV = {"kind": tk.StringVar()}
    pos_lbl, tv = tk.StringVar(), tk.StringVar()
    pt_lbl = tk.StringVar()
    ttk.Label(tp, textvariable=pt_lbl, wraplength=190).pack(anchor="w")
    pc = ttk.Combobox(tp, textvariable=PV["kind"], values=list(TRANSMISSION), state="disabled", width=22); pc.pack(anchor="w")
    pc.bind("<<ComboboxSelected>>", lambda e: apply_point())
    ttk.Label(tp, textvariable=pos_lbl).pack(anchor="w")
    for lab, fn in (("Delete point", lambda: del_point(st["pt"])), ("Undo last point", lambda: del_point(points[-1] if points else None)),
                    ("Clear path", lambda: (points.clear(), keys.clear(), st.update(pt=None, key=None), refresh_keys(), redraw()))):
        ttk.Button(tp, text=lab, command=fn).pack(anchor="w")
    ttk.Separator(tp).pack(fill="x", pady=6)
    ttk.Label(tp, text="Keyframes: click the waveform to add one\nfor the selected point (Timed model)").pack(anchor="w")
    kl = tk.Listbox(tp, width=30, height=10, exportselection=False); kl.pack(anchor="w")
    kl.bind("<<ListboxSelect>>", lambda e: key_sel())
    ttk.Entry(tp, textvariable=tv, width=12).pack(anchor="w")
    ttk.Button(tp, text="Set time (s)", command=lambda: set_time()).pack(anchor="w")
    ttk.Button(tp, text="Delete keyframe", command=lambda: del_key()).pack(anchor="w")

    # ---- drawing ----
    def shape_item(b, shape, taper, **kw):
        x0, y0, x1, y1 = [v * S for v in b]
        if shape == "circle":
            cx, cy, rad = (x0 + x1) / 2, (y0 + y1) / 2, min(x1 - x0, y1 - y0) / 2
            return cv.create_oval(cx - rad, cy - rad, cx + rad, cy + rad, **kw)
        if shape == "tapered":
            i = (x1 - x0) * (1 - taper) / 2
            return cv.create_polygon(x0 + i, y0, x1 - i, y0, x1, y1, x0, y1, **kw)
        return cv.create_rectangle(x0, y0, x1, y1, **kw)

    def pidx(p): return next((i for i, q in enumerate(points) if q is p), -1) + 1
    def is_entry(p):  # first point of a new room (never the very first point of the path)
        i = pidx(p) - 1
        return i > 0 and points[i - 1]["room"] is not p["room"]
    def sdur(): return len(st["x"]) / st["sr"] if st["x"] is not None else 0.0

    def draw_wave():
        wv.delete("all")
        x = st["x"]
        if x is None:
            wv.create_text(10, 14, anchor="w", fill="#888", text="Load an input WAV to see its waveform; click it to add a keyframe for the selected point.")
            return
        step = max(1, -(-len(x) // MW)); m = len(x) // step
        blk = x[:m * step].reshape(m, step)
        mid, amp = WH / 2, WH / 2 - 6
        peak = max(np.abs(x).max(), 1e-9)
        for i, (lo, hi) in enumerate(zip(blk.min(1), blk.max(1))):
            px = i * MW / m
            wv.create_line(px, mid - hi / peak * amp, px, mid - lo / peak * amp, fill="#5b8def")
        for k in keys:
            px = k["time"] / sdur() * MW
            wv.create_line(px, 0, px, WH, fill="#ffd166" if k is st["key"] else "#ea580c", width=2)
            wv.create_text(px + 4, 8, anchor="w", fill="#ffd166", text=str(pidx(k["pt"])))
        if st["head"]:
            px = st["head"][0] / sdur() * MW
            wv.create_line(px, 0, px, WH, fill="#10b981", width=3)

    def redraw():
        cv.delete("all")
        for gx in range(0, MW, S): cv.create_line(gx, 0, gx, MH, fill="#2a2a2d")
        for gy in range(0, MH, S): cv.create_line(0, gy, MW, gy, fill="#2a2a2d")
        for r in rooms:
            shape_item((r["x0"], r["y0"], r["x1"], r["y1"]), r["shape"], r["taper"], width=2, fill="#26303f",
                       outline="#ffb454" if r is st["room"] else "#8ab4f8")
            cv.create_text(r["x0"] * S + 6, r["y0"] * S + 6, anchor="nw", fill="white",
                           text=f"{r['name']}\n{r['shape']}, {r['ceiling']} ceiling\n{r['x1'] - r['x0']:g}x{r['y1'] - r['y0']:g}x{r['h']:g} m, {r['T']:g}C {r['rh']:g}%"
                                + ("\nIR" if r["ir"] is not None else ""))
        src = st["head"][1] if st["head"] else ((st["pt"]["x"], st["pt"]["y"]) if st["pt"] else None)
        r0 = next((r for r in reversed(rooms) if src and room_contains(r, *src)), None)
        if r0:  # the rays that feed the early reflections from this position
            t, _ = cast_rays(r0, src)
            for k in range(0, N_RAYS, 10):
                cv.create_line(src[0] * S, src[1] * S, (src[0] + RAY_DIR[k, 0] * t[k]) * S, (src[1] + RAY_DIR[k, 1] * t[k]) * S, fill="#2c5f4f")
        pts = [(p["x"] * S, p["y"] * S) for p in points]
        if len(pts) > 1: cv.create_line(*[c for q in pts for c in q], fill="#ff6b6b", width=2)
        for i, p in enumerate(points):
            x, y = pts[i]; sel = p is st["pt"]
            ent, col, ol = is_entry(p), "#b91c1c" if sel else "#ea580c", "#fff" if sel else ""
            (cv.create_rectangle if ent else cv.create_oval)(x - 9, y - 9, x + 9, y + 9, fill=col, outline=ol)
            cv.create_text(x, y, text=str(i + 1), fill="white")
            if ent: cv.create_text(x, y - 17, text=p["kind"], fill="#ffd166")
        if st["head"]:
            x, y = st["head"][1][0] * S, st["head"][1][1] * S
            cv.create_oval(x - 13, y - 13, x + 13, y + 13, fill="#0b4f3a", outline="")
            cv.create_oval(x - 7, y - 7, x + 7, y + 7, fill="#10b981", outline="#fff", width=2)
        draw_wave()

    # ---- panels <-> data ----
    def load_room(r):
        if not r: return
        for k in ("name", "shape", "taper", "ceiling", "h", "h_end", "dome_rise", "T", "rh", "mat", "trim", "engine"): RV[k].set(r[k])
        RV["w"].set(f"{r['x1'] - r['x0']:g}"); RV["d"].set(f"{r['y1'] - r['y0']:g}")
        mf.set(r["matfilt"]); erv.set(r["er"]); ir_lbl.set("IR: " + r["irname"] if r["ir"] is not None else "IR: none")

    def apply_room():
        r = st["room"]
        if not r: return
        try:
            w, d = float(RV["w"].get()), float(RV["d"].get())
            if RV["shape"].get() == "circle": w = d = min(w, d)
            r.update(name=RV["name"].get(), shape=RV["shape"].get(), x1=min(r["x0"] + max(w, 1), MW / S), y1=min(r["y0"] + max(d, 1), MH / S),
                     taper=min(max(float(RV["taper"].get()), 0.1), 1.0), ceiling=RV["ceiling"].get(), h=max(float(RV["h"].get()), 1.0),
                     h_end=max(float(RV["h_end"].get()), 1.0), dome_rise=max(float(RV["dome_rise"].get()), 0.0), T=float(RV["T"].get()),
                     rh=min(max(float(RV["rh"].get()), 0.0), 100.0), mat=RV["mat"].get(), trim=min(max(float(RV["trim"].get()), 0.5), 1.5),
                     engine=RV["engine"].get(), matfilt=mf.get(), er=erv.get())
        except ValueError:
            return say("Numeric fields must be numbers.")
        load_room(r); redraw()

    def load_point(p):
        if p:
            ent = is_entry(p); PV["kind"].set(p["kind"] if ent else ""); pc.configure(state="readonly" if ent else "disabled")
            pos_lbl.set(f"Point {pidx(p)} at {p['x']:.1f}, {p['y']:.1f} m ({p['room']['name']})")
            pt_lbl.set("ENTRY point: passage from the previous room" if ent else "Start point" if pidx(p) == 1 else "Traversal point (no passage setting)")
        else:
            pos_lbl.set("No point selected"); pt_lbl.set(""); PV["kind"].set(""); pc.configure(state="disabled")

    def apply_point():
        if st["pt"] and is_entry(st["pt"]): st["pt"].update(kind=PV["kind"].get()); redraw()

    def del_point(p):
        if p is None: return
        i = pidx(p) - 1
        if is_entry(p) and i + 1 < len(points) and points[i + 1]["room"] is p["room"]: points[i + 1]["kind"] = p["kind"]  # next point becomes the entry
        points[:] = [q for q in points if q is not p]; keys[:] = [k for k in keys if k["pt"] is not p]
        st["pt"] = points[-1] if points else None; st["key"] = None
        load_point(st["pt"]); refresh_keys(); redraw()

    def del_room():
        r = st["room"]
        if not r: return
        gone = [p for p in points if p["room"] is r]
        rooms[:] = [q for q in rooms if q is not r]
        points[:] = [p for p in points if p["room"] is not r]; keys[:] = [k for k in keys if all(k["pt"] is not p for p in gone)]
        st.update(room=None, pt=points[-1] if points else None, key=None); load_point(st["pt"]); refresh_keys(); redraw()

    def refresh_keys():
        st["klist"] = sorted(keys, key=lambda k: k["time"]); kl.delete(0, "end")
        for k in st["klist"]: kl.insert("end", f"{k['time']:.3f} s -> point {pidx(k['pt'])} ({k['pt']['room']['name']})")

    def key_sel():
        s = kl.curselection()
        if not s: return
        k = st["klist"][s[0]]; st.update(key=k, pt=k["pt"], room=k["pt"]["room"]); tv.set(f"{k['time']:.3f}")
        load_point(st["pt"]); load_room(st["room"]); redraw()

    def set_time():
        if not st["key"]: return
        try: st["key"]["time"] = min(max(float(tv.get()), 0.0), sdur())
        except ValueError: return say("Time must be a number.")
        refresh_keys(); redraw()

    def del_key():
        keys[:] = [k for k in keys if k is not st["key"]]; st["key"] = None; refresh_keys(); redraw()

    def load_ir():
        r = st["room"]
        if not r: return say("Select a room first.")
        fn = filedialog.askopenfilename(filetypes=[("WAV", "*.wav")])
        if fn:
            sr, d = wavfile.read(fn); r.update(ir=to_float_mono(d), irsr=sr, irname=fn.split("/")[-1]); load_room(r); redraw()

    def clear_ir():
        if st["room"]: st["room"]["ir"] = None; load_room(st["room"]); redraw()

    def load_in():
        fn = filedialog.askopenfilename(filetypes=[("WAV", "*.wav")])
        if not fn: return
        st["sr"], d = wavfile.read(fn); st["x"] = to_float_mono(d); in_lbl.set(fn.split("/")[-1])
        keys.clear()
        if points: keys.extend([dict(time=0.0, pt=points[0]), dict(time=sdur(), pt=points[-1])])
        refresh_keys(); redraw(); say("Input loaded. Timed model: edit the initial keyframes.")

    # ---- mouse ----
    sx, sy = (lambda v: min(max(round(v / S * 2) / 2, 0), MW / S)), (lambda v: min(max(round(v / S * 2) / 2, 0), MH / S))
    def at_room(mx, my):
        return next((r for r in reversed(rooms) if room_contains(r, mx / S, my / S)), None)
    def at_pt(mx, my):
        return next((p for p in reversed(points) if np.hypot(p["x"] * S - mx, p["y"] * S - my) <= 14), None)
    def box(a, b):
        (xa, ya), (xb, yb) = a, b
        if shape_new.get() == "circle":
            s = min(abs(xb - xa), abs(yb - ya)); xb, yb = xa + np.sign(xb - xa) * s, ya + np.sign(yb - ya) * s
        return min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb)

    def press(e):
        m = mode.get()
        if m == "draw": st["start"] = (sx(e.x), sy(e.y))
        elif m == "select":
            p = at_pt(e.x, e.y); r = p["room"] if p else at_room(e.x, e.y)
            if p: st["pt"] = p; load_point(p)
            if r: st["room"] = r; load_room(r)
            redraw()
        else:
            r = at_room(e.x, e.y)
            if not r: return say("Place points inside a room outline.")
            p = dict(x=e.x / S, y=e.y / S, room=r, kind="open")
            points.append(p); st.update(pt=p, room=r); load_point(p); load_room(r); redraw()

    def drag(e):
        if mode.get() == "draw" and st["start"]:
            redraw(); shape_item(box(st["start"], (sx(e.x), sy(e.y))), shape_new.get(), 0.6, outline="#ffb454", dash=(4, 3), fill="")

    def release(e):
        if mode.get() != "draw" or not st["start"]: return
        b = box(st["start"], (sx(e.x), sy(e.y))); st["start"] = None
        if b[2] - b[0] >= 1 and b[3] - b[1] >= 1:
            st["n"] += 1; r = new_room(st["n"], *b, shape_new.get()); rooms.append(r); st["room"] = r; load_room(r); mode.set("select")
        redraw()

    def wave_click(e):
        if st["x"] is None or st["pt"] is None: return say("Load audio and select a path point first.")
        k = dict(time=min(max(e.x / MW, 0), 1) * sdur(), pt=st["pt"]); keys.append(k); st["key"] = k
        refresh_keys(); redraw(); say(f"Keyframe added at {k['time']:.3f} s for point {pidx(k['pt'])}.")

    cv.bind("<ButtonPress-1>", press); cv.bind("<B1-Motion>", drag); cv.bind("<ButtonRelease-1>", release)
    wv.bind("<Button-1>", wave_click)

    # ---- rendering / preview ----
    def compute(nch, lines):
        if st["x"] is None: raise ValueError("Load an input WAV first.")
        if not points: raise ValueError("Add at least one path point (Add path points mode).")
        if model.get().startswith("Timed"): return render_timed(st["x"], st["sr"], points, keys, nch, lines.append)
        return render(st["x"], st["sr"], stages_from_points(points), nch, lines.append)

    def stop_preview():
        try:
            import sounddevice as sd; sd.stop()
        except Exception: pass
        if st["play"]: root.after_cancel(st["play"]); st["play"] = None
        st["head"] = None

    def preview():
        try: import sounddevice as sd
        except ImportError: return messagebox.showinfo("Preview", "Preview needs sounddevice (pip install sounddevice).\nUse Render... to export a WAV instead.")
        stop_preview(); say("Rendering preview..."); root.update_idletasks()
        try:
            y = compute(1, [])[0]
            timed = model.get().startswith("Timed")
            kn = make_timeline(points, keys, sdur()) if timed else None
        except ValueError as ex:
            return say(str(ex))
        sd.play(y.astype(np.float32), st["sr"]); t0 = time.perf_counter(); say("Playing.")
        def tick():
            t = time.perf_counter() - t0
            if t > len(y) / st["sr"]: st["head"] = None; st["play"] = None; redraw(); return say("Preview finished.")
            if timed: tt = min(t, sdur()); st["head"] = (tt, position_at(points, kn, tt))
            redraw(); st["play"] = root.after(40, tick)
        tick()

    def do_render():
        stop_preview(); apply_room(); lines = []; nch = N_OUT if outm.get().startswith("16") else 1
        say("Rendering..."); root.update_idletasks()
        try: y = compute(nch, lines)
        except ValueError as ex: return messagebox.showerror("Cannot render", str(ex))
        out = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV", "*.wav")])
        if not out: return say("Render discarded.")
        wavfile.write(out, st["sr"], y.T.astype(np.float32) if nch > 1 else (y[0] * 32767).astype(np.int16))
        say(f"Saved {out}"); messagebox.showinfo("Rendered", "\n".join(lines) + f"\n\nSaved {out}")

    # ---- bars ----
    mb = ttk.Frame(root); mb.grid(row=2, column=0, sticky="w", padx=6)
    for m, lab in (("draw", "Draw room"), ("select", "Select"), ("path", "Add path points")):
        ttk.Radiobutton(mb, text=lab, variable=mode, value=m).pack(side="left", padx=4)
    ttk.Combobox(mb, textvariable=shape_new, values=SHAPES, state="readonly", width=10).pack(side="left", padx=4)
    fb = ttk.Frame(root); fb.grid(row=3, column=0, sticky="w", padx=6, pady=4)
    in_lbl = tk.StringVar(value="no input")
    ttk.Button(fb, text="Load input WAV...", command=load_in).pack(side="left")
    ttk.Label(fb, textvariable=in_lbl, width=16).pack(side="left")
    ttk.Combobox(fb, textvariable=model, values=["Cascade (series)", "Timed (moving source)"], state="readonly", width=20).pack(side="left", padx=4)
    ttk.Combobox(fb, textvariable=outm, values=["mono", "16ch 8.1.7"], state="readonly", width=10).pack(side="left", padx=4)
    ttk.Button(fb, text="Preview", command=preview).pack(side="left")
    ttk.Button(fb, text="Stop", command=lambda: (stop_preview(), redraw())).pack(side="left")
    ttk.Button(fb, text="Render...", command=do_render).pack(side="left", padx=8)
    ttk.Label(root, textvariable=msg).grid(row=4, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 6))
    redraw(); root.mainloop()


if __name__ == "__main__":
    main()
