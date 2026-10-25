# poetry lily: a lily grows from light, blooms, then its particles settle into lines of your poem.
# run:  pip install pygame-ce numpy   then   python lily.py      (esc closes the window)
import math
import numpy as np
import pygame

pygame.init()
w, h = 540, 960
screen = pygame.display.set_mode((w, h))
pygame.display.set_caption("poetry lily")
clock = pygame.time.Clock()
fps = 60
rng = np.random.default_rng(11)

bg = (9, 6, 26)
cx = w // 2
base_y = 800
focal = 1100.0
pitch = math.radians(14)
cos_p, sin_p = math.cos(pitch), math.sin(pitch)

font = pygame.font.SysFont("georgia,cambria,timesnewroman,dejavuserif", 9, bold=True)

soil_text = "nobody tells the seed that the dark is a beginning. it only knows the weight of everything above it, and pushes anyway. "
root_text = "i grew down before i ever grew up, learning the names of everything that could hold me. "
stem_text = "every inch of me was a decision to want the light more than i feared the climb. "
petal_texts = [
    "some things stay closed until they are sure. being soft is not the same as being small. ",
    "no one sees what opening costs. a petal is only a held breath finally let go. ",
    "turn me and see i was never one face. i am every season i survived, all at once. ",
]

# timeline in seconds
t_stem0, t_stem1 = 1.0, 5.5
t_leaf1, t_leaf2 = 3.0, 3.6
t_bud0, t_bud1 = 5.6, 8.2
t_open0, t_open1 = 8.2, 13.0
t_morph = 14.5
morph_wave = 6.0
morph_len = 2.2

pink = np.array([240, 70, 170], dtype=float)
cream = np.array([255, 238, 225], dtype=float)
gold = np.array([255, 214, 130], dtype=float)


def ease(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def ease_a(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


# ---------- glyphs ----------
adv_cache = {}


def adv(ch):
    if ch not in adv_cache:
        adv_cache[ch] = max(3, font.size(ch)[0])
    return adv_cache[ch]


glyph_cache = {}


def glyph(ch, color, ang, level):
    a = int(round(ang / 10.0)) * 10 % 360
    key = (ch, color, a, level)
    s = glyph_cache.get(key)
    if s is None:
        s = font.render(ch, True, color)
        s = pygame.transform.rotate(s, a)
        s.set_alpha(level)
        glyph_cache[key] = s
    return s


def place_text(poly, text, shift=0, spacing=0.95):
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    pos, tan, chars, ifr = [], [], [], []
    d, i = 0.0, shift
    while True:
        ch = text[i % len(text)]
        a = adv(ch) * spacing
        s = d + a / 2
        if s >= total:
            break
        k = int(np.searchsorted(cum, s))
        k = min(max(k, 1), len(poly) - 1)
        f = (s - cum[k - 1]) / max(seg[k - 1], 1e-6)
        p = poly[k - 1] * (1 - f) + poly[k] * f
        t = poly[k] - poly[k - 1]
        t = t / max(np.linalg.norm(t), 1e-6)
        if ch != " ":
            pos.append(p)
            tan.append(t)
            chars.append(ch)
            ifr.append(k - 1 + f)
        d += a
        i += 1
    return np.array(pos).reshape(-1, 3), np.array(tan).reshape(-1, 3), chars, np.array(ifr)


def nearest(points, anchors, chunk=500):
    out = np.empty(len(points), dtype=np.int32)
    for i in range(0, len(points), chunk):
        d = ((points[i:i + chunk, None, :] - anchors[None, :, :]) ** 2).sum(-1)
        out[i:i + chunk] = d.argmin(1)
    return out


# ---------- letters (the final poem layout) ----------
L_pos, L_tan, L_ch, L_col, L_part = [], [], [], [], []


def add_letters(pos, tan, chars, colors, part):
    start = sum(len(p) for p in L_pos)
    L_pos.append(pos)
    L_tan.append(tan)
    L_ch.extend(chars)
    L_col.extend(colors)
    L_part.extend([part] * len(chars))
    return start


# ---------- geometry ----------
stem_h = 430.0


def stem_center(s):
    s = np.asarray(s, dtype=float)
    return np.stack([16 * np.sin(2.4 * s) * s, s * stem_h, 9 * np.sin(3.2 * s)], axis=1)


stem_top = stem_center([1.0])[0]


def leaf_pos(a, az, length, wmax, t, u):
    d = np.array([math.cos(az), 0.0, math.sin(az)])
    b = np.array([-math.sin(az), 0.0, math.cos(az)])
    width = wmax * np.sin(np.pi * t ** 0.75) ** 0.85
    rise = 120 * t - 95 * t ** 2 + 10 * u * u * t
    return (a[None, :] + d[None, :] * (length * t)[:, None] + np.array([0, 1.0, 0])[None, :] * rise[:, None]
            + b[None, :] * (u * width)[:, None])


petals = []
for i in range(6):
    inner = i % 2
    petals.append(dict(az=i * math.pi / 3 + 0.3, L=138.0 if inner else 152.0, wmax=46.0 if inner else 38.0,
                       th0=math.radians(18 if inner else 30), bend=math.radians(58 if inner else 78)))
p_az = np.array([p["az"] for p in petals])
p_L = np.array([p["L"] for p in petals])
p_w = np.array([p["wmax"] for p in petals])
p_th0 = np.array([p["th0"] for p in petals])
p_bend = np.array([p["bend"] for p in petals])
ntab = 48
tab_t = np.linspace(0, 1, ntab)


def wprof(t):
    return np.sin(np.pi * np.asarray(t) ** 0.75) ** 0.85


def petal_tables(o, sc):
    oo = ease(o)
    th_b = 0.12 + (p_th0 - 0.12) * oo
    bend = 0.1 + (p_bend - 0.1) * oo
    theta = th_b[:, None] + bend[:, None] * tab_t[None, :] ** 1.4
    dt = 1.0 / (ntab - 1)
    def cum(f):
        c = np.cumsum((f[:, 1:] + f[:, :-1]) / 2 * dt, axis=1)
        return np.concatenate([np.zeros((6, 1)), c], axis=1)
    r0 = 7.0 * sc
    xr = r0 + (p_L * sc)[:, None] * cum(np.sin(theta))
    yy = (p_L * sc)[:, None] * cum(np.cos(theta))
    wf = 0.5 + 0.5 * oo
    wd = (p_w * sc * wf)[:, None] * wprof(tab_t)[None, :]
    return dict(xr=xr, yy=yy, th=theta, wd=wd, cup=1.1 - 0.75 * oo)


def petal_pos(pid, t, u, o, sc, tb=None):
    if tb is None:
        tb = petal_tables(o, sc)
    ti = np.asarray(t) * (ntab - 1)
    i0 = np.minimum(ti.astype(int), ntab - 2)
    f = ti - i0

    def g(a):
        return a[pid, i0] * (1 - f) + a[pid, i0 + 1] * f
    xr, yy, th, wd = g(tb["xr"]), g(tb["yy"]), g(tb["th"]), g(tb["wd"])
    off = tb["cup"] * u * u * wd
    rr = xr - np.cos(th) * off
    yv = yy + np.sin(th) * off
    az = p_az[pid]
    side = u * wd
    x = stem_top[0] + np.cos(az) * rr - np.sin(az) * side
    z = stem_top[2] + np.sin(az) * rr + np.cos(az) * side
    return np.stack([x, stem_top[1] + yv, z], axis=1)


def petal_color(t, u):
    f = np.clip(0.55 * u * u + 0.8 * np.asarray(t) ** 2.5, 0, 1)
    c = pink[None, :] * (1 - f)[:, None] + cream[None, :] * f[:, None]
    th = np.exp(-np.asarray(t) / 0.07)[:, None]
    return c * (1 - th) + np.array([235, 235, 130])[None, :] * th, f


# ---------- build static particles, letters and targets ----------
S_base, S_tar, S_grp, S_tp, S_col, S_br, S_lid = [], [], [], [], [], [], []


def add_static(base, tar, grp, tp, col, lid):
    n = len(base)
    S_base.append(base)
    S_tar.append(tar)
    S_grp.append(np.full(n, grp))
    S_tp.append(tp)
    S_col.append(col)
    S_br.append(rng.uniform(0.5, 1.0, n))
    S_lid.append(lid)


# soil: rings of the seed lines
soil_a = []
soil_start = 0
for r in range(26, 132, 17):
    ang = np.linspace(0, 2 * math.pi, 240)
    y = 18 * max(0.0, 1 - r / 140.0) ** 1.2
    poly = np.stack([r * np.cos(ang), np.full_like(ang, y), r * np.sin(ang) * 0.9], axis=1)
    pos, tan, ch, _ = place_text(poly, soil_text, shift=r)
    soil_a.append((pos, tan, ch))
soil_pos = np.concatenate([a[0] for a in soil_a])
soil_start = add_letters(soil_pos, np.concatenate([a[1] for a in soil_a]),
                         [c for a in soil_a for c in a[2]], [(170, 120, 215)] * len(soil_pos), 0)
n = 3200
rr_ = 135 * np.sqrt(rng.random(n))
an_ = rng.random(n) * 2 * math.pi
base = np.stack([rr_ * np.cos(an_), 18 * np.clip(1 - rr_ / 140.0, 0, 1) ** 1.2 + rng.normal(0, 1.5, n), rr_ * np.sin(an_) * 0.9], axis=1)
idx = nearest(base, soil_pos)
add_static(base, soil_pos[idx], 0, rr_ / 135.0,
           np.array([170, 120, 215.0])[None, :] * rng.uniform(0.7, 1.0, (n, 1)), soil_start + idx)

# stem: three twisting strands of letters, particles fill the stem
ss = np.linspace(0, 1, 500)
cen = stem_center(ss)
sp, stn, sch = [], [], []
for k in range(3):
    ph = 2 * math.pi * k / 3 + 5.0 * ss
    rad = 6.5 - 3.2 * ss
    poly = cen + np.stack([np.cos(ph) * rad, np.zeros_like(ss), np.sin(ph) * rad], axis=1)
    pos, tan, ch, _ = place_text(poly, stem_text, shift=11 * k)
    sp.append(pos)
    stn.append(tan)
    sch.extend(ch)
stem_pos = np.concatenate(sp)
stem_start = add_letters(stem_pos, np.concatenate(stn), sch, [(120, 235, 175)] * len(stem_pos), 2)
n = 4200
s_ = rng.random(n)
a_ = rng.random(n) * 2 * math.pi
r_ = (6.2 - 3.0 * s_) * np.sqrt(rng.random(n))
base = stem_center(s_) + np.stack([np.cos(a_) * r_, rng.normal(0, 0.6, n), np.sin(a_) * r_], axis=1)
idx = nearest(base, stem_pos)
add_static(base, stem_pos[idx], 1, s_, np.array([110, 235, 170.0])[None, :] * rng.uniform(0.75, 1.0, (n, 1)), stem_start + idx)


def build_leaf(s_attach, az, length, wmax, grp):
    a = stem_center([s_attach])[0]
    tt = np.linspace(0, 1, 140)
    lp, lt, lc = [], [], []
    for li, u in enumerate(np.linspace(-1, 1, 7)):
        poly = leaf_pos(a, az, length, wmax, tt, np.full_like(tt, u))
        pos, tan, ch, _ = place_text(poly, root_text, shift=9 * li)
        lp.append(pos)
        lt.append(tan)
        lc.extend(ch)
    pos_all = np.concatenate(lp)
    start = add_letters(pos_all, np.concatenate(lt), lc, [(150, 240, 185)] * len(pos_all), 1)
    n2 = 3200
    t = rng.random(n2)
    u = rng.uniform(-1, 1, n2)
    keep = rng.random(n2) < wprof(t) * 0.9 + 0.1
    t, u = t[keep], u[keep]
    base = leaf_pos(a, az, length, wmax, t, u)
    idx = nearest(base, pos_all)
    add_static(base, pos_all[idx], grp, t, np.array([140, 240, 180.0])[None, :] * rng.uniform(0.75, 1.0, (len(t), 1)), start + idx)


build_leaf(0.16, math.pi * 0.93, 150, 32, 2)
build_leaf(0.27, 0.15, 135, 29, 3)

# petals
P_pid, P_t, P_u, P_tar, P_col, P_br, P_lid = [], [], [], [], [], [], []
tt = np.linspace(0.05, 1.0, 160)
for pid in range(6):
    pp, ptn, pch, pcol = [], [], [], []
    for li, u in enumerate(np.linspace(-0.9, 0.9, 9)):
        poly = petal_pos(np.full(160, pid), tt, np.full(160, u), 1.0, 1.0)
        pos, tan, ch, ifr = place_text(poly, petal_texts[pid % 3], shift=9 * li + 17 * pid)
        ta = 0.05 + 0.95 * ifr / 159.0
        wd = p_w[pid] * wprof(ta)
        keep = np.ones(len(ta), bool) if abs(u) < 1e-9 else (0.225 * wd >= 6.0)
        cols, f = petal_color(ta, np.full(len(ta), u))
        fq = np.round(f * 3) / 3
        cq = pink[None, :] * (1 - fq)[:, None] + cream[None, :] * fq[:, None]
        for j in np.nonzero(keep)[0]:
            pp.append(pos[j])
            ptn.append(tan[j])
            pch.append(ch[j])
            pcol.append(tuple(int(v) for v in cq[j]))
    pp = np.array(pp)
    start = add_letters(pp, np.array(ptn), pch, pcol, 3 + pid % 3)
    n3 = 2300
    t = rng.random(n3 * 2)
    t = t[rng.random(len(t)) < wprof(t)][:n3]
    t = np.clip(t, 0.0, 1.0)
    u = rng.uniform(-1, 1, len(t))
    fin = petal_pos(np.full(len(t), pid), t, u, 1.0, 1.0)
    idx = nearest(fin, pp)
    cols, _ = petal_color(t, u)
    P_pid.append(np.full(len(t), pid))
    P_t.append(t)
    P_u.append(u)
    P_tar.append(pp[idx])
    P_col.append(cols * rng.uniform(0.8, 1.0, (len(t), 1)))
    P_br.append(rng.uniform(0.55, 1.0, len(t)))
    P_lid.append(start + idx)

P_pid, P_t, P_u = np.concatenate(P_pid), np.concatenate(P_t), np.concatenate(P_u)
P_tar, P_col, P_br, P_lid = np.concatenate(P_tar), np.concatenate(P_col), np.concatenate(P_br), np.concatenate(P_lid)

S_base, S_tar, S_grp, S_tp = [np.concatenate(x) for x in (S_base, S_tar, S_grp, S_tp)]
S_col, S_br, S_lid = np.concatenate(S_col), np.concatenate(S_br), np.concatenate(S_lid)
S_dp = rng.uniform(0, 0.6, len(S_base))
P_dp = rng.uniform(0, 0.6, len(P_pid))
S_sw = rng.normal(0, 14, (len(S_base), 3))
P_sw = rng.normal(0, 14, (len(P_pid), 3))

lp_pos = np.concatenate(L_pos)
lp_tan = np.concatenate(L_tan)
l_part = np.array(L_part)
l_dl = (1 - np.clip(lp_pos[:, 1] / (stem_top[1] + 120), 0, 1)) * morph_wave

# stamens and pistil (they stay as light)
st_az, st_s, st_R, st_H, st_off, st_col, st_br = [], [], [], [], [], [], []
for k in range(6):
    az = k * math.pi / 3 + math.pi / 6
    s = np.linspace(0, 1, 55)
    for sv in s:
        st_az.append(az); st_s.append(sv); st_R.append(40.0); st_H.append(88.0)
        st_off.append(np.zeros(3)); st_col.append((235, 225, 150)); st_br.append(0.8)
    for _ in range(26):
        st_az.append(az); st_s.append(1.0); st_R.append(40.0); st_H.append(88.0)
        st_off.append(rng.normal(0, [3.5, 1.6, 1.6])); st_col.append((255, 140, 40)); st_br.append(1.0)
for sv in np.linspace(0, 1, 70):
    st_az.append(0.0); st_s.append(sv); st_R.append(12.0); st_H.append(112.0)
    st_off.append(np.zeros(3)); st_col.append((200, 235, 140)); st_br.append(0.8)
for _ in range(16):
    st_az.append(0.0); st_s.append(1.0); st_R.append(12.0); st_H.append(112.0)
    st_off.append(rng.normal(0, 2.2, 3)); st_col.append((255, 235, 120)); st_br.append(1.0)
st_az, st_s, st_R, st_H = map(np.array, (st_az, st_s, st_R, st_H))
st_off, st_col, st_br = np.array(st_off), np.array(st_col, dtype=float), np.array(st_br)

# floating dust
n_d = 80
d_x, d_z = rng.uniform(-230, 230, n_d), rng.uniform(-150, 150, n_d)
d_y0, d_sp, d_ph = rng.uniform(0, 820, n_d), rng.uniform(8, 22, n_d), rng.uniform(0, 6.28, n_d)
d_col = np.array([(255, 214, 130), (255, 160, 200), (200, 170, 255)], dtype=float)[rng.integers(0, 3, n_d)]


# ---------- backdrop, glow, camera ----------
def make_backdrop():
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt(((xx - cx) / 360.0) ** 2 + ((yy - 540) / 640.0) ** 2)
    k = np.clip(1 - d, 0, 1) ** 1.7
    glow = np.array([34, 18, 66], dtype=np.float32)
    img = np.array(bg, dtype=np.float32)[None, None, :] * (1 - k[..., None]) + glow[None, None, :] * k[..., None]
    return img.reshape(-1, 3).astype(np.float32)


backdrop = make_backdrop()


def make_glow(size=440):
    yy, xx = np.mgrid[0:size, 0:size]
    d = np.sqrt((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / (size / 2)
    k = np.clip(1 - d, 0, 1) ** 2.2
    col = np.array([255, 110, 190], dtype=float)
    arr = (k[..., None] * col[None, None, :] * 0.55).astype(np.uint8)
    surf = pygame.Surface((size, size))
    pygame.surfarray.blit_array(surf, arr.swapaxes(0, 1))
    return surf


glow_surf = make_glow()

yaw_tt = np.arange(0, 100, 0.02)
yaw_tab = np.cumsum(0.32 * ease_a((yaw_tt - 10.0) / 5.0)) * 0.02


def yaw_at(t):
    return float(np.interp(t, yaw_tt, yaw_tab))


def project(p, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    x = p[:, 0] * c + p[:, 2] * s
    z1 = -p[:, 0] * s + p[:, 2] * c
    up = p[:, 1] * cos_p + z1 * sin_p
    depth = -p[:, 1] * sin_p + z1 * cos_p
    sc = focal / (focal + depth)
    return cx + x * sc, base_y - up * sc, depth



# ---------- reading beats: one per part of the poem ----------
# set beat_secs to the seconds you take to read each part aloud (seed, roots, stem, bud, bloom, turning)
t_read0 = 24.0
beat_secs = [5.5, 5.5, 5.5, 5.5, 5.5, 6.5]
beat_t = [t_read0 + sum(beat_secs[:i]) for i in range(6)]
beat_lines = [
    ["nobody tells the seed that", "the dark is a beginning. it only", "knows the weight of everything", "above it, and pushes anyway."],
    ["i grew down before i ever grew up,", "learning the names of everything", "that could hold me."],
    ["every inch of me was a decision", "to want the light more than", "i feared the climb."],
    ["some things stay closed", "until they are sure.", "being soft is not the same", "as being small."],
    ["no one sees what opening costs.", "a petal is only a held breath", "finally let go."],
    ["turn me and see", "i was never one face.", "i am every season i survived,", "all at once."],
]
read_font = pygame.font.SysFont("georgia,cambria,timesnewroman,dejavuserif", 22, bold=True)
read_cream = (255, 240, 225)
read_shadow = (12, 6, 28)
read_cache = {}


def read_glyph(ch, color):
    key = (ch, color)
    if key not in read_cache:
        read_cache[key] = read_font.render(ch, True, color)
    return read_cache[key]


beat_data = []
line_gap, sec_gap = 34, 20
total_h = sum(len(l) for l in beat_lines) * line_gap + (len(beat_lines) - 1) * sec_gap
y_cur = (h - total_h) / 2 + line_gap / 2
for b, lines in enumerate(beat_lines):
    chars, slots = [], []
    for li, line in enumerate(lines):
        total_w = sum(read_font.size(c)[0] for c in line)
        x = cx - total_w / 2
        for c in line:
            cw = read_font.size(c)[0]
            if c != " ":
                chars.append(c)
                slots.append((x + cw / 2, y_cur))
            x += cw
        y_cur += line_gap
    y_cur += sec_gap
    part_idx = np.nonzero(l_part == b)[0]
    src = rng.permutation(part_idx)
    src = np.resize(src, len(chars)) if len(part_idx) < len(chars) else src[:len(chars)]
    beat_data.append(dict(chars=chars, slots=slots, src=src, curl=rng.normal(0, 28, (len(chars), 2))))

lines_cache = {}


def lines_surface(b):
    if b not in lines_cache:
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        bd = beat_data[b]
        for ch, (x, y) in zip(bd["chars"], bd["slots"]):
            sh = read_glyph(ch, read_shadow)
            surf.blit(sh, (x + 1.5 - sh.get_width() / 2, y + 1.5 - sh.get_height() / 2))
        for ch, (x, y) in zip(bd["chars"], bd["slots"]):
            g_ = read_glyph(ch, read_cream)
            surf.blit(g_, (x - g_.get_width() / 2, y - g_.get_height() / 2))
        lines_cache[b] = surf
    return lines_cache[b]


# ---------- frame ----------
def growth(t):
    return np.array([min(1.0, t / 1.5), ease((t - t_stem0) / (t_stem1 - t_stem0)),
                     ease((t - t_leaf1) / 2.5), ease((t - t_leaf2) / 2.5)])


def draw(t):
    yaw = yaw_at(t)
    tm = t - t_morph
    g = growth(t)
    bud = ease((t - t_bud0) / (t_bud1 - t_bud0))
    o = ease((t - t_open0) / (t_open1 - t_open0))
    pf = np.array([ease((t - beat_t[bi] - 0.5) / 2.0) for bi in range(6)])
    sc = (0.12 + 0.5 * bud) + 0.38 * o
    petal_alpha = ease((t - t_bud0) / 0.8)
    oo = ease(o)

    pos_l, col_l, br_l = [], [], []

    # static parts
    vis = S_tp <= g[np.minimum(S_grp, 3)]
    gg = g[np.minimum(S_grp, 3)]
    pos = S_base
    br = S_br.copy()
    col = S_col
    if tm > -1:
        m = ease_a((tm - l_dl[S_lid] - S_dp) / morph_len)
        pos = S_base + (S_tar - S_base) * m[:, None] + S_sw * (np.sin(np.pi * m) * 0.35)[:, None]
        br = br * (1 - 0.78 * m)
    br = br * (1 - pf[l_part[S_lid]])
    front = vis & (S_tp > gg - 0.03) & (gg < 0.999) & (S_grp >= 1)
    col = np.where(front[:, None], gold[None, :], col)
    pos_l.append(pos[vis]); col_l.append(col[vis]); br_l.append(br[vis])

    # petals
    if petal_alpha > 0:
        pp = petal_pos(P_pid, P_t, P_u, o, sc)
        pb = P_br * petal_alpha * (0.3 + 0.7 * min(1.0, (sc / 0.9) ** 2))
        if tm > -1:
            m = ease_a((tm - l_dl[P_lid] - P_dp) / morph_len)
            pp = pp + (P_tar - pp) * m[:, None] + P_sw * (np.sin(np.pi * m) * 0.35)[:, None]
            pb = pb * (1 - 0.78 * m)
        pb = pb * (1 - pf[l_part[P_lid]])
        pos_l.append(pp); col_l.append(P_col); br_l.append(pb)

    # stamens
    if oo > 0.02:
        R = st_R * oo
        Hh = st_H * (0.3 + 0.7 * oo)
        radial = R * st_s ** 1.6
        px = stem_top[0] + np.cos(st_az) * radial + st_off[:, 0]
        pz = stem_top[2] + np.sin(st_az) * radial + st_off[:, 2]
        py = stem_top[1] + Hh * st_s * (1 - 0.12 * st_s) + st_off[:, 1]
        pos_l.append(np.stack([px, py, pz], axis=1)); col_l.append(st_col); br_l.append(st_br * min(1.0, oo * 3) * (1 - pf[5]))

    # dust
    dy = (d_y0 + d_sp * t) % 820
    pos_l.append(np.stack([d_x + 8 * np.sin(t * 0.7 + d_ph), dy, d_z], axis=1))
    col_l.append(d_col)
    br_l.append(0.25 + 0.6 * (0.5 + 0.5 * np.sin(t * 2 + d_ph)))

    P = np.concatenate(pos_l)
    C = np.concatenate(col_l)
    B = np.concatenate(br_l)
    x, y, dep = project(P, yaw)
    bs = 0.55 + 0.45 * np.clip((130 - dep) / 260, 0, 1)
    ix, iy = x.astype(np.int32), y.astype(np.int32)
    ok = (ix >= 1) & (ix < w - 2) & (iy >= 1) & (iy < h - 2)
    ix, iy, C, wgt = ix[ok], iy[ok], C[ok], (B * bs * 0.42)[ok]
    idx = iy * w + ix
    big = wgt > 0.34
    idx2 = np.concatenate([idx, idx[big] + 1, idx[big] + w])
    w2 = np.concatenate([wgt, wgt[big] * 0.5, wgt[big] * 0.5])
    C2 = np.concatenate([C, C[big], C[big]])
    buf = backdrop.copy()
    for c in range(3):
        buf[:, c] += np.bincount(idx2, weights=C2[:, c] * w2, minlength=w * h).astype(np.float32)
    arr = np.clip(buf, 0, 255).astype(np.uint8).reshape(h, w, 3).swapaxes(0, 1)
    pygame.surfarray.blit_array(screen, arr)

    # bloom
    for div, k in ((4, 85), (10, 70)):
        small = pygame.transform.smoothscale(screen, (w // div, h // div))
        up = pygame.transform.smoothscale(small, (w, h))
        up.fill((k, k, k), special_flags=pygame.BLEND_RGB_MULT)
        screen.blit(up, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

    # soft glow behind the bloom
    if oo > 0.01:
        fx, fy, _ = project(np.array([[stem_top[0], stem_top[1] + 70, stem_top[2]]]), yaw)
        inten = int(255 * (0.08 + 0.3 * oo) * (1 - 0.5 * ease(tm / 4)) * (1 - pf[3:6].mean()))
        gl = glow_surf.copy()
        gl.fill((inten, inten, inten), special_flags=pygame.BLEND_RGB_MULT)
        screen.blit(gl, (fx[0] - 220, fy[0] - 220), special_flags=pygame.BLEND_RGB_ADD)

    # letters of the poem
    if tm > -1:
        ml = ease_a((tm - l_dl - 0.5) / morph_len)
        ga = ease_a((ml - 0.5) / 0.5)
        lx0, ly0, ldep = project(lp_pos, yaw)
        lx1, ly1, _ = project(lp_pos + lp_tan * 3.0, yaw)
        ang_all = np.degrees(np.arctan2(-(ly1 - ly0), lx1 - lx0))
        hid = np.zeros(len(lp_pos))
        dim = np.ones(len(lp_pos))
        moving, done = [], []
        if t > t_read0 - 1.5:
            rp = ease((t - (t_read0 - 1.5)) / 1.0)
            env = np.array([ease((t - beat_t[bi] + 0.3) / 0.6) for bi in range(6)])
            for bi in range(6):
                bd = beat_data[bi]
                kk = np.arange(len(bd["chars"]))
                m = ease_a((t - beat_t[bi] - 0.012 * kk) / 0.9)
                if m.max() > 0.001:
                    np.maximum.at(hid, bd["src"], m)
                    (done if m.min() > 0.999 else moving).append((bi, m))
            dim = 1 - 0.75 * rp * (1 - env[l_part])
        va = ga * (1 - hid) * dim * (1 - pf[l_part])
        sel = np.nonzero(va > 0.03)[0]
        if len(sel):
            kd = np.clip((120.0 - ldep[sel]) / 240.0, 0, 1)
            order = np.argsort(-ldep[sel])
            lvl = np.minimum(255, np.round(255 * (0.45 + 0.55 * kd) * va[sel] / 43.0).astype(int) * 43)
            angb = (np.round(ang_all[sel] / 10.0) * 10).astype(int) % 360
            items = []
            for j in order:
                if lvl[j] <= 0:
                    continue
                gsurf = glyph(L_ch[sel[j]], L_col[sel[j]], angb[j], int(lvl[j]))
                items.append((gsurf, (lx0[sel[j]] - gsurf.get_width() / 2, ly0[sel[j]] - gsurf.get_height() / 2)))
            screen.blits(items, False)
        # finished lines stay on screen, letters still flying are drawn one by one
        for bi, m in done:
            screen.blit(lines_surface(bi), (0, 0))
        for bi, m in moving:
            bd = beat_data[bi]
            for k in np.nonzero(m > 0.001)[0]:
                si = bd["src"][k]
                mm = float(m[k])
                ex, ey = bd["slots"][k]
                px = lx0[si] + (ex - lx0[si]) * mm + bd["curl"][k][0] * math.sin(math.pi * mm)
                py = ly0[si] + (ey - ly0[si]) * mm + bd["curl"][k][1] * math.sin(math.pi * mm)
                ang = ((ang_all[si] + 180) % 360 - 180) * (1 - mm)
                scl = 9.0 / 22.0 + (1 - 9.0 / 22.0) * mm
                if mm > 0.6:
                    sh = pygame.transform.rotozoom(read_glyph(bd["chars"][k], read_shadow), ang, scl)
                    screen.blit(sh, (px + 1.5 - sh.get_width() / 2, py + 1.5 - sh.get_height() / 2))
                gs = pygame.transform.rotozoom(read_glyph(bd["chars"][k], read_cream), ang, scl)
                gs.set_alpha(int(255 * (0.55 + 0.45 * mm)))
                screen.blit(gs, (px - gs.get_width() / 2, py - gs.get_height() / 2))

    if t < 2.0:
        pulse = 0.5 + 0.5 * math.sin(t * 6)
        pygame.draw.circle(screen, (255, int(190 + 30 * pulse), 120), (cx, base_y - 12), int(3 + 2 * pulse))


def run():
    running = True
    start = pygame.time.get_ticks() / 1000.0
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                running = False
        draw(pygame.time.get_ticks() / 1000.0 - start)
        pygame.display.flip()
        clock.tick(fps)
    pygame.quit()


if __name__ == "__main__":
    run()