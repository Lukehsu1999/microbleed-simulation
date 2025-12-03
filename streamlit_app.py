# gibbs_explorer_v7.py
# Streamlit app — Lesion-first truncation THEN overlay on MRI,
# with rich visualizations (zoom, k-space, profiles), noisy boundary option,
# and an export feature that downloads the overlaid image + a txt of all config.

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import zipfile, json
from datetime import datetime

# ----- Try enhancing in-lesion contrast -------
# def enhance_lesion_alpha(lesion_alpha_trunc, lesion_alpha_clean,
#                          thresh=0.05, gamma=1.0):
#     """
#     Boost contrast of the truncated lesion alpha *within* the lesion support.
#     - thresh: define support from clean alpha (where lesion exists)
#     - gamma: optional nonlinearity (gamma<1 -> fatter core, gamma>1 -> thinner core)
#     """
#     a = lesion_alpha_trunc.copy()
#     support = lesion_alpha_clean > thresh
#     vals = a[support]
#     if vals.size > 0:
#         vmin, vmax = vals.min(), vals.max()
#         if vmax - vmin > 1e-6:
#             vals = (vals - vmin) / (vmax - vmin)  # stretch to [0,1]
#             if gamma != 1.0:
#                 vals = vals**gamma
#             a[support] = vals
#     return np.clip(a, 0.0, 1.0)
def enhance_lesion_alpha(lesion_alpha_trunc, lesion_alpha_clean,
                         core_thresh=0.5, gamma=1.0):
    """
    Boost contrast mainly in the core; leave rim more stable.
    """
    a = lesion_alpha_trunc.copy()

    # core only (avoid including the outer rim / ripples)
    core = lesion_alpha_clean > core_thresh

    vals = a[core]
    if vals.size > 0:
        vmin, vmax = vals.min(), vals.max()
        if vmax - vmin > 1e-6:
            vals = (vals - vmin) / (vmax - vmin)
            if gamma != 1.0:
                vals = vals**gamma
            a[core] = vals

    return np.clip(a, 0.0, 1.0)

# ---------- Fourier helpers (centered FFT/IFFT) ----------
def fft2c(x: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))

def ifft2c(k: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k)))

# ---------- Shapes (return soft binary "alpha" masks) ----------
def _ellipse_rmap(shape, cx: float, cy: float, oval_ratio: float):
    H, W = shape
    yy, xx = np.ogrid[:H, :W]
    dx = (xx - cx) / max(oval_ratio, 1e-8)
    dy = (yy - cy)
    rmap = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)  # use scaled dx for consistent ellipse angle
    return rmap, theta

def _noisy_radius(theta, radius, noise_amp, seed=0, n_modes=12):
    """Low-frequency angular noise around the rim; normalized so noise_amp is intuitive."""
    rng = np.random.default_rng(seed)
    amps = rng.normal(scale=1.0, size=n_modes)
    phases = rng.uniform(0, 2*np.pi, size=n_modes)
    ks = np.arange(1, n_modes + 1)

    ang = theta + np.pi  # [0, 2pi]
    noise = np.zeros_like(ang, dtype=float)
    for a, p, k in zip(amps, phases, ks):
        noise += a * np.cos(k * ang + p) / k  # smooth (1/k)
    noise = noise / (np.std(noise) + 1e-12)
    return radius * (1.0 + noise_amp * noise)

def make_disk(
    shape,
    radius: float,
    cx: float,
    cy: float,
    oval_ratio: float = 1.0,
    edge_sigma: float = 0.0,
    noisy_boundary: bool = False,
    noise_amp: float = 0.0,
    seed: int = 0,
    n_modes: int = 12,
) -> np.ndarray:
    """Soft/hard filled ellipse (1 inside, 0 outside) with optional wobbly rim."""
    rmap, theta = _ellipse_rmap(shape, cx, cy, oval_ratio)
    if noisy_boundary and noise_amp > 0.0:
        r_eff = _noisy_radius(theta, radius, noise_amp, seed=seed, n_modes=n_modes)
        signed = rmap - r_eff
    else:
        signed = rmap - radius

    if edge_sigma <= 1e-8:
        return (signed <= 0).astype(float)
    return 1.0 / (1.0 + np.exp(signed / max(edge_sigma, 1e-8)))

def make_two_disks(
    shape,
    r1: float, r2: float,
    center: tuple,
    center_dist: float,
    vertical: bool = False,
    oval_ratio1: float = 1.0,
    oval_ratio2: float = 1.0,
    edge_sigma: float = 0.0,
    noisy_boundary: bool = False,
    noise_amp: float = 0.0,
    seed: int = 0,
    n_modes: int = 12,
) -> np.ndarray:
    """Two ellipses centered around 'center' separated by center_dist. Returns union mask."""
    cx, cy = center
    if vertical:
        c1 = (cx, cy - center_dist / 2)
        c2 = (cx, cy + center_dist / 2)
    else:
        c1 = (cx - center_dist / 2, cy)
        c2 = (cx + center_dist / 2, cy)

    d1 = make_disk(shape, r1, c1[0], c1[1], oval_ratio1, edge_sigma,
                   noisy_boundary=noisy_boundary, noise_amp=noise_amp, seed=seed, n_modes=n_modes)
    d2 = make_disk(shape, r2, c2[0], c2[1], oval_ratio2, edge_sigma,
                   noisy_boundary=noisy_boundary, noise_amp=noise_amp, seed=seed+1, n_modes=n_modes)
    return np.clip(d1 + d2, 0, 1)

# ---------- k-space mask ----------
def make_lowpass_mask(shape, keep_frac: float, circular: bool = True) -> np.ndarray:
    H, W = shape
    N = min(H, W)
    keep_frac = float(np.clip(keep_frac, 0.0, 1.0))
    if keep_frac <= 0: return np.zeros((H, W), float)
    if keep_frac >= 1: return np.ones((H, W), float)

    if circular:
        rad_k = keep_frac * (N / 2)
        yy, xx = np.ogrid[:H, :W]
        cy = (H - 1) / 2
        cx = (W - 1) / 2
        r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        return (r <= rad_k).astype(float)
    else:
        half_h = int(np.round(keep_frac * H / 2))
        half_w = int(np.round(keep_frac * W / 2))
        mask = np.zeros((H, W), float)
        cy = H // 2
        cx = W // 2
        mask[cy-half_h:cy+half_h+1, cx-half_w:cx+half_w+1] = 1.0
        return mask

def truncate_image(img: np.ndarray, keep_frac: float, circular=True):
    k = fft2c(img)
    mask = make_lowpass_mask(img.shape, keep_frac, circular)
    k_trunc = k * mask
    rec = np.real(ifft2c(k_trunc))
    return rec, mask, k, k_trunc

# ---------- Utilities ----------
def norm01(a: np.ndarray) -> np.ndarray:
    amin, amax = np.min(a), np.max(a)
    if amax - amin < 1e-12: return np.zeros_like(a)
    return (a - amin) / (amax - amin)

def to_gray01(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img.convert('L'), dtype=np.float32)
    arr = (arr - arr.min()) / (max(1e-6, arr.max() - arr.min()))
    return arr

def blend_lesion(background: np.ndarray, lesion_alpha: np.ndarray, dark_val: float, strength: float) -> np.ndarray:
    """Overlay lesion onto background using (possibly ringed) alpha."""
    a = np.clip(lesion_alpha, 0.0, 1.0) * np.clip(strength, 0.0, 1.0)
    return background * (1.0 - a) + dark_val * a

def line_profile(img: np.ndarray, axis: str = "x", coord: int = None):
    H, W = img.shape
    if coord is None:
        coord = H // 2 if axis == "x" else W // 2
    if axis == "x":
        return img[coord, :]
    else:
        return img[:, coord]

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Gibbs Explorer v7 — export config + image", layout="wide")
st.title("Gibbs Artifact Explorer — Lesion-First Truncation (Noisy Boundary + Export)")

with st.sidebar:
    st.header("Background MRI")
    up = st.file_uploader("Upload a brain MRI (PNG/JPG)", type=["png", "jpg", "jpeg"])
    keep_size = st.checkbox("Keep original resolution (recommended)", value=True)
    if not keep_size:
        N = st.select_slider("Canvas size (square)", options=[128,192,256,320,384,512,640,768], value=512)

    st.header("Lesion shape")
    mode = st.radio("Type", ["Single disk/ellipse", "Two disks (overlap)"], index=0)
    edge_sigma = st.slider("Lesion edge softness (px)", 0.0, 3.0, 0.3, 0.05)

    st.subheader("Boundary irregularity")
    noisy_boundary = st.checkbox("Enable noisy boundary (wobbly rim)", value=False)
    noise_amp = st.slider("Boundary noise amplitude", 0.0, 0.3, 0.10, 0.01, disabled=not noisy_boundary)
    n_modes = st.slider("Noise modes (smoothness)", 4, 24, 12, 1, help="Fewer modes ⇒ smoother wobble", disabled=not noisy_boundary)
    seed = st.number_input("Noise seed", value=0, step=1, disabled=not noisy_boundary)

    st.subheader("Placement")
    specify_center = st.checkbox("Specify center (x,y)", value=True)

    st.header("Lesion appearance")
    dark_val = st.slider("Target dark intensity", 0.0, 1.0, 0.05, 0.01)
    strength = st.slider("Blend strength", 0.0, 1.0, 0.85, 0.01)

    st.header("Gibbs (on lesion ONLY)")
    keep_frac = st.slider("Keep fraction", 0.02, 1.0, 0.5, 0.01)
    circular_lp = st.checkbox("Circular low-pass", value=True)
    show_kspace = st.checkbox("Show lesion |K| (log)", value=True)

    in_mcb_enhance_contrast = st.checkbox("Enhance contrast within MCB", value=True)

# ----- Load background -----
if up is not None:
    bg_img = Image.open(up)
    bg = to_gray01(bg_img)
    bg_name = getattr(up, "name", "uploaded_image")
else:
    H, W = 384, 384
    yy, xx = np.mgrid[0:H, 0:W]
    bg = norm01(0.6 + 0.2*np.sin(xx/20) + 0.1*np.cos(yy/27))
    bg_name = "placeholder_gradient"

if not keep_size:
    bg = np.array(Image.fromarray((bg*255).astype(np.uint8)).resize((N, N), Image.BILINEAR), dtype=np.float32) / 255.0

H, W = bg.shape
cx0, cy0 = W // 2, H // 2

# ----- Build lesion -----
with st.sidebar:
    if mode == "Single disk/ellipse":
        radius = st.slider("Radius (px)", 1, 30, value=8)
        oval_ratio = st.slider("Oval ratio (x stretch)", 0.5, 2.0, 1.0, 0.01)
        if specify_center:
            cx = st.slider("Center X (px)", 0, W-1, cx0)
            cy = st.slider("Center Y (px)", 0, H-1, cy0)
        else:
            cx, cy = cx0, cy0
        lesion_alpha_clean = make_disk(
            (H, W), radius, cx, cy, oval_ratio, edge_sigma,
            noisy_boundary=noisy_boundary, noise_amp=noise_amp, seed=seed, n_modes=n_modes
        )
        # For config export:
        lesion_params = {
            "type": "single_ellipse",
            "center": {"x": float(cx), "y": float(cy)},
            "radius": float(radius),
            "oval_ratio": float(oval_ratio),
        }
    else:
        r1 = st.slider("Radius #1 (px)", 1, max(4, min(H, W)//3), value=max(4, min(H, W)//40))
        r2 = st.slider("Radius #2 (px)", 1, max(4, min(H, W)//3), value=max(4, min(H, W)//40))
        center_dist = st.slider("Center distance (px)", 0, min(H, W)//2, min(H, W)//10, 1)
        vertical = st.checkbox("Vertical arrangement", value=False)
        if specify_center:
            cx = st.slider("Group center X (px)", 0, W-1, cx0)
            cy = st.slider("Group center Y (px)", 0, H-1, cy0)
        else:
            cx, cy = cx0, cy0
        oval_ratio1 = st.slider("Oval ratio #1", 0.5, 2.0, 1.0, 0.01)
        oval_ratio2 = st.slider("Oval ratio #2", 0.5, 2.0, 1.0, 1.0)
        lesion_alpha_clean = make_two_disks(
            (H, W), r1, r2, (cx, cy), center_dist, vertical,
            oval_ratio1, oval_ratio2, edge_sigma,
            noisy_boundary=noisy_boundary, noise_amp=noise_amp, seed=seed, n_modes=n_modes
        )
        lesion_params = {
            "type": "two_disks",
            "group_center": {"x": float(cx), "y": float(cy)},
            "r1": float(r1), "r2": float(r2),
            "center_distance": float(center_dist),
            "vertical": bool(vertical),
            "oval_ratio1": float(oval_ratio1),
            "oval_ratio2": float(oval_ratio2),
        }

# ---------- Truncate and Blend -------
if in_mcb_enhance_contrast:
  lesion_alpha_trunc, lp_mask, k, k_trunc = truncate_image(lesion_alpha_clean, keep_frac, circular=circular_lp)
  lesion_alpha_trunc = enhance_lesion_alpha(lesion_alpha_trunc, lesion_alpha_clean, core_thresh=0.05, gamma=1.0)
  composite = blend_lesion(bg, lesion_alpha_trunc, dark_val=dark_val, strength=strength)
else:
  lesion_alpha_trunc, lp_mask, k, k_trunc = truncate_image(lesion_alpha_clean, keep_frac, circular=circular_lp)
  lesion_alpha_trunc = np.clip(lesion_alpha_trunc, 0.0, 1.0)
  composite = blend_lesion(bg, lesion_alpha_trunc, dark_val=dark_val, strength=strength)

# ----- Zoom -----
with st.sidebar:
    st.header("Zoom & profiles")
    zoom_size = st.slider("Zoom window (px)", 16, min(512, min(H, W)), value=min(96, min(H, W)//2), step=2)
    zoom_center_x = st.slider("Zoom center X", 0, W-1, int(cx))
    zoom_center_y = st.slider("Zoom center Y", 0, H-1, int(cy))
    show_profiles = st.checkbox("Show intensity profiles", value=True)

x0 = int(np.clip(zoom_center_x - zoom_size // 2, 0, W - zoom_size))
y0 = int(np.clip(zoom_center_y - zoom_size // 2, 0, H - zoom_size))
zoom_bg = bg[y0:y0 + zoom_size, x0:x0 + zoom_size]
zoom_alpha_trunc = lesion_alpha_trunc[y0:y0 + zoom_size, x0:x0 + zoom_size]
zoom_comp = composite[y0:y0 + zoom_size, x0:x0 + zoom_size]


# ---------- Layout ----------
top_cols = st.columns(4 if show_kspace else 3, gap="large")
with top_cols[0]:
    st.subheader("Background MRI")
    st.image(bg, clamp=True, use_container_width=True, caption=f"H×W={H}×{W}")
with top_cols[1]:
    st.subheader("Lesion alpha (clean)")
    st.image(lesion_alpha_clean, clamp=True, use_container_width=True)
with top_cols[2]:
    st.subheader("Lesion alpha (after truncation)")
    st.image(lesion_alpha_trunc, clamp=True, use_container_width=True, caption=f"keep_frac={keep_frac:.2f}")
if show_kspace:
    with top_cols[3]:
        mag = np.log1p(np.abs(k_trunc))
        st.subheader("|K| (log)")
        st.image(norm01(mag), clamp=True, use_container_width=True)

st.divider()

cols2 = st.columns(2, gap="large")
with cols2[0]:
    st.subheader("Final overlay (lesion-first truncation)")
    st.image(composite, clamp=True, use_container_width=True)
with cols2[1]:
    st.subheader("k-space mask (applied to lesion)")
    st.image(lp_mask, clamp=True, use_container_width=True)

st.divider()

# Zooms
z1, z2, z3 = st.columns(3, gap="large")
z1.image(zoom_bg, clamp=True, caption="Zoomed Background")
z2.image(zoom_alpha_trunc, clamp=True, caption="Zoomed Lesion α (truncated)")
z3.image(zoom_comp, clamp=True, caption="Zoomed Composite")

# Profiles
fig_row = None
fig_col = None
if show_profiles:
    st.divider()
    st.subheader("Center-line intensity profiles (Composite)")
    cx_prof, cy_prof = int(cx), int(cy)
    prow = line_profile(composite, axis="x", coord=cy_prof)
    pcol = line_profile(composite, axis="y", coord=cx_prof)
    x_row = np.arange(composite.shape[1])
    x_col = np.arange(composite.shape[0])

    c1, c2 = st.columns(2)
    with c1:
        fig_row, ax = plt.subplots()
        ax.plot(x_row, prow)
        ax.axvline(cx_prof, linestyle=":", alpha=0.7)
        ax.set_title(f"Row profile (y={cy_prof})")
        st.pyplot(fig_row, use_container_width=True)
    with c2:
        fig_col, ax = plt.subplots()
        ax.plot(x_col, pcol)
        ax.axvline(cy_prof, linestyle=":", alpha=0.7)
        ax.set_title(f"Column profile (x={cx_prof})")
        st.pyplot(fig_col, use_container_width=True)

st.divider()

# ----- Export (all images + config.txt) -----
st.subheader("💾 Export all images + configuration")

# Prepare normalized PIL images
def to_pil_gray(a):
    return Image.fromarray((norm01(a) * 255).astype(np.uint8))

img_bg = to_pil_gray(bg)
img_overlay = to_pil_gray(composite)
img_alpha_clean = to_pil_gray(lesion_alpha_clean)
img_alpha_trunc = to_pil_gray(lesion_alpha_trunc)

# Convert to bytes
def img_bytes(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# Build config dictionary (same as before)
config = {
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "background": {
        "source_name": bg_name,
        "height": int(H),
        "width": int(W),
        "kept_original_size": bool(keep_size),
    },
    "lesion": {
        **lesion_params,
        "edge_sigma": float(edge_sigma),
        "noisy_boundary": bool(noisy_boundary),
        "noise_amp": float(noise_amp if noisy_boundary else 0.0),
        "noise_modes": int(n_modes if noisy_boundary else 0),
        "noise_seed": int(seed if noisy_boundary else 0),
        "dark_val": float(dark_val),
        "blend_strength": float(strength),
        "enhance_contrast_in_mcb": bool(in_mcb_enhance_contrast),
    },
    "gibbs": {
        "keep_frac": float(keep_frac),
        "circular_lowpass": bool(circular_lp),
    },
    "zoom_for_display": {
        "zoom_center_x": int(zoom_center_x),
        "zoom_center_y": int(zoom_center_y),
        "zoom_size": int(zoom_size),
    },
}

# Generate readable text + JSON
lines = []
lines.append("# Gibbs Explorer Export")
lines.append(f"timestamp: {config['timestamp']}")
lines.append(f"background.source_name: {config['background']['source_name']}")
lines.append(f"background.size: {H}x{W}")
lines.append(f"background.kept_original_size: {config['background']['kept_original_size']}")
lines.append("")
lines.append("## Lesion")
for k, v in config["lesion"].items():
    lines.append(f"lesion.{k}: {v}")
lines.append("")
lines.append("## Gibbs")
for k, v in config["gibbs"].items():
    lines.append(f"gibbs.{k}: {v}")
lines.append("")
lines.append("## Zoom (for visualization only)")
for k, v in config["zoom_for_display"].items():
    lines.append(f"zoom.{k}: {v}")
lines.append("")
lines.append("## JSON (full)")
lines.append(json.dumps(config, indent=2))
cfg_txt = "\n".join(lines).encode("utf-8")

# Pack into ZIP
zip_buf = BytesIO()
with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("background_original.png", img_bytes(img_bg))
    zf.writestr("gibbs_overlay.png", img_bytes(img_overlay))
    zf.writestr("lesion_alpha_clean.png", img_bytes(img_alpha_clean))
    zf.writestr("lesion_alpha_trunc.png", img_bytes(img_alpha_trunc))

    # Save profiles if they exist
    if show_profiles and fig_row is not None and fig_col is not None:
        buf_row = BytesIO()
        fig_row.savefig(buf_row, format="png", bbox_inches="tight")
        buf_row.seek(0)
        zf.writestr("profile_row.png", buf_row.getvalue())

        buf_col = BytesIO()
        fig_col.savefig(buf_col, format="png", bbox_inches="tight")
        buf_col.seek(0)
        zf.writestr("profile_col.png", buf_col.getvalue())

    zf.writestr("config.txt", cfg_txt)
zip_bytes = zip_buf.getvalue()

st.download_button(
    label="Download all (original + synthetic + masks + config)",
    data=zip_bytes,
    file_name="gibbs_full_export.zip",
    mime="application/zip",
)

st.caption(
    "Export includes: background_original.png, gibbs_overlay.png, lesion_alpha_clean.png, "
    "lesion_alpha_trunc.png, and config.txt."
)
