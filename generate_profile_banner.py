#!/usr/bin/env python3
"""
Dynamic Terminal GitHub Profile Banner Generator (v3 - Ultra Sharp & Extended Docker Timer)
Author: Pair-programmed for Abdullah Zaheer (@Abdullah-Zaheer)
"""

import os
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageDraw
from scipy.ndimage import binary_erosion
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRATCH_DIR = os.path.join(WORKSPACE_DIR, '.cache_data')
os.makedirs(SCRATCH_DIR, exist_ok=True)

# Transparent input photo
IMG_PATH = '/Users/abdullahzaheer/.gemini/antigravity-ide/brain/035750bf-e298-44f0-ade7-0fbd033fc780/.user_uploaded/media_1787410737668.png'

TOTAL_W = 1180
TOTAL_H = 610

PORTRAIT_W = 280
PORTRAIT_H = 320

# 1. Load and process image
img = Image.open(IMG_PATH)
alpha = np.array(img.split()[3])
ys, xs = np.where(alpha > 20)
min_y, max_y = min(ys), max(ys)
min_x, max_x = min(xs), max(xs)

cropped = img.crop((min_x, min_y, max_x, max_y))
resized = cropped.resize((PORTRAIT_W, PORTRAIT_H), Image.Resampling.LANCZOS)

r, g, b, a = resized.split()
rgb = Image.merge('RGB', (r, g, b))
mask = np.array(a) > 40

gray = rgb.convert('L')
gray_auto = ImageOps.autocontrast(gray, cutoff=1)
gray_sharp = gray_auto.filter(ImageFilter.UnsharpMask(radius=3, percent=140))
enhancer = ImageEnhance.Contrast(gray_sharp)
gray_enhanced = enhancer.enhance(1.3)
gray_arr = np.array(gray_enhanced, dtype=np.float32)

dark_input = gray_arr.copy()
dark_input[~mask] = 0.0

light_input = 255.0 - gray_arr
light_input[~mask] = 0.0

def floyd_steinberg_dither(img_2d, mask_2d):
    h, w = img_2d.shape
    arr = img_2d.copy().astype(np.float32)
    out = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        if y % 2 == 0:
            x_range = range(w)
            direction = 1
        else:
            x_range = range(w - 1, -1, -1)
            direction = -1
        for x in x_range:
            if not mask_2d[y, x]:
                arr[y, x] = 0
                out[y, x] = 0
                continue
            old_val = arr[y, x]
            new_val = 255.0 if old_val > 127.5 else 0.0
            out[y, x] = 1 if new_val == 255.0 else 0
            err = old_val - new_val
            if direction == 1:
                if x + 1 < w:
                    arr[y, x + 1] += err * (7.0 / 16.0)
                if y + 1 < h:
                    if x - 1 >= 0:
                        arr[y + 1, x - 1] += err * (3.0 / 16.0)
                    arr[y + 1, x] += err * (5.0 / 16.0)
                    if x + 1 < w:
                        arr[y + 1, x + 1] += err * (1.0 / 16.0)
            else:
                if x - 1 >= 0:
                    arr[y, x - 1] += err * (7.0 / 16.0)
                if y + 1 < h:
                    if x + 1 < w:
                        arr[y + 1, x + 1] += err * (3.0 / 16.0)
                    arr[y + 1, x] += err * (5.0 / 16.0)
                    if x - 1 >= 0:
                        arr[y + 1, x - 1] += err * (1.0 / 16.0)
    return out

dither_dark = floyd_steinberg_dither(dark_input, mask)
dither_light = floyd_steinberg_dither(light_input, mask)
print(f"Dark mode portrait dots: {np.sum(dither_dark)}")
print(f"Light mode portrait dots: {np.sum(dither_light)}")

# 2. Ultra-Sharp HD Morphing Logos (1,000 Traveller Dots with Edge Contour Tracing)
N_TRAVELLERS = 1000
SCALE = 4
SW, SH = PORTRAIT_W * SCALE, PORTRAIT_H * SCALE

def create_python_mask():
    img = Image.new('L', (SW, SH), 0)
    draw = ImageDraw.Draw(img)
    cx, cy = SW // 2, SH // 2
    
    # Upper Snake
    draw.rounded_rectangle([cx - 56*SCALE, cy - 68*SCALE, cx + 26*SCALE, cy - 24*SCALE], radius=16*SCALE, fill=255)
    draw.rounded_rectangle([cx + 4*SCALE, cy - 68*SCALE, cx + 56*SCALE, cy + 24*SCALE], radius=16*SCALE, fill=255)
    draw.rectangle([cx - 22*SCALE, cy - 24*SCALE, cx + 24*SCALE, cy + 6*SCALE], fill=255)
    draw.ellipse([cx - 32*SCALE, cy - 54*SCALE, cx - 18*SCALE, cy - 40*SCALE], fill=0)
    
    # Lower Snake
    draw.rounded_rectangle([cx - 26*SCALE, cy + 24*SCALE, cx + 56*SCALE, cy + 68*SCALE], radius=16*SCALE, fill=255)
    draw.rounded_rectangle([cx - 56*SCALE, cy - 24*SCALE, cx - 4*SCALE, cy + 68*SCALE], radius=16*SCALE, fill=255)
    draw.rectangle([cx - 24*SCALE, cy - 6*SCALE, cx + 22*SCALE, cy + 24*SCALE], fill=255)
    draw.ellipse([cx + 18*SCALE, cy + 40*SCALE, cx + 32*SCALE, cy + 54*SCALE], fill=0)
    
    return np.array(img.resize((PORTRAIT_W, PORTRAIT_H), Image.Resampling.LANCZOS)) > 120

def create_shield_mask():
    img = Image.new('L', (SW, SH), 0)
    draw = ImageDraw.Draw(img)
    cx, cy = SW // 2, SH // 2 - 8*SCALE
    
    shield_outer = [
        (cx - 72*SCALE, cy - 72*SCALE), (cx + 72*SCALE, cy - 72*SCALE),
        (cx + 72*SCALE, cy - 10*SCALE), (cx + 52*SCALE, cy + 52*SCALE),
        (cx, cy + 86*SCALE),
        (cx - 52*SCALE, cy + 52*SCALE), (cx - 72*SCALE, cy - 10*SCALE)
    ]
    draw.polygon(shield_outer, fill=255)
    
    shield_inner = [
        (cx - 54*SCALE, cy - 54*SCALE), (cx + 54*SCALE, cy - 54*SCALE),
        (cx + 54*SCALE, cy - 10*SCALE), (cx + 38*SCALE, cy + 38*SCALE),
        (cx, cy + 66*SCALE),
        (cx - 38*SCALE, cy + 38*SCALE), (cx - 54*SCALE, cy - 10*SCALE)
    ]
    draw.polygon(shield_inner, fill=0)
    
    draw.rounded_rectangle([cx - 24*SCALE, cy - 38*SCALE, cx + 24*SCALE, cy - 2*SCALE], radius=12*SCALE, fill=255)
    draw.rounded_rectangle([cx - 14*SCALE, cy - 28*SCALE, cx + 14*SCALE, cy - 2*SCALE], radius=7*SCALE, fill=0)
    draw.rounded_rectangle([cx - 30*SCALE, cy - 8*SCALE, cx + 30*SCALE, cy + 38*SCALE], radius=8*SCALE, fill=255)
    draw.ellipse([cx - 7*SCALE, cy + 6*SCALE, cx + 7*SCALE, cy + 20*SCALE], fill=0)
    draw.polygon([(cx - 5*SCALE, cy + 16*SCALE), (cx + 5*SCALE, cy + 16*SCALE), (cx + 7*SCALE, cy + 30*SCALE), (cx - 7*SCALE, cy + 30*SCALE)], fill=0)
    
    return np.array(img.resize((PORTRAIT_W, PORTRAIT_H), Image.Resampling.LANCZOS)) > 120

def create_docker_mask():
    img = Image.new('L', (SW, SH), 0)
    draw = ImageDraw.Draw(img)
    cx, cy = SW // 2, SH // 2
    cw, ch, gap = 16*SCALE, 14*SCALE, 3*SCALE
    
    for i in range(5):
        bx = cx - 50*SCALE + i * (cw + gap)
        by = cy - 24*SCALE
        draw.rectangle([bx, by, bx + cw, by + ch], fill=255)
    for i in range(4):
        bx = cx - 31*SCALE + i * (cw + gap)
        by = cy - 24*SCALE - ch - gap
        draw.rectangle([bx, by, bx + cw, by + ch], fill=255)
    bx = cx - 12*SCALE
    by = cy - 24*SCALE - (ch + gap) * 2
    draw.rectangle([bx, by, bx + cw, by + ch], fill=255)
    
    hull = [
        (cx - 82*SCALE, cy - 8*SCALE), (cx - 76*SCALE, cy + 18*SCALE), (cx - 55*SCALE, cy + 40*SCALE),
        (cx + 25*SCALE, cy + 44*SCALE), (cx + 72*SCALE, cy + 28*SCALE), (cx + 92*SCALE, cy + 8*SCALE),
        (cx + 86*SCALE, cy - 16*SCALE), (cx + 75*SCALE, cy - 10*SCALE), (cx + 68*SCALE, cy + 6*SCALE),
        (cx + 48*SCALE, cy + 6*SCALE), (cx + 48*SCALE, cy - 8*SCALE), (cx - 56*SCALE, cy - 8*SCALE), (cx - 82*SCALE, cy - 8*SCALE)
    ]
    draw.polygon(hull, fill=255)
    draw.ellipse([cx - 64*SCALE, cy + 8*SCALE, cx - 56*SCALE, cy + 16*SCALE], fill=0)
    draw.ellipse([cx + 60*SCALE, cy - 30*SCALE, cx + 68*SCALE, cy - 22*SCALE], fill=255)
    draw.ellipse([cx + 75*SCALE, cy - 40*SCALE, cx + 84*SCALE, cy - 31*SCALE], fill=255)
    
    return np.array(img.resize((PORTRAIT_W, PORTRAIT_H), Image.Resampling.LANCZOS)) > 120

def sample_sharp_pts(mask, n=N_TRAVELLERS):
    eroded = binary_erosion(mask, iterations=1)
    edges = mask & (~eroded)
    interior = eroded
    
    edge_ys, edge_xs = np.where(edges)
    int_ys, int_xs = np.where(interior)
    
    n_edge = int(n * 0.55)
    n_int = n - n_edge
    
    if len(edge_xs) > 0:
        edge_idx = np.linspace(0, len(edge_xs) - 1, n_edge, dtype=int)
        edge_pts = np.column_stack([edge_xs[edge_idx], edge_ys[edge_idx]]).astype(np.float32)
    else:
        edge_pts = np.empty((0, 2), dtype=np.float32)
        
    if len(int_xs) > 0:
        int_idx = np.random.choice(len(int_xs), n_int, replace=(len(int_xs) < n_int))
        int_pts = np.column_stack([int_xs[int_idx], int_ys[int_idx]]).astype(np.float32)
    else:
        int_pts = np.empty((0, 2), dtype=np.float32)
        
    return np.vstack([edge_pts, int_pts])

np.random.seed(42)
pts1 = sample_sharp_pts(create_python_mask(), N_TRAVELLERS)
pts2 = sample_sharp_pts(create_shield_mask(), N_TRAVELLERS)
pts3 = sample_sharp_pts(create_docker_mask(), N_TRAVELLERS)

# Hungarian mapping
col2 = linear_sum_assignment(cdist(pts1, pts2))[1]
pts2_m = pts2[col2]
col3 = linear_sum_assignment(cdist(pts2_m, pts3))[1]
pts3_m = pts3[col3]
print("Optimal transport matching done.")

# 3. Build SVGs with Extended Docker Timer (17.0s total cycle)
def build_svg_file(theme="dark"):
    is_dark = (theme == "dark")
    active_dither = dither_dark if is_dark else dither_light
    
    bg_color = "#0A101F" if is_dark else "#F8FAFC"
    border_color = "#22D3EE" if is_dark else "#0891B2"
    panel_bg = "#070D1A" if is_dark else "#FFFFFF"
    card_border = "#1E293B" if is_dark else "#E2E8F0"
    portrait_dot_color = "#A78BFA" if is_dark else "#7C3AED"
    traveller_color = "#22D3EE" if is_dark else "#0891B2"
    text_primary = "#F8FAFC" if is_dark else "#0F172A"
    text_secondary = "#94A3B8" if is_dark else "#475569"
    text_cyan = "#22D3EE" if is_dark else "#0891B2"
    text_accent = "#10B981"
    leader_color = "#334155" if is_dark else "#CBD5E1"
    
    py_dots, px_dots = np.where(active_dither == 1)
    tot_dots = len(px_dots)
    
    np.random.seed(1337)
    intro_groups = np.random.randint(0, 60, size=tot_dots)
    
    cx, cy = PORTRAIT_W / 2.0, PORTRAIT_H / 2.0
    
    # 17.0s Total Loop:
    # 0s - 3.0s (3.0s): Portrait
    # 3.0s - 4.2s (1.2s): Morph to Python
    # 4.2s - 6.2s (2.0s): Python Hold
    # 6.2s - 7.4s (1.2s): Morph to Shield
    # 7.4s - 9.4s (2.0s): Shield Hold
    # 9.4s - 10.6s (1.2s): Morph to Docker
    # 10.6s - 15.0s (4.4s EXTENDED DOCKER HOLD!)
    # 15.0s - 16.2s (1.2s): Return dissolve
    # 16.2s - 17.0s (0.8s): Settle to Portrait
    
    # Normalized keyTimes:
    # [0.0, 0.176, 0.247, 0.365, 0.435, 0.553, 0.624, 0.882, 0.953, 1.0]
    kt_str = "0;0.176;0.247;0.365;0.435;0.553;0.624;0.882;0.953;1.0"
    
    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {TOTAL_W} {TOTAL_H}" width="{TOTAL_W}" height="{TOTAL_H}">')
    svg.append('<defs>')
    svg.append('  <style>')
    svg.append('    @keyframes pulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.3; transform: scale(0.85); } }')
    svg.append('    .live-dot { animation: pulse 1.8s ease-in-out infinite; transform-origin: 605px 92px; }')
    svg.append('    .mono-hdr { font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; font-size: 13px; font-weight: 600; }')
    svg.append('    .mono-row { font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; font-size: 13px; }')
    svg.append('    .mono-pill { font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; font-size: 12.5px; font-weight: 600; }')
    svg.append('  </style>')
    svg.append('</defs>')
    
    # Background
    svg.append(f'  <rect width="{TOTAL_W}" height="{TOTAL_H}" rx="14" fill="{bg_color}" stroke="{border_color}" stroke-width="1.5" stroke-opacity="0.35"/>')
    
    # Window Top Bar
    svg.append(f'  <rect x="0" y="0" width="{TOTAL_W}" height="48" rx="14" fill="{card_border}" fill-opacity="0.25"/>')
    svg.append(f'  <line x1="0" y1="48" x2="{TOTAL_W}" y2="48" stroke="{card_border}" stroke-width="1"/>')
    svg.append('  <circle cx="28" cy="24" r="6" fill="#EF4444"/>')
    svg.append('  <circle cx="48" cy="24" r="6" fill="#F59E0B"/>')
    svg.append('  <circle cx="68" cy="24" r="6" fill="#10B981"/>')
    svg.append(f'  <text x="100" y="28" fill="{text_secondary}" class="mono-hdr">profile.sh <tspan fill="{text_cyan}">--live</tspan></text>')
    svg.append(f'  <text x="{TOTAL_W - 35}" y="28" text-anchor="end" fill="{text_secondary}" class="mono-hdr" opacity="0.6">sys_arch::x86_64</text>')
    
    # Left Frame: VISUAL.MAP
    frame_x, frame_y, frame_w, frame_h = 32, 68, 415, 510
    svg.append(f'  <rect x="{frame_x}" y="{frame_y}" width="{frame_w}" height="{frame_h}" rx="10" fill="{panel_bg}" stroke="{card_border}" stroke-width="1.2"/>')
    svg.append(f'  <rect x="{frame_x}" y="{frame_y}" width="{frame_w}" height="38" rx="10" fill="{card_border}" fill-opacity="0.2"/>')
    svg.append(f'  <line x1="{frame_x}" y1="{frame_y+38}" x2="{frame_x+frame_w}" y2="{frame_y+38}" stroke="{card_border}" stroke-width="1"/>')
    svg.append(f'  <text x="{frame_x+18}" y="{frame_y+24}" fill="{text_cyan}" class="mono-hdr" letter-spacing="1">VISUAL.MAP</text>')
    svg.append(f'  <text x="{frame_x+frame_w-18}" y="{frame_y+24}" text-anchor="end" fill="{text_secondary}" font-family="monospace" font-size="11">DITHER:1-BIT RES:280x320</text>')
    
    # Sub-status inside portrait frame
    svg.append(f'  <text x="{frame_x+18}" y="{frame_y+frame_h-18}" fill="{text_secondary}" font-family="monospace" font-size="11.5">LAYER: <tspan fill="{portrait_dot_color}">PORTRAIT</tspan> ⇄ <tspan fill="{text_cyan}">MORPH_VEC</tspan></text>')
    svg.append(f'  <text x="{frame_x+frame_w-18}" y="{frame_y+frame_h-18}" text-anchor="end" fill="{text_accent}" font-family="monospace" font-size="11.5">SYNC: OK</text>')

    # Portrait Placement
    px_base = frame_x + (frame_w - PORTRAIT_W) // 2
    py_base = frame_y + 60
    
    # Layer 1: Portrait Dots
    svg.append(f'  <!-- Layer 1: Portrait Dither Dots -->')
    svg.append(f'  <g id="portrait-layer" transform="translate({px_base}, {py_base})">')
    
    for g_id in range(60):
        mask_g = (intro_groups == g_id)
        if not np.any(mask_g):
            continue
        g_xs = px_dots[mask_g]
        g_ys = py_dots[mask_g]
        
        path_parts = [f'M{x},{y}h1.2' for x, y in zip(g_xs, g_ys)]
        d_str = " ".join(path_parts)
        
        avg_dx = (cx - np.mean(g_xs)) * 0.42
        avg_dy = (cy - np.mean(g_ys)) * 0.42
        intro_delay = round(g_id * 0.035, 3)
        
        svg.append(f'    <g>')
        svg.append(f'      <animate attributeName="opacity" from="0" to="1" dur="1.8s" begin="{intro_delay}s" fill="freeze"/>')
        svg.append(f'      <animateTransform attributeName="transform" type="translate" dur="17.0s" repeatCount="indefinite" keyTimes="{kt_str}" values="0 0; 0 0; {avg_dx:.1f} {avg_dy:.1f}; {avg_dx:.1f} {avg_dy:.1f}; {avg_dx*0.5:.1f} {avg_dy*0.5:.1f}; {avg_dx*0.5:.1f} {avg_dy*0.5:.1f}; {avg_dx:.1f} {avg_dy:.1f}; {avg_dx:.1f} {avg_dy:.1f}; 0 0; 0 0"/>')
        svg.append(f'      <path d="{d_str}" stroke="{portrait_dot_color}" stroke-width="1.2" shape-rendering="crispEdges">')
        svg.append(f'        <animate attributeName="opacity" dur="17.0s" repeatCount="indefinite" keyTimes="{kt_str}" values="1;1;0;0;0;0;0;0;1;1"/>')
        svg.append(f'      </path>')
        svg.append(f'    </g>')
        
    svg.append('  </g>')
    
    # Layer 2: Travellers (Ultra Sharp & Extended Docker Hold)
    svg.append(f'  <!-- Layer 2: Morphing Logo Travellers -->')
    svg.append(f'  <g id="travellers-layer" transform="translate({px_base}, {py_base})">')
    trav_op_vals = "0;0;1;1;1;1;1;1;0;0"
    
    for i in range(N_TRAVELLERS):
        x1, y1 = pts1[i]
        x2, y2 = pts2_m[i]
        x3, y3 = pts3_m[i]
        
        x_vals = f"{x1:.1f};{x1:.1f};{x1:.1f};{x1:.1f};{x2:.1f};{x2:.1f};{x3:.1f};{x3:.1f};{x1:.1f};{x1:.1f}"
        y_vals = f"{y1:.1f};{y1:.1f};{y1:.1f};{y1:.1f};{y2:.1f};{y2:.1f};{y3:.1f};{y3:.1f};{y1:.1f};{y1:.1f}"
        
        svg.append(f'    <circle cx="{x1:.1f}" cy="{y1:.1f}" r="1.3" fill="{traveller_color}">')
        svg.append(f'      <animate attributeName="cx" dur="17.0s" repeatCount="indefinite" keyTimes="{kt_str}" values="{x_vals}"/>')
        svg.append(f'      <animate attributeName="cy" dur="17.0s" repeatCount="indefinite" keyTimes="{kt_str}" values="{y_vals}"/>')
        svg.append(f'      <animate attributeName="opacity" dur="17.0s" repeatCount="indefinite" keyTimes="{kt_str}" values="{trav_op_vals}"/>')
        svg.append(f'    </circle>')
        
    svg.append('  </g>')
    
    # Right Frame: SYSTEM.INFO Readout
    info_x, info_y, info_w, info_h = 465, 68, 683, 510
    svg.append(f'  <rect x="{info_x}" y="{info_y}" width="{info_w}" height="{info_h}" rx="10" fill="{panel_bg}" stroke="{card_border}" stroke-width="1.2"/>')
    svg.append(f'  <rect x="{info_x}" y="{info_y}" width="{info_w}" height="38" rx="10" fill="{card_border}" fill-opacity="0.2"/>')
    svg.append(f'  <line x1="{info_x}" y1="{info_y+38}" x2="{info_x+info_w}" y2="{info_y+38}" stroke="{card_border}" stroke-width="1"/>')
    
    # Info Panel Header
    svg.append(f'  <text x="{info_x+18}" y="{info_y+24}" fill="{text_cyan}" class="mono-hdr" letter-spacing="1">SYSTEM.INFO</text>')
    svg.append(f'  <g class="live-dot"><circle cx="{info_x+140}" cy="{info_y+20}" r="4.5" fill="#EF4444"/></g>')
    svg.append(f'  <text x="{info_x+152}" y="{info_y+24}" fill="#EF4444" font-family="monospace" font-size="11.5" font-weight="bold">LIVE</text>')
    
    # User Pill Badge
    pill_w, pill_h = 165, 24
    pill_x = info_x + info_w - pill_w - 18
    pill_y = info_y + 7
    svg.append(f'  <rect x="{pill_x}" y="{pill_y}" width="{pill_w}" height="{pill_h}" rx="12" fill="{card_border}" stroke="{portrait_dot_color}" stroke-width="1"/>')
    svg.append(f'  <text x="{pill_x+pill_w/2}" y="{pill_y+16}" text-anchor="middle" fill="{portrait_dot_color}" class="mono-pill">@Abdullah-Zaheer</text>')
    
    # Data Rows
    rows = [
        ("Subject", "Abdullah Zaheer"),
        ("Role", "Software Engineer / Security Researcher"),
        ("Origin", "Islamabad, Pakistan"),
        ("Education", "BS Software Engineering"),
        ("Status", "Building + Securing + Shipping"),
        ("ToolChain", "VS Code · Linux · Docker · Wireshark · Git · GDB"),
        ("Core.Lang", "Python · Rust · C/C++ · Go · TypeScript · Bash"),
        ("Core.Systems", "Distributed Architecture · Low-Level Systems"),
        ("Core.Security", "AppSec · Cryptography · Network Forensics"),
        ("Core.Infra", "Docker · Kubernetes · AWS · Linux · CI/CD"),
        ("Core.Database", "PostgreSQL · Redis · MongoDB · SQLite"),
        ("Grid.LinkedIn", "in/abdullah-zaheer-se"),
        ("Grid.Instagram", "@_abdullahzaheer_"),
        ("Grid.Mail", "info@abdullahzaheer.me"),
        ("Grid.Portfolio", "abdullahzaheer.me"),
    ]
    
    row_start_y = info_y + 68
    row_spacing = 28.5
    label_x = info_x + 22
    val_x = info_x + info_w - 22
    
    for idx, (label, val) in enumerate(rows):
        cur_y = row_start_y + idx * row_spacing
        
        if label.startswith("Core."):
            lbl_col = text_cyan
            val_col = text_primary
        elif label.startswith("Grid."):
            lbl_col = portrait_dot_color
            val_col = text_secondary
        elif label == "Subject":
            lbl_col = text_cyan
            val_col = text_cyan
        elif label == "Status":
            lbl_col = text_secondary
            val_col = text_accent
        else:
            lbl_col = text_secondary
            val_col = text_primary
            
        svg.append(f'    <g>')
        svg.append(f'      <text x="{label_x}" y="{cur_y}" fill="{lbl_col}" class="mono-row">{label}</text>')
        svg.append(f'      <text x="{val_x}" y="{cur_y}" text-anchor="end" fill="{val_col}" class="mono-row">{val}</text>')
        svg.append(f'      <line x1="{label_x + len(label)*8.5 + 12}" y1="{cur_y - 4}" x2="{val_x - len(val)*7.8 - 12}" y2="{cur_y - 4}" stroke="{leader_color}" stroke-width="1" stroke-dasharray="2 5" opacity="0.6"/>')
        svg.append(f'    </g>')
        
    svg.append('</svg>')
    return "\n".join(svg)

# Build both dark and light versions
dark_svg = build_svg_file("dark")
light_svg = build_svg_file("light")

with open(os.path.join(WORKSPACE_DIR, 'dark.svg'), 'w', encoding='utf-8') as f:
    f.write(dark_svg)
with open(os.path.join(WORKSPACE_DIR, 'light.svg'), 'w', encoding='utf-8') as f:
    f.write(light_svg)

print("SVGs successfully updated with ultra-sharp logos and extended Docker timer!")
