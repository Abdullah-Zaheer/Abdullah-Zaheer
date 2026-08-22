#!/usr/bin/env python3
"""
Dynamic Terminal GitHub Profile Banner Generator
Author: Pair-programmed for Abdullah Zaheer (@Abdullah-Zaheer)
"""

import os
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageDraw
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.ndimage import binary_fill_holes

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRATCH_DIR = os.path.join(WORKSPACE_DIR, '.cache_data')
os.makedirs(SCRATCH_DIR, exist_ok=True)

IMG_PATH = '/Users/abdullahzaheer/.gemini/antigravity-ide/brain/035750bf-e298-44f0-ade7-0fbd033fc780/.user_uploaded/media_1787409562685.jpg'

TOTAL_W = 1180
TOTAL_H = 610

PORTRAIT_W = 280
PORTRAIT_H = 320

# 1. Load and process image
img = Image.open(IMG_PATH).convert('RGB')
w, h = img.size
crop_box = (int(w * 0.05), int(h * 0.05), int(w * 0.95), int(h * 0.88))
cropped = img.crop(crop_box)
resized = cropped.resize((PORTRAIT_W, PORTRAIT_H), Image.Resampling.LANCZOS)
arr = np.array(resized, dtype=np.float32)
r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]

y_coords, x_coords = np.mgrid[0:PORTRAIT_H, 0:PORTRAIT_W]

is_in_bounds = np.zeros((PORTRAIT_H, PORTRAIT_W), dtype=bool)
for y in range(PORTRAIT_H):
    if y < 55:
        x_min, x_max = 140, 140
    elif y < 75:
        x_min, x_max = 115, 160
    elif y < 115:
        x_min, x_max = 105, 162
    elif y < 140:
        x_min, x_max = 104, 166
    elif y < 185:
        x_min, x_max = 100, 172
    elif y < 210:
        x_min, x_max = 75, 205
    elif y < 245:
        x_min, x_max = 45, 235
    elif y < 280:
        x_min, x_max = 28, 248
    else:
        x_min, x_max = 25, 245
    is_in_bounds[y, x_min:x_max] = True

is_gold_stone = (r > 95) & (g > 65) & ((r - b) > 40) & ((g - b) > 15)
is_face_core = (y_coords >= 100) & (y_coords <= 170) & (x_coords >= 110) & (x_coords <= 160)
fg_mask = is_in_bounds & ((~is_gold_stone) | is_face_core)
fg_mask = binary_fill_holes(fg_mask) & is_in_bounds

# Contrast & Sharpening
gray = resized.convert('L')
gray_auto = ImageOps.autocontrast(gray, cutoff=1)
gray_sharp = gray_auto.filter(ImageFilter.UnsharpMask(radius=3, percent=140))
enhancer = ImageEnhance.Contrast(gray_sharp)
gray_enhanced = enhancer.enhance(1.35)
gray_arr = np.array(gray_enhanced, dtype=np.float32)

dark_input = gray_arr.copy()
dark_input[~fg_mask] = 0.0

light_input = 255.0 - gray_arr
light_input[~fg_mask] = 0.0

def floyd_steinberg_dither(img_2d, mask):
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
            if not mask[y, x]:
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

dither_dark = floyd_steinberg_dither(dark_input, fg_mask)
dither_light = floyd_steinberg_dither(light_input, fg_mask)
print(f"Dark mode portrait dots: {np.sum(dither_dark)}")
print(f"Light mode portrait dots: {np.sum(dither_light)}")

# 2. Logos & Travellers Generation
# Selected for Software Engineer (Python), Security Researcher (Security Shield), System Architect (Docker Whale)
N_TRAVELLERS = 750

def render_python_logo():
    img = Image.new('L', (PORTRAIT_W, PORTRAIT_H), 0)
    draw = ImageDraw.Draw(img)
    cx, cy = PORTRAIT_W // 2, PORTRAIT_H // 2
    draw.rounded_rectangle([cx - 55, cy - 65, cx + 25, cy - 25], radius=15, fill=255)
    draw.rounded_rectangle([cx + 5, cy - 65, cx + 55, cy + 25], radius=15, fill=255)
    draw.rectangle([cx - 20, cy - 25, cx + 25, cy + 5], fill=255)
    draw.ellipse([cx - 30, cy - 50, cx - 20, cy - 40], fill=0)
    draw.rounded_rectangle([cx - 25, cy + 25, cx + 55, cy + 65], radius=15, fill=255)
    draw.rounded_rectangle([cx - 55, cy - 25, cx - 5, cy + 65], radius=15, fill=255)
    draw.rectangle([cx - 25, cy - 5, cx + 20, cy + 25], fill=255)
    draw.ellipse([cx + 20, cy + 40, cx + 30, cy + 50], fill=0)
    return np.array(img) > 128

def render_security_shield():
    img = Image.new('L', (PORTRAIT_W, PORTRAIT_H), 0)
    draw = ImageDraw.Draw(img)
    cx, cy = PORTRAIT_W // 2, PORTRAIT_H // 2 - 5
    shield_pts = [
        (cx - 65, cy - 65), (cx + 65, cy - 65), (cx + 65, cy - 10),
        (cx + 45, cy + 45), (cx, cy + 75), (cx - 45, cy + 45), (cx - 65, cy - 10),
    ]
    draw.polygon(shield_pts, fill=255)
    inner_pts = [
        (cx - 50, cy - 50), (cx + 50, cy - 50), (cx + 50, cy - 10),
        (cx + 35, cy + 35), (cx, cy + 60), (cx - 35, cy + 35), (cx - 50, cy - 10),
    ]
    draw.polygon(inner_pts, fill=0)
    draw.rounded_rectangle([cx - 22, cy - 35, cx + 22, cy - 5], radius=10, fill=255)
    draw.rounded_rectangle([cx - 12, cy - 25, cx + 12, cy - 5], radius=6, fill=0)
    draw.rounded_rectangle([cx - 28, cy - 10, cx + 28, cy + 32], radius=6, fill=255)
    draw.ellipse([cx - 6, cy + 2, cx + 6, cy + 14], fill=0)
    draw.polygon([(cx - 4, cy + 10), (cx + 4, cy + 10), (cx + 6, cy + 24), (cx - 6, cy + 24)], fill=0)
    return np.array(img) > 128

def render_docker_logo():
    img = Image.new('L', (PORTRAIT_W, PORTRAIT_H), 0)
    draw = ImageDraw.Draw(img)
    cx, cy = PORTRAIT_W // 2, PORTRAIT_H // 2
    cw, ch, gap = 14, 12, 3
    for i in range(5):
        bx = cx - 45 + i * (cw + gap)
        by = cy - 20
        draw.rectangle([bx, by, bx + cw, by + ch], fill=255)
    for i in range(4):
        bx = cx - 28 + i * (cw + gap)
        by = cy - 20 - ch - gap
        draw.rectangle([bx, by, bx + cw, by + ch], fill=255)
    bx = cx - 11
    by = cy - 20 - (ch + gap) * 2
    draw.rectangle([bx, by, bx + cw, by + ch], fill=255)
    hull = [
        (cx - 75, cy - 5), (cx - 70, cy + 15), (cx - 50, cy + 35),
        (cx + 20, cy + 40), (cx + 65, cy + 25), (cx + 85, cy + 5),
        (cx + 80, cy - 15), (cx + 70, cy - 10), (cx + 65, cy + 5),
        (cx + 45, cy + 5), (cx + 45, cy - 5), (cx - 50, cy - 5), (cx - 75, cy - 5)
    ]
    draw.polygon(hull, fill=255)
    draw.ellipse([cx - 58, cy + 8, cx - 52, cy + 14], fill=0)
    draw.ellipse([cx + 55, cy - 25, cx + 62, cy - 18], fill=255)
    draw.ellipse([cx + 68, cy - 35, cx + 75, cy - 28], fill=255)
    return np.array(img) > 128

def sample_pts(mask, n):
    ys, xs = np.where(mask)
    indices = np.random.choice(len(xs), n, replace=(len(xs) < n))
    return np.column_stack([xs[indices], ys[indices]]).astype(np.float32)

np.random.seed(42)
pts1 = sample_pts(render_python_logo(), N_TRAVELLERS)
pts2 = sample_pts(render_security_shield(), N_TRAVELLERS)
pts3 = sample_pts(render_docker_logo(), N_TRAVELLERS)

# Hungarian mapping
col2 = linear_sum_assignment(cdist(pts1, pts2))[1]
pts2_m = pts2[col2]
col3 = linear_sum_assignment(cdist(pts2_m, pts3))[1]
pts3_m = pts3[col3]
print("Optimal transport matching done.")

# 3. Build SVGs
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
    kt_str = "0;0.211;0.303;0.444;0.535;0.676;0.768;0.908;1.0"
    
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
        svg.append(f'      <animateTransform attributeName="transform" type="translate" dur="14.2s" repeatCount="indefinite" keyTimes="{kt_str}" values="0 0; 0 0; {avg_dx:.1f} {avg_dy:.1f}; {avg_dx:.1f} {avg_dy:.1f}; {avg_dx*0.5:.1f} {avg_dy*0.5:.1f}; {avg_dx*0.5:.1f} {avg_dy*0.5:.1f}; {avg_dx:.1f} {avg_dy:.1f}; 0 0; 0 0"/>')
        svg.append(f'      <path d="{d_str}" stroke="{portrait_dot_color}" stroke-width="1.2" shape-rendering="crispEdges">')
        svg.append(f'        <animate attributeName="opacity" dur="14.2s" repeatCount="indefinite" keyTimes="{kt_str}" values="1;1;0;0;0;0;0;1;1"/>')
        svg.append(f'      </path>')
        svg.append(f'    </g>')
        
    svg.append('  </g>')
    
    # Layer 2: Travellers
    svg.append(f'  <!-- Layer 2: Morphing Logo Travellers -->')
    svg.append(f'  <g id="travellers-layer" transform="translate({px_base}, {py_base})">')
    trav_op_vals = "0;0;1;1;1;1;1;0;0"
    
    for i in range(N_TRAVELLERS):
        x1, y1 = pts1[i]
        x2, y2 = pts2_m[i]
        x3, y3 = pts3_m[i]
        
        x_vals = f"{x1:.1f};{x1:.1f};{x1:.1f};{x1:.1f};{x2:.1f};{x2:.1f};{x3:.1f};{x3:.1f};{x1:.1f}"
        y_vals = f"{y1:.1f};{y1:.1f};{y1:.1f};{y1:.1f};{y2:.1f};{y2:.1f};{y3:.1f};{y3:.1f};{y1:.1f}"
        
        svg.append(f'    <circle cx="{x1:.1f}" cy="{y1:.1f}" r="1.3" fill="{traveller_color}">')
        svg.append(f'      <animate attributeName="cx" dur="14.2s" repeatCount="indefinite" keyTimes="{kt_str}" values="{x_vals}"/>')
        svg.append(f'      <animate attributeName="cy" dur="14.2s" repeatCount="indefinite" keyTimes="{kt_str}" values="{y_vals}"/>')
        svg.append(f'      <animate attributeName="opacity" dur="14.2s" repeatCount="indefinite" keyTimes="{kt_str}" values="{trav_op_vals}"/>')
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
    
    # Accurate Data Rows with user's verified links & stack
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

print("SVGs successfully updated with your exact handles!")
