"""Stage 4.2 — Fast Screening: Identify linear defect candidates.

Scans all 781 def_front images using Sobel gradient statistics on the
interior casting face region. No per-component loops.

Usage:
    D:\\FYP\\.venv\\Scripts\\python.exe scripts/screen_linear_defects.py
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

DEF_DIR = (
    _PROJECT_ROOT / "data" / "raw" / "casting_512x512"
    / "casting_512x512" / "def_front"
)
SCREEN_DIR = _PROJECT_ROOT / "outputs" / "evaluation" / "defect_dataset_analysis" / "screening"


def create_interior_mask(size: int = 512) -> np.ndarray:
    """Annular mask r=55..195 from center. Excludes outer boundary flash and central bore."""
    y, x = np.ogrid[:size, :size]
    cx, cy = size // 2, size // 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return ((r > 55) & (r < 195)).astype(np.uint8)


def compute_scores(gray_arr: np.ndarray, interior_mask: np.ndarray) -> dict:
    """Fast gradient-based scoring. Fully vectorized, no Python loops over pixels."""
    masked = gray_arr.astype(np.float32) * interior_mask

    sx = ndimage.sobel(masked, axis=1)
    sy = ndimage.sobel(masked, axis=0)
    mag = np.sqrt(sx ** 2 + sy ** 2)
    mag_masked = mag * interior_mask

    vals = mag_masked[interior_mask > 0]
    if len(vals) == 0:
        return {"p99": 0, "p95": 0, "max_grad": 0, "mean_grad": 0,
                "strong_ratio": 0, "very_strong_count": 0}

    p95 = float(np.percentile(vals, 95))
    p99 = float(np.percentile(vals, 99))
    mx = float(np.max(vals))
    mn = float(np.mean(vals))

    # Count of pixels above p99
    very_strong = np.sum(mag_masked > p99)

    # Ratio of p99 to p95 — higher means sharper, more localized features
    sharpness = p99 / max(p95, 1e-6)

    # Directional coherence: ratio of dominant gradient direction
    # Linear defects produce strongly directional gradients
    strong_mask = mag_masked > p95
    if np.sum(strong_mask) > 10:
        sx_strong = sx[strong_mask]
        sy_strong = sy[strong_mask]
        angles = np.arctan2(sy_strong, sx_strong)
        # Use circular variance as coherence measure
        cos_sum = np.abs(np.mean(np.cos(2 * angles)))
        sin_sum = np.abs(np.mean(np.sin(2 * angles)))
        dir_coherence = float(np.sqrt(cos_sum ** 2 + sin_sum ** 2))
    else:
        dir_coherence = 0.0

    return {
        "p99": round(p99, 2),
        "p95": round(p95, 2),
        "max_grad": round(mx, 2),
        "mean_grad": round(mn, 2),
        "sharpness": round(sharpness, 3),
        "dir_coherence": round(dir_coherence, 4),
        "very_strong_count": int(very_strong),
    }


def main():
    print("=" * 70)
    print("STAGE 4.2 — Linear Defect Candidate Screening (Vectorized)")
    print("=" * 70)

    SCREEN_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(list(DEF_DIR.glob("*.jpeg")))
    print(f"Scanning {len(files)} defect images...")

    interior_mask = create_interior_mask(512)

    t0 = time.perf_counter()
    results = []
    for idx, f in enumerate(files):
        if idx % 200 == 0:
            print(f"  Processing {idx}/{len(files)}...")
        gray = np.array(Image.open(f).convert("L"))
        scores = compute_scores(gray, interior_mask)
        scores["filename"] = f.name
        scores["filepath"] = str(f)
        results.append(scores)

    elapsed = time.perf_counter() - t0
    print(f"  Processed {len(files)} images in {elapsed:.1f}s "
          f"({elapsed / len(files) * 1000:.0f}ms/image)")

    # Composite score: sharpness + dir_coherence (both indicative of linear features)
    for r in results:
        r["composite"] = round(r["sharpness"] * 0.6 + r["dir_coherence"] * 0.4 + r["p99"] / 200, 2)

    results.sort(key=lambda x: x["composite"], reverse=True)

    # Top 60 contact sheet
    top_n = min(60, len(results))
    thumb = 180
    cols = 6
    rows = (top_n + cols - 1) // cols
    pad = 8
    lbl_h = 20
    hdr_h = 40
    cw, ch = thumb + pad, thumb + lbl_h + pad
    gw, gh = cols * cw + pad, hdr_h + rows * ch + pad

    grid = Image.new("RGB", (gw, gh), (25, 25, 28))
    d = ImageDraw.Draw(grid)
    d.text((gw // 2, 12), f"Top {top_n} Linear Defect Candidates (Interior Face)",
           fill=(240, 240, 240), anchor="mt")

    for i, r in enumerate(results[:top_n]):
        ri, ci = i // cols, i % cols
        x, y = pad + ci * cw, hdr_h + ri * ch
        img = Image.open(r["filepath"]).resize((thumb, thumb), Image.LANCZOS)
        grid.paste(img, (x, y))
        sn = r["filename"].replace("cast_def_0_", "").replace(".jpeg", "")
        d.text((x + 2, y + thumb + 2), f"#{i+1} {sn} C:{r['composite']:.2f}",
               fill=(200, 200, 200))

    grid.save(SCREEN_DIR / "screening_top60.png")
    print(f"  Saved screening_top60.png")

    # Top 30 larger thumbnails
    top30 = results[:min(30, len(results))]
    thumb2 = 220
    cols2 = 5
    rows2 = (len(top30) + cols2 - 1) // cols2
    cw2, ch2 = thumb2 + pad, thumb2 + lbl_h + 5 + pad
    gw2, gh2 = cols2 * cw2 + pad, hdr_h + rows2 * ch2 + pad

    grid2 = Image.new("RGB", (gw2, gh2), (25, 25, 28))
    d2 = ImageDraw.Draw(grid2)
    d2.text((gw2 // 2, 12), "Top 30 Linear Defect Candidates", fill=(240, 240, 240), anchor="mt")

    for i, r in enumerate(top30):
        ri, ci = i // cols2, i % cols2
        x, y = pad + ci * cw2, hdr_h + ri * ch2
        img = Image.open(r["filepath"]).resize((thumb2, thumb2), Image.LANCZOS)
        grid2.paste(img, (x, y))
        sn = r["filename"].replace("cast_def_0_", "").replace(".jpeg", "")
        d2.text((x + 2, y + thumb2 + 2),
                f"#{i+1} {sn} S:{r['sharpness']:.2f} D:{r['dir_coherence']:.2f}",
                fill=(200, 200, 200))

    grid2.save(SCREEN_DIR / "screening_top30.png")
    print(f"  Saved screening_top30.png")

    # Print top 30
    print(f"\nTop 30 by composite score:")
    print(f"{'Rank':>4} {'Filename':<28} {'Composite':>10} {'Sharpness':>10} "
          f"{'DirCoher':>10} {'P99Grad':>8}")
    for i, r in enumerate(top30):
        print(f"{i+1:>4} {r['filename']:<28} {r['composite']:>10.2f} "
              f"{r['sharpness']:>10.3f} {r['dir_coherence']:>10.4f} "
              f"{r['p99']:>8.1f}")

    # Save CSV
    csv_path = SCREEN_DIR / "all_scores.csv"
    fields = ["filename", "composite", "sharpness", "dir_coherence",
              "p99", "p95", "max_grad", "mean_grad", "very_strong_count"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"\n  Saved all_scores.csv ({len(results)} rows)")
    print("Done.")


if __name__ == "__main__":
    main()
