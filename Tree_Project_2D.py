"""
MMA3001 - Dataset 1: 2D apple-tree segmentation and skeleton extraction
"""

# ======================== 1. SETTINGS ========================
from pathlib import Path
import csv
import re
import time
from datetime import datetime
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')       # Save figures without opening a window for each tree.
import matplotlib.pyplot as plt

MAX_PAIRS = None           # None processes the entire dataset.
FIRST_PREVIEW = "R10N07_Winter_12-10-12"  # Inspect the supplied pair first, if present.
MAX_FIGURE_PAIRS = None    # None saves comparisons for every pair.
SENSITIVITY_LIMIT = 20      # Threshold sweep on the first 20 development observations.
SAVE_MASK_IMAGES = False   # True also saves separate binary masks and skeleton PNGs.

INVALID_DEPTH_VALUES = [0]  # Confirm missing-depth codes with the data provider.

ROI = None
GREEN_THRESHOLD = 0.08
DEPTH_CONTRAST = 0.10       # Fraction of valid depth display range above row background.
DEPTH_DIRECTION = 1         # +1 fits supplied winter display; -1 for reversed encoding.
DENSE_ROW_FRACTION = 0.40   # Rows with many valid pixels may contain ground.
SEED_BOX = (0.35, 0.55, 0.65, 0.88)  # Relative x0,y0,x1,y1 around the lower trunk.
SENSITIVITY_CONTRASTS = [0.06, 0.10, 0.14, 0.18]
SKELETON_TOLERANCE = 3      # Pixel distance allowed when comparing to GT.

GT_FILES = {}
TEST_TREES = set()

# ======================== 2. READ THE DATA ========================
def find_pairs(folder):
    """Find exact RGB/D pairs; report missing or ambiguous captures."""
    pattern = re.compile(r'(R\d+[A-Za-z]+\d+)_(\w+)_(RGB|D)-(\d+-\d+-\d+)$', re.I)
    groups = {}
    for path in sorted(Path(folder).rglob('*')):
        if path.suffix.lower() not in {'.png', '.tif', '.tiff'}:
            continue
        # Keep annotations separate from inputs, even if filenames are similar.
        if any(re.search(r'(^|[^a-z])(gt|ground.?truth|annotations?)([^a-z]|$)',
                         part, re.I) for part in path.relative_to(folder).parts):
            continue
        match = pattern.fullmatch(path.stem)
        if match:
            tree, season, kind, stamp = match.groups()
            key = (tree.upper(), season.capitalize(), stamp)
            groups.setdefault(key, {}).setdefault(kind.upper(), []).append(path)
    pairs = []
    for (tree, season, stamp), files in groups.items():
        if len(files.get('RGB', [])) == 1 and len(files.get('D', [])) == 1:
            pairs.append(dict(name=f'{tree}_{season}_{stamp}', tree=tree,
                              season=season, rgb=files['RGB'][0], depth=files['D'][0]))
        else:
            print('Skipped unmatched/ambiguous frame:', tree, season, stamp)
    return pairs


def load_images(pair):
    """Load RGB in [0,1] and depth; missing depth becomes NaN."""
    rgb = np.asarray(Image.open(pair['rgb']).convert('RGB')) / 255.0
    raw = np.asarray(Image.open(pair['depth']))
    if raw.ndim == 3:
        if not (np.array_equal(raw[:, :, 0], raw[:, :, 1]) and
                np.array_equal(raw[:, :, 0], raw[:, :, 2])):
            raise ValueError('Expected a greyscale depth image, not a colourised display.')
        raw = raw[:, :, 0]
    if raw.shape != rgb.shape[:2]:
        raise ValueError('RGB and depth must be aligned and have the same size.')
    valid = np.isfinite(raw) & ~np.isin(raw, INVALID_DEPTH_VALUES)
    depth = raw.astype(float)
    depth[~valid] = np.nan
    return rgb, depth


# ======================== 3. FIND THE TREE ========================
def segment_tree(rgb):
    """Original colour-only baseline; may retain background grass and wires."""
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    brightness = rgb.mean(axis=2)
    mask = (2 * green - red - blue < GREEN_THRESHOLD)
    mask &= (brightness > .03) & (brightness < .72)
    return apply_roi(mask)


def apply_roi(mask):
    """Apply an optional rectangle without resizing or shifting the image."""
    if ROI is None:
        return mask
    left, top, right, bottom = ROI
    if not (0 <= left < right <= mask.shape[1] and 0 <= top < bottom <= mask.shape[0]):
        raise ValueError('ROI is outside the image.')
    cropped = np.zeros_like(mask)
    cropped[top:bottom, left:right] = mask[top:bottom, left:right]
    return cropped


def fit_depth_background(depth):
    valid = np.isfinite(depth)
    dense = valid.mean(axis=1) > DENSE_ROW_FRACTION
    rows = np.flatnonzero(dense)
    background = np.zeros(depth.shape[0])
    info = dict(fit_status='insufficient_dense_rows', fit_rows=len(rows),
                intercept=None, slope=None, background_fit_rmse=None,
                fitted_row_fraction=float(dense.mean()),
                sparse_valid_fraction=float(np.count_nonzero(valid & ~dense[:, None]) /
                                            max(np.count_nonzero(valid), 1)))
    if len(rows) < 3:
        # Do not silently claim that regression ran on insufficient data.
        return background, np.zeros_like(dense), info
    y = np.linspace(-1., 1., depth.shape[0])
    samples = np.array([np.median(depth[row, valid[row]]) for row in rows])
    A = np.column_stack((np.ones(len(rows)), y[rows]))
    coefficients, _, rank, _ = np.linalg.lstsq(A, samples, rcond=None)
    if rank != 2:
        raise ValueError('Background regression is rank deficient.')
    background = coefficients[0] + coefficients[1] * y
    residual = samples - background[rows]
    info.update(fit_status='fitted', intercept=float(coefficients[0]),
                slope=float(coefficients[1]),
                background_fit_rmse=float(np.sqrt(np.mean(residual**2))))
    return background, dense, info


def depth_tree(depth, contrast=0.10, model=None):
    """Use depth contrast and trunk connectivity to remove much of the background.
    1. Ignore missing depth.
    2. Use least-squares linear regression to estimate background intensity
       from the median depths of densely observed rows.
    3. Retain pixels sufficiently different from that background. Sparse rows
       retain valid observations without attempting a ground estimate.
    4. Close one-pixel gaps and retain the component most present in SEED_BOX.
    """
    valid = np.isfinite(depth)
    if not valid.any():
        return np.zeros(depth.shape, dtype=bool)
    if DEPTH_DIRECTION not in [-1, 1] or contrast < 0:
        raise ValueError('Use DEPTH_DIRECTION +1/-1 and nonnegative contrast.')
    low, high = np.percentile(depth[valid], [5, 95])
    margin = contrast * (high - low)
    background, dense_rows, _ = fit_depth_background(depth) if model is None else model
    difference = DEPTH_DIRECTION * (depth - background[:, None])
    mask = valid & (~dense_rows[:, None] | (difference > margin))
    # A small closing joins one-pixel gaps; it does not recover hidden branches.
    mask = apply_roi(ndimage.binary_closing(mask, iterations=1))
    labels, count = ndimage.label(mask, structure=np.ones((3, 3)))
    if count == 0:
        return np.zeros(depth.shape, dtype=bool)
    height, width = depth.shape
    x0, y0, x1, y1 = SEED_BOX
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError('SEED_BOX must be normalised coordinates in [0,1].')
    seed = labels[int(y0*height):int(y1*height), int(x0*width):int(x1*width)]
    overlaps = np.bincount(seed.ravel(), minlength=count+1)
    overlaps[0] = 0
    if overlaps.max() == 0:
        return np.zeros(depth.shape, dtype=bool)  # Flag failure instead of guessing another tree.
    return labels == overlaps.argmax()


def skeletonise(mask):
    """Thin a binary shape to a centreline using the Zhang-Suen method.

    Each pass removes eligible boundary pixels while retaining connectivity.
    Stop when neither of the two substeps removes another pixel.
    """
    image = np.pad(mask.astype(bool), 1)
    while True:
        removed = 0
        for step in [0, 1]:
            neighbours = [image[:-2, 1:-1], image[:-2, 2:], image[1:-1, 2:],
                          image[2:, 2:], image[2:, 1:-1], image[2:, :-2],
                          image[1:-1, :-2], image[:-2, :-2]]
            number = sum(p.astype(np.uint8) for p in neighbours)
            changes = sum((~neighbours[i] & neighbours[(i + 1) % 8]).astype(np.uint8)
                          for i in range(8))
            north, east, south, west = [neighbours[i] for i in [0, 2, 4, 6]]
            if step == 0:
                protected = (north & east & south) | (east & south & west)
            else:
                protected = (north & east & west) | (north & south & west)
            delete = image[1:-1, 1:-1] & (number >= 2) & (number <= 6)
            delete &= (changes == 1) & ~protected
            removed += int(delete.sum())
            image[1:-1, 1:-1][delete] = False
        if removed == 0:
            return image[1:-1, 1:-1]


# ======================== 4. EVALUATE AND PLOT ========================
def read_ground_truth(name, shape):
    """Read a declared aligned binary mask; never mistake depth for GT."""
    if name not in GT_FILES:
        return None
    gt = np.asarray(Image.open(GT_FILES[name]))
    if gt.ndim == 3:
        if not (np.array_equal(gt[:, :, 0], gt[:, :, 1]) and
                np.array_equal(gt[:, :, 0], gt[:, :, 2])):
            raise ValueError('GT must be binary, not a coloured annotation overlay.')
        gt = gt[:, :, 0]
    if gt.shape != shape or not set(np.unique(gt)).issubset({0, 1, 255}):
        raise ValueError('GT must match RGB dimensions, with values 0/1 or 0/255.')
    if not np.any(gt):
        raise ValueError('Empty GT mask cannot validate a tree skeleton.')
    return gt > 0


def compare_with_gt(mask, skeleton, gt):
    """Return mask IoU and centreline precision/recall/F1 within a pixel tolerance.

    Reference centreline is derived from the GT region mask. These scores assess
    2D agreement with the annotated region, not biological branch-axis accuracy.
    """
    if gt is None:
        return dict(iou=None, precision=None, recall=None, f1=None)
    union = np.count_nonzero(mask | gt)
    iou = np.count_nonzero(mask & gt) / union
    predicted = np.argwhere(skeleton)
    reference = np.argwhere(skeletonise(gt))
    if len(predicted) == 0:
        return dict(iou=iou, precision=0., recall=0., f1=0.)
    distance_p = cKDTree(reference).query(predicted)[0]
    distance_g = cKDTree(predicted).query(reference)[0]
    precision = np.mean(distance_p <= SKELETON_TOLERANCE)
    recall = np.mean(distance_g <= SKELETON_TOLERANCE)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    return dict(iou=float(iou), precision=float(precision), recall=float(recall), f1=float(f1))


def save_figures(rgb, mask, skeleton, output, filename):
    """Save uniquely named plots into shared folders for this entire run."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(rgb); axes[0].set_title('RGB image')
    axes[1].imshow(mask, cmap='gray'); axes[1].set_title('Estimated foreground mask')
    axes[2].imshow(rgb)
    v, u = np.nonzero(skeleton)
    axes[2].scatter(u, v, s=1, color='red'); axes[2].set_title('Estimated skeleton')
    for ax in axes:
        ax.axis('off')
    fig.suptitle(filename)
    fig.tight_layout(); fig.savefig(output / 'comparisons' / (filename + '.png'), dpi=140); plt.close(fig)


def write_table(path, rows):
    """Save a list of dictionaries to CSV, with missing scores left blank."""
    if rows:
        with open(path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)


def check_calculations():
    """Verify known regression coefficients and synthetic foreground recovery.

    Synthetic checks verify implementation under the model assumptions;
    they do not validate performance on real trees.
    """
    y = np.linspace(-1., 1., 80)
    depth = np.repeat((30 + 10*y)[:, None], 100, axis=1)
    truth = np.zeros(depth.shape, bool); truth[5:75, 46:54] = True
    depth[truth] += 60
    model = fit_depth_background(depth)
    np.testing.assert_allclose([model[2]['intercept'], model[2]['slope']], [30, 10], atol=1e-10)
    recovered = depth_tree(depth, model=model)
    assert np.count_nonzero(recovered & truth) / np.count_nonzero(recovered | truth) > .99
    missing = np.full((20, 20), np.nan)
    assert fit_depth_background(missing)[2]['fit_status'] == 'insufficient_dense_rows'
    assert not depth_tree(missing).any()
    mask = np.zeros((20, 20), bool); mask[2:18, 8:12] = True
    centre = skeletonise(mask)
    assert ndimage.label(centre, structure=np.ones((3, 3)))[1] == 1
    assert centre.sum() < mask.sum()
    np.testing.assert_array_equal(skeletonise(centre), centre)
    scores = compare_with_gt(mask, centre, mask)
    assert scores['iou'] == 1 and scores['f1'] == 1
    return ('Passed: known linear coefficients; synthetic trunk recovery; missing depth; '
            'connected/idempotent thinning; perfect-match metrics.\n'
            'Synthetic verification is not real-tree validation.\n')


# ======================== 5. RUN THE PROJECT ========================
def choose_folder(title):
    """Open a folder picker; return an empty string on cancellation."""
    import tkinter as tk
    from tkinter import filedialog
    window = tk.Tk()
    window.withdraw()
    window.attributes('-topmost', True)
    try:
        folder = filedialog.askdirectory(
            parent=window,
            title=title,
            mustexist=True)
    finally:
        window.destroy()
    return folder


def main():
    """Run all paired images, compare two methods and save report-ready evidence."""
    folder = choose_folder('Choose the folder containing your RGB and depth photos')
    if not folder:
        print('No folder selected.'); return
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise ValueError('Choose an existing folder containing the images.')
    # The second folder may repeat the normal images as well as contain GT.
    # Use Dataset for observations, so duplicate copies are not paired twice.
    if (folder / "Dataset").is_dir():
        gt_folder = folder / "Dataset (With Summer GT)"
        folder = folder / "Dataset"
        print("Using normal observations from:", folder)
        if gt_folder.is_dir():
            print("Summer GT folder found:", gt_folder)
            print("Set GT_FILES to the actual binary annotations; depth is not GT.")
    pairs = find_pairs(folder)
    print(f'Total matching pairs found: {len(pairs)}')
    pairs.sort(key=lambda pair: (pair['name'] != FIRST_PREVIEW, pair['name']))
    if MAX_PAIRS is not None:
        pairs = pairs[:MAX_PAIRS]
    if not pairs:
        raise ValueError('No exact RGB/depth pairs found. Check folder and filename format.')
    results_folder = choose_folder('Choose where to save the results')
    if not results_folder:
        print('No results folder selected. Processing cancelled.'); return
    out = Path(results_folder).resolve() / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    print('Saving results to:', out)
    for folder_name in ['comparisons']:
        (out / folder_name).mkdir()
    if SAVE_MASK_IMAGES:
        (out / 'masks').mkdir()
    (out / 'checks.txt').write_text(check_calculations())
    # Record the actual settings used so a run can be reproduced.
    settings = dict(folder=str(folder), main_unit_method='5.1 Linear Regression',
                    background_model='median depth intensity = b0 + b1 * normalised row',
                    invalid_depth_values=INVALID_DEPTH_VALUES, roi=ROI, green_threshold=GREEN_THRESHOLD,
                    depth_contrast=DEPTH_CONTRAST, sensitivity_contrasts=SENSITIVITY_CONTRASTS,
                    depth_direction=DEPTH_DIRECTION, dense_row_fraction=DENSE_ROW_FRACTION,
                    seed_box=SEED_BOX, max_pairs=MAX_PAIRS, first_preview=FIRST_PREVIEW,
                    max_figure_pairs=MAX_FIGURE_PAIRS, sensitivity_limit=SENSITIVITY_LIMIT,
                    gt_files=GT_FILES, test_trees=sorted(TEST_TREES), tolerance_px=SKELETON_TOLERANCE)
    import json
    (out / 'settings.json').write_text(json.dumps(settings, indent=2))
    results, sensitivity, errors, model_fits = [], [], [], []
    print(f'Found {len(pairs)} paired observations.')
    print('Running 2D segmentation and skeleton extraction.')
    sensitivity_count = 0
    for pair_number, pair in enumerate(pairs):
        print(f"Processing {pair_number + 1}/{len(pairs)}:", pair["name"], flush=True)
        try:
            rgb, depth = load_images(pair)
            gt = read_ground_truth(pair['name'], depth.shape)
            split = 'test' if pair['tree'] in TEST_TREES else 'development'
            model = None
            for method in ['baseline', 'regression_depth']:
                start = time.perf_counter()
                if method == 'baseline':
                    mask = segment_tree(rgb)
                else:
                    model = fit_depth_background(depth)
                    model_fits.append(dict(observation=pair['name'], **model[2]))
                    if model[2]['fit_status'] != 'fitted':
                        print('  Insufficient background samples: using valid-depth fallback.')
                    mask = depth_tree(depth, DEPTH_CONTRAST, model=model)
                skeleton = skeletonise(mask)
                valid_skeleton_pixels = int(np.count_nonzero(skeleton & np.isfinite(depth)))
                seconds = time.perf_counter() - start  # Exclude GT scoring and plotting.
                scores = compare_with_gt(mask, skeleton, gt)
                results.append(dict(tree=pair['tree'], observation=pair['name'], season=pair['season'],
                                    split=split, method=method, time_s=seconds,
                                    skeleton_pixels=int(skeleton.sum()), skeleton_pixels_with_depth=valid_skeleton_pixels,
                                    depth_coverage=valid_skeleton_pixels/int(skeleton.sum()) if skeleton.any() else 0., **scores))
                filename = pair['name'] + '_' + method
                if SAVE_MASK_IMAGES:
                    Image.fromarray((mask * 255).astype('uint8')).save(out / 'masks' / (filename + '_mask.png'))
                    Image.fromarray((skeleton * 255).astype('uint8')).save(out / 'masks' / (filename + '_skeleton.png'))
                if MAX_FIGURE_PAIRS is None or pair_number < MAX_FIGURE_PAIRS:
                    save_figures(rgb, mask, skeleton, out, filename)
            # Save progress after each pair in case you interrupt a long run.
            write_table(out / 'results.csv', results)
            write_table(out / 'regression_models.csv', model_fits)
            # Never tune depth contrast on test trees. Keep the chosen test settings fixed.
            if split == 'development' and sensitivity_count < SENSITIVITY_LIMIT:
                sensitivity_count += 1
                for contrast in SENSITIVITY_CONTRASTS:
                    start = time.perf_counter()
                    mask = depth_tree(depth, contrast, model=model)
                    sk = skeletonise(mask)
                    elapsed = time.perf_counter() - start
                    sensitivity.append(dict(tree=pair['tree'], observation=pair['name'], depth_contrast=contrast,
                                            foreground_pixels=int(mask.sum()),
                                            skeleton_pixels=int(sk.sum()), time_s=elapsed,
                                            **compare_with_gt(mask, sk, gt)))
        except (ValueError, OSError) as error:
            errors.append(dict(observation=pair['name'], error=str(error)))
            print('  Could not process:', error)
    write_table(out / 'results.csv', results)
    write_table(out / 'sensitivity.csv', sensitivity)
    write_table(out / 'errors.csv', errors)
    # Average frames within each tree, then average trees: busy trees get no extra weight.
    summary = []
    for split in ['development', 'test']:
        for method in ['baseline', 'regression_depth']:
            selected = [r for r in results if r['split'] == split and r['method'] == method]
            for metric in ['iou', 'f1', 'time_s', 'depth_coverage']:
                values = []
                for tree in sorted({r['tree'] for r in selected}):
                    frame_values = [r[metric] for r in selected if r['tree'] == tree and r[metric] is not None]
                    if frame_values:
                        values.append(np.mean(frame_values))
                if values:
                    summary.append(dict(split=split, method=method, metric=metric,
                                        trees=len(values), mean=float(np.mean(values))))
    write_table(out / 'summary.csv', summary)
    note = ('Results describe two-dimensional foreground masks and skeletons. Inspect segmentation failures.\n'
            'Depth segmentation assumes a lower-centre trunk and the configured depth encoding.\n'
            'Main unit method: 5.1 Linear Regression, fitted to dense-row median depth intensities.\n'
            'Regression assumes a linear row trend and background-dominated row medians.\n'
            'Inspect regression_models.csv: fit RMSE measures background fit, NOT tree accuracy.\n'
            'Sparse rows keep valid depth; insufficient dense rows trigger a recorded fallback.\n'
            'Real-tree accuracy, including summer wood extraction, is not validated without GT.\n'
            'Blank GT scores mean no annotation was provided, not zero error.\n'
            'Sensitivity uses a limited development subset, not a representative population sample.\n'
            'Without GT, sensitivity measures output changes, not which threshold is most accurate.\n'
            'Sensitivity time_s excludes the cached regression fit; main results include fitting.\n'
            'Do not choose settings using test scores.\n'
            'Timings are single-run compute measurements, not a rigorous performance benchmark.\n'
            f'Failed observations: {len(errors)}. Do not silently omit these from your report.\n')
    if not TEST_TREES:
        note += 'No held-out trees were declared. Results are exploratory, not generalisation evidence.\n'
    (out / 'READ_RESULTS.txt').write_text(note)
    print(f'Finished. Results saved in: {out.resolve()}')
    print('All comparison images are together in:', (out / 'comparisons').resolve())
    print('See results.csv and summary.csv for the numerical results.')


if __name__ == '__main__':
    main()
