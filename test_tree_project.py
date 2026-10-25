"""Verification tests for Tree_Project_2D.py; no real dataset or GUI required.

Run: python -m pytest test_tree_project.py -v
Tests use pytest's temporary directories, never the user's photo/results folders.
AI-assisted test development. Passing verifies these cases, not real-tree accuracy.
"""
from pathlib import Path
import csv
import importlib.util
import json

import numpy as np
from PIL import Image
import pytest

SOURCE = Path(__file__).with_name('Tree_Project_2D.py')
if not SOURCE.exists():
    SOURCE = Path(__file__).with_name('Tree_Project_2D(1).py')
spec = importlib.util.spec_from_file_location('tree_project_under_test', SOURCE)
tree = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tree)


@pytest.fixture(autouse=True)
def default_settings(monkeypatch):
    """Isolate each test from changes to module settings in previous tests."""
    defaults = dict(ROI=None, GREEN_THRESHOLD=.08, DEPTH_DIRECTION=1,
                    DENSE_ROW_FRACTION=.4, SEED_BOX=(.35,.55,.65,.88),
                    INVALID_DEPTH_VALUES=[0], SKELETON_TOLERANCE=3,
                    GT_FILES={}, TEST_TREES=set(), MAX_PAIRS=None,
                    MAX_FIGURE_PAIRS=None, SENSITIVITY_LIMIT=20,
                    SAVE_MASK_IMAGES=False, DEPTH_CONTRAST=.1,
                    SENSITIVITY_CONTRASTS=[.06,.1,.14,.18])
    for key, value in defaults.items():
        monkeypatch.setattr(tree, key, value)


@pytest.fixture
def scene():
    """Manufactured linearly varying background and narrow bright trunk."""
    y = np.linspace(-1, 1, 80)
    depth = np.repeat((30 + 10*y)[:, None], 100, axis=1)
    truth = np.zeros((80,100), bool)
    truth[5:75,46:54] = True
    depth[truth] += 60
    return depth, truth


def save_pair(folder, depth, name='R10N07_Winter', stamp='12-10-12'):
    folder.mkdir(parents=True, exist_ok=True)
    rgb_path = folder / f'{name}_RGB-{stamp}.png'
    depth_path = folder / f'{name}_D-{stamp}.png'
    Image.fromarray(np.full((*depth.shape[:2], 3), 90, np.uint8)).save(rgb_path)
    Image.fromarray(np.asarray(depth, dtype=np.uint8)).save(depth_path)
    return dict(rgb=rgb_path, depth=depth_path)


def test_pairing_recursive_and_annotations_excluded(tmp_path):
    save_pair(tmp_path/'photos', np.ones((8,8))*30)
    save_pair(tmp_path/'Dataset (With Summer GT)', np.ones((8,8))*30)
    pairs = tree.find_pairs(tmp_path)
    assert len(pairs) == 1
    assert pairs[0]['tree'] == 'R10N07'
    assert pairs[0]['name'] == 'R10N07_Winter_12-10-12'


def test_pairing_skips_duplicates_and_missing_counterpart(tmp_path):
    pair = save_pair(tmp_path/'a', np.ones((8,8))*30)
    extra = tmp_path/'b'; extra.mkdir()
    (extra/pair['rgb'].name).write_bytes(pair['rgb'].read_bytes())
    (tmp_path/'R21U04_Summer_RGB-14-25-32.png').touch()
    assert tree.find_pairs(tmp_path) == []


def test_load_normalises_rgb_and_marks_missing_depth(tmp_path):
    pair = save_pair(tmp_path, np.array([[0,20],[40,80]],np.uint8))
    rgb, depth = tree.load_images(pair)
    np.testing.assert_allclose(rgb, 90/255)
    assert np.isnan(depth[0,0])
    np.testing.assert_array_equal(depth[1], [40,80])


def test_load_accepts_equal_channel_depth(tmp_path):
    pair = save_pair(tmp_path, np.full((4,4),30,np.uint8))
    Image.fromarray(np.full((4,4,3),30,np.uint8)).save(pair['depth'])
    _, depth = tree.load_images(pair)
    np.testing.assert_array_equal(depth, np.full((4,4),30))


@pytest.mark.parametrize('problem', ['colour', 'shape'])
def test_load_rejects_unsupported_depth(tmp_path, problem):
    pair = save_pair(tmp_path, np.full((4,4),30,np.uint8))
    if problem == 'colour':
        bad = np.zeros((4,4,3),np.uint8); bad[:,:,0] = 30
    else:
        bad = np.full((3,4),30,np.uint8)
    Image.fromarray(bad).save(pair['depth'])
    with pytest.raises(ValueError):
        tree.load_images(pair)


@pytest.mark.parametrize('intercept,slope', [(30,10),(100,-20),(50,0)])
def test_regression_recovers_analytical_coefficients(intercept, slope):
    y = np.linspace(-1,1,30)
    depth = np.repeat((intercept+slope*y)[:,None], 40, axis=1)
    background, dense, info = tree.fit_depth_background(depth)
    np.testing.assert_allclose(background, intercept+slope*y, atol=1e-10)
    np.testing.assert_allclose([info['intercept'],info['slope']], [intercept,slope], atol=1e-10)
    assert dense.all() and info['fit_status'] == 'fitted'
    assert info['background_fit_rmse'] < 1e-10


def test_regression_ignores_narrow_trunk_in_row_medians(scene):
    depth, _ = scene
    _, _, info = tree.fit_depth_background(depth)
    np.testing.assert_allclose([info['intercept'],info['slope']], [30,10], atol=1e-10)


def test_regression_records_insufficient_background():
    depth = np.full((20,20),np.nan); depth[:2] = 30
    _, dense, info = tree.fit_depth_background(depth)
    assert info['fit_status'] == 'insufficient_dense_rows'
    assert info['fit_rows'] == 2 and info['intercept'] is None
    assert not dense.any()


def test_depth_extraction_recovers_manufactured_trunk(scene):
    depth, truth = scene
    result = tree.depth_tree(depth)
    assert np.count_nonzero(result & truth)/np.count_nonzero(result | truth) > .99


def test_cached_model_matches_uncached_extraction(scene):
    depth, _ = scene
    np.testing.assert_array_equal(tree.depth_tree(depth),
        tree.depth_tree(depth, model=tree.fit_depth_background(depth)))


def test_reversed_depth_encoding_with_matching_direction(scene, monkeypatch):
    depth, truth = scene
    monkeypatch.setattr(tree,'DEPTH_DIRECTION',-1)
    result = tree.depth_tree(200-depth)
    assert np.count_nonzero(result & truth)/np.count_nonzero(result | truth) > .99


def test_all_missing_depth_returns_empty():
    assert not tree.depth_tree(np.full((20,20),np.nan)).any()


@pytest.mark.parametrize('setting,value', [('DEPTH_DIRECTION',0),('SEED_BOX',(1,0,0,1))])
def test_invalid_segmentation_settings_raise(scene, monkeypatch, setting, value):
    monkeypatch.setattr(tree,setting,value)
    with pytest.raises(ValueError):
        tree.depth_tree(scene[0])


def test_negative_depth_contrast_rejected(scene):
    with pytest.raises(ValueError):
        tree.depth_tree(scene[0], contrast=-.1)


def test_seed_without_overlap_returns_empty(scene, monkeypatch):
    monkeypatch.setattr(tree,'SEED_BOX',(0,.1,.1,.3))
    assert not tree.depth_tree(scene[0]).any()


def test_baseline_classifies_known_colours():
    rgb = np.array([[[.3,.3,.3],[.1,.8,.1],[0,0,0],[1,1,1]]])
    np.testing.assert_array_equal(tree.segment_tree(rgb), [[True,False,False,False]])


def test_roi_preserves_coordinates(monkeypatch):
    monkeypatch.setattr(tree,'ROI',(2,3,6,8))
    original = np.ones((10,10),bool)
    expected = np.zeros_like(original); expected[3:8,2:6] = True
    np.testing.assert_array_equal(tree.apply_roi(original),expected)
    assert original.all()


def test_invalid_roi_raises(monkeypatch):
    monkeypatch.setattr(tree,'ROI',(-1,0,5,5))
    with pytest.raises(ValueError):
        tree.apply_roi(np.ones((10,10),bool))


def test_skeleton_preserves_connected_structure_and_is_idempotent():
    mask = np.zeros((40,40),bool);mask[5:35,17:23]=True;mask[10:16,7:33]=True
    original = mask.copy()
    sk = tree.skeletonise(mask)
    assert 0 < sk.sum() < mask.sum()
    assert tree.ndimage.label(sk, structure=np.ones((3,3)))[1] == 1
    assert not (sk & ~mask).any()
    np.testing.assert_array_equal(tree.skeletonise(sk),sk)
    np.testing.assert_array_equal(mask,original)


def test_empty_skeleton_input_returns_empty():
    assert not tree.skeletonise(np.zeros((10,10),bool)).any()


def test_gt_not_configured_returns_none():
    assert tree.read_ground_truth('unknown',(10,10)) is None


def test_valid_gt_loads_as_boolean(tmp_path, monkeypatch):
    mask = np.zeros((10,10),np.uint8);mask[2:8,4:6]=255
    path = tmp_path/'gt.png';Image.fromarray(mask).save(path)
    monkeypatch.setattr(tree,'GT_FILES',{'example':str(path)})
    np.testing.assert_array_equal(tree.read_ground_truth('example',(10,10)),mask>0)


@pytest.mark.parametrize('problem',['empty','nonbinary','shape','colour'])
def test_invalid_gt_rejected(tmp_path, monkeypatch, problem):
    mask = np.zeros((10,10),np.uint8);mask[2:8,4:6]=255
    if problem=='empty':mask[:]=0
    if problem=='nonbinary':mask[2,4]=50
    if problem=='colour':
        mask = np.repeat(mask[:,:,None],3,axis=2);mask[0,0,0]=255
    path=tmp_path/'gt.png';Image.fromarray(mask).save(path)
    monkeypatch.setattr(tree,'GT_FILES',{'example':str(path)})
    with pytest.raises(ValueError):
        tree.read_ground_truth('example',(8,8) if problem=='shape' else (10,10))


def test_metrics_perfect_match_and_missing_reference():
    mask=np.zeros((20,20),bool);mask[2:18,8:12]=True
    sk=tree.skeletonise(mask)
    assert tree.compare_with_gt(mask,sk,mask)==dict(iou=1.,precision=1.,recall=1.,f1=1.)
    assert all(x is None for x in tree.compare_with_gt(mask,sk,None).values())


def test_metrics_empty_prediction():
    gt=np.zeros((20,20),bool);gt[2:18,8:12]=True
    empty=np.zeros_like(gt)
    assert tree.compare_with_gt(empty,empty,gt)==dict(iou=0.,precision=0.,recall=0.,f1=0.)


def test_metric_tolerance_boundary():
    gt=np.zeros((30,30),bool);gt[5:25,10]=True
    near=np.zeros_like(gt);near[5:25,13]=True
    far=np.zeros_like(gt);far[5:25,14]=True
    assert tree.compare_with_gt(near,near,gt)['f1']==1
    assert tree.compare_with_gt(far,far,gt)['f1']==0


def test_csv_preserves_missing_scores(tmp_path):
    path=tmp_path/'results.csv'
    tree.write_table(path,[dict(name='tree',iou=None,count=10)])
    with path.open(newline='') as file:rows=list(csv.DictReader(file))
    assert rows == [dict(name='tree',iou='',count='10')]
    absent=tmp_path/'empty.csv';tree.write_table(absent,[])
    assert not absent.exists()


def test_builtin_verification_passes():
    assert tree.check_calculations().startswith('Passed:')


def test_main_cancelled_input_does_not_process(monkeypatch):
    monkeypatch.setattr(tree,'choose_folder',lambda title:'')
    monkeypatch.setattr(tree,'find_pairs',lambda folder:pytest.fail('Processing should not start'))
    assert tree.main() is None


def test_main_cancelled_results_creates_no_output(tmp_path, monkeypatch, scene):
    photos=tmp_path/'photos';save_pair(photos,scene[0])
    choices=iter([str(photos),''])
    monkeypatch.setattr(tree,'choose_folder',lambda title:next(choices))
    before=set(tmp_path.rglob('*'))
    assert tree.main() is None
    assert set(tmp_path.rglob('*'))==before


def test_main_end_to_end_outputs_and_test_split(tmp_path, monkeypatch, scene):
    """Exercise real file loading, computation and plotting; only GUI is replaced."""
    photos=tmp_path/'photos';save_pair(photos,scene[0])
    save_pair(photos,scene[0],name='R21U04_Summer',stamp='14-25-32')
    out=tmp_path/'results';out.mkdir()
    choices=iter([str(photos),str(out)])
    monkeypatch.setattr(tree,'choose_folder',lambda title:next(choices))
    monkeypatch.setattr(tree,'TEST_TREES',{'R21U04'})
    tree.main()
    runs=list(out.iterdir());assert len(runs)==1
    run=runs[0]
    def records(name):
        with (run/name).open(newline='') as file:return list(csv.DictReader(file))
    results=records('results.csv');assert len(results)==4
    assert {r['method'] for r in results}=={'baseline','regression_depth'}
    assert all(r['split']=='test' for r in results if r['tree']=='R21U04')
    assert all(r['iou']=='' for r in results)
    sensitivity=records('sensitivity.csv')
    assert len(sensitivity)==4 and {r['tree'] for r in sensitivity}=={'R10N07'}
    assert len(records('regression_models.csv'))==2
    assert records('summary.csv')
    figures=list((run/'comparisons').glob('*.png'));assert len(figures)==4
    for figure in figures:
        with Image.open(figure) as image:assert image.width>0 and image.height>0
    assert not (run/'masks').exists()
    assert not (run/'errors.csv').exists()
    assert (run/'checks.txt').read_text().startswith('Passed:')
    assert (run/'READ_RESULTS.txt').is_file()
    assert json.loads((run/'settings.json').read_text())['test_trees']==['R21U04']
