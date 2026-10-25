# Apple tree skeleton extraction

MMA3001 Dataset 1 project — Nelson Bawden.

This single Python program extracts 2D foreground masks and skeletons from paired RGB and grayscale depth observations. Its main numerical method is least-squares linear regression of background row medians, compared with a colour-only baseline.

## Install and run

Use a desktop Python environment with Tkinter available. From the repository folder:

```shell
python -m pip install -r requirements-test.txt
python Tree_Project_2D.py
```

Select the photo folder, then select where results should be saved. The program searches subfolders. Selecting the TreeID parent uses its Dataset child for observations.

Expected example filenames:

```text
R10N07_Winter_RGB-12-10-12.png
R10N07_Winter_D-12-10-12.png
```

Images must already be aligned and have equal dimensions. Depth is treated as raw grayscale intensity; zero is configured as missing. The dataset is not included. Obtain it through the authorised unit source.

## Settings

Edit settings near the top of the source. MAX_PAIRS=None processes all exact pairs. MAX_FIGURE_PAIRS=None saves all comparisons. SENSITIVITY_LIMIT=20 sweeps four thresholds on the first 20 eligible development observations. SAVE_MASK_IMAGES enables separate binary mask and skeleton images.

GT_FILES maps observation names to aligned binary visible-wood masks. TEST_TREES identifies physical trees reserved for evaluation and excluded from sensitivity sweeps. Both are empty by default, so default results are exploratory and have no real-image accuracy scores.

## Outputs

Each run creates a timestamped folder containing comparison PNGs, settings.json, checks.txt, results.csv, regression_models.csv and, when records exist, sensitivity.csv, summary.csv and errors.csv. READ_RESULTS.txt explains limitations. All comparison images share one comparisons folder.

## Documentation

The source reference is [docs/index.html](docs/index.html). Download/open it in a browser to view the formatted page. It documents every function, settings, inputs, outputs and exceptions. Update documentation if the source changes.

## Tests

Keep test_tree_project.py beside Tree_Project_2D.py. From the repository folder:

```shell
python -m pytest test_tree_project.py -v --html=test_report.html --self-contained-html --junitxml=test_results.xml
```

Open test_report.html in a browser. The supplied [recorded report](test-results/test_report.html) contains 38 passing tests from an AI-assisted Linux Python 3.12.14 run. It tests analytical regression, synthetic foreground extraction, pairing, invalid inputs, thinning, metrics, CSV export and a synthetic end-to-end run. Native folder dialogs are replaced in the tests. Exact tested versions and the execution log are in test-results/.

These tests do not establish real-tree accuracy, Windows GUI behaviour or full-dataset performance. Rerun them in your own environment after changes.

## Scope and limitations

The program does not produce calibrated 3D geometry, register images, identify trees across seasons or plan robot paths. Camera calibration and physical depth scale have not been established. Leaves and fruit can obscure branches and contaminate extracted summer skeletons. Background-fit RMSE and skeleton pixel count are not segmentation accuracy. Real-image ground truth and held-out evaluation remain necessary.

## AI assistance

ChatGPT through Codex assisted substantially with code drafting and revision, method explanation, documentation, test development and initial test execution. The student directed scope and remains responsible for reviewing the code, validating results and explaining the methods. The report provides further disclosure.

## Licence

The project source code is licensed under the MIT License; see LICENSE. Supplied TreeID data, photographs, annotations and university teaching materials are excluded and remain subject to their original terms. Third-party dependencies retain their own licences. Do not redistribute restricted dataset files.
