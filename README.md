# 🥭 icepaya

**Grain and bubble morphometry of ice thin sections using Minkowski tensors.**

icepaya segments greyscale thin-section images of ice into grains (or bubbles) and measures the size and shape of each object. Besides standard descriptors (area, perimeter, major axis, eccentricity), it computes the irreducible Minkowski tensors ψ<sub>s</sub> and structure metrics q<sub>s</sub> with [papaya2](https://morphometry.org/software/papaya2/). These quantify the *s*-fold symmetry and orientation of each grain, and the preferred alignment of the whole microstructure.

## What it does

- **Segmentation** with contrast enhancement (CLAHE), Otsu thresholding and small-structure filtering, calibrated interactively on one tile of the image.
- **Tiling** of large images, with the option to skip damaged or unrepresentative tiles.
- **Per-grain shape metrics.** Each grain's boundary gives ψ<sub>s</sub> = Σ<sub>k</sub> L<sub>k</sub> exp(i s φ<sub>k</sub>), where L<sub>k</sub> and φ<sub>k</sub> are the length and outward normal angle of boundary segment *k*. From this:
  - q<sub>s</sub> = |ψ<sub>s</sub>| / perimeter (0 to 1) measures *s*-fold symmetry: q<sub>2</sub> for elongation, q<sub>3</sub> for triangularity, and so on.
  - arg(ψ<sub>s</sub>)/s gives the corresponding orientation.
- **Aggregate metrics.** q<sub>s</sub> computed from ψ<sub>s</sub> summed over all grains and divided by total perimeter. Randomly oriented grains cancel out, so these measure the preferred alignment of the microstructure rather than the shapes of individual grains.
- **An area cut-off** that excludes small objects.
- **Results** saved as a pickle, including the settings used, plus summary figures.

## Requirements

Python 3 with 

```bash
pip install numpy scipy scikit-image matplotlib PyQt5 opencv-python-headless
```

Use the *headless* OpenCV build. The regular `opencv-python` package bundles its own Qt, which may conflict with PyQt5.

`pypaya2`, the Python interface of papaya2, must be built from source. 

## Usage

### GUI

```bash
python GUI.py [image.png]
```
Further instructions are available within the GUI. 

### Scripting

```python
from icepaya import icepaya

ip = icepaya('section.png', ntiles=2)
ip.set_params('grains', clahe=9, filt=150, amin=500)  # mode and segmentation parameters
d = ip.process()                                      # dict of per-grain arrays
ip.save(d)                                            # -> section_grains.pkl
ip.save_summary_fig(d)                                # -> section_grains_stats.png
```
## Output

`icepaya.load_data('section_grains.pkl')` returns a dict with the following fields.

| Field | Description |
|---|---|
| `centroid` | (row, col) in the full image [px] |
| `area`, `perim` | Area [px²] and perimeter [px] from papaya2 |
| `ori` | Elongation orientation from ψ<sub>2</sub> [deg] |
| `qi`, `pi` | Dicts `{s: array}` of q<sub>s</sub> and complex ψ<sub>s</sub>, for s = 2 to 8 |
| `qi_agg`, `pi_agg` | Aggregate q<sub>s</sub> and ψ<sub>s</sub>, indexed by s |
| `majorax`, `minorax`, `ecc`, `equivdiam` | Moment-ellipse descriptors (scikit-image) |
| `area0`, `perim0`, `ori0` | Area, perimeter and orientation from scikit-image |
| `tile`, `label` | Tile index and label within the tile |
| `meta` | Settings, image name, timestamp and library versions |

## Caveats

- **Segmentation quality dominates the results.** Missing boundary segments merge grains, and scratches or subgrain boundaries split them. Check several tiles, not just the calibration tile.
- **Grains touching a tile edge are discarded.** This biases results against large grains, and more tiles make it worse.
- **q<sub>s</sub> has a pixel-grid noise floor.** Grain masks are smoothed (`SIGMA` in `icepaya.py`) to reduce it, but for small grains q<sub>4</sub>–q<sub>8</sub> values below roughly 0.05–0.1 are noise. Higher orders are more sensitive to rough boundaries.
- **The q<sub>s</sub> are not independent.** They include harmonics, e.g. a triangle has q<sub>3</sub> = q<sub>6</sub> = 1, and an ellipse has non-zero q<sub>4</sub> and q<sub>6</sub>. Interpret them together.
- **Orientation needs clear elongation.** It is only meaningful when q<sub>2</sub> is well above the noise level.
- **Thin sections are 2D cuts of 3D grains.** Orientations are measured relative to the image axes. Biases can be quantified by e.g. [3D imaging](https://doi.org/10.1017/jog.2026.10124).

## Test images

```bash
python make-test-shapes.py 1024
```

This makes idealized images of randomly placed circles, ellipses, triangles, and a mix of all three, together with ground-truth files (`*_truth.pkl`) containing exact shape parameters and q<sub>s</sub>. The scene is identical at any resolution, which makes it useful for testing pixel-grid errors.

## Credit

Developed as part of the Villum Foundation investigator project *IceFlow* (former package name `LASM`) and the Novo Nordisk Foundation challenge project *PRECISE* (new package name `icepaya`).

By *Nicholas M. Rathmann, Niels Bohr Institute, UCPH*

## References

- Schaller, F. M., Wagner, J., & Kapfer, S. C. (2020). papaya2: 2D Irreducible Minkowski Tensor computation. *Journal of Open Source Software*, 5(54), 2538.
