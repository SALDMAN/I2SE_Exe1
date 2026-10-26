# Ex1: Plane Projection of a 3D Volume

> Introduction to Software Engineering (OOP / OOD)
> Given a 3D raster volume (NRRD) and a plane, compute the 2D image of the volume sampled on that plane, and render a list of planes as an MP4 video.

---

## 1. Problem and Scope

### 1.1 Problem
Given a 3D matrix `M` of size `sx * sy * sz` (unsigned 8/16/32-bit values, `1 <= sx,sy,sz < 2^10`) and a plane `p`, compute a 2D matrix `E` (an image) that is the projection of `p` onto `M`. The assignment leaves several details open (coordinate conventions, interpolation, output size, handling of out-of-volume points), so this README fixes them explicitly (section 3).

### 1.2 In scope
- Reading 3D NRRD volumes (8/16/32-bit unsigned).
- Defining a plane by 3 points (isosceles triangle) or 4 points (isosceles trapezoid).
- The projection algorithm, designed to be correct and fast (>= 30 FPS required, >= 100 FPS target for a 500^3 byte volume).
- A CLI demo: `python Ex1.py file.nrrd file.csv file.mp4`.
- Tests, a small NRRD data set, and a short experiment write-up.

### 1.3 Out of scope
- Interactive GUI / 3D rendering.
- Arbitrary (non-isosceles) planes, volume rendering, MIP.
- Non-NRRD formats, signed or floating-point voxels.

---

## 2. Software Requirements Specification (SRS)

### 2.1 Functional requirements

| ID | Requirement |
|----|-------------|
| FR1 | The system shall load a 3D volume from a `.nrrd` file with element type `uint8`, `uint16` or `uint32`. |
| FR2 | The system shall load a `.csv` file where each row is a plane: 3 points (9 numbers) or 4 points (12 numbers). Each point is `x,y,z`. |
| FR3 | The system shall validate each plane: 3 points `(a,b,c)` require `abs(dist(a,b) - dist(a,c)) < EPS`. 4 points `(a1,a2,b,c)` require `abs(dist(a1,b) - dist(a2,c)) < EPS`, `dist(a1,a2) <= dist(b,c)`, and coplanarity. `EPS = 0.001`. |
| FR4 | For every valid plane, the system shall compute the 2D projection image `E` using the algorithm in section 4. |
| FR5 | Points of the plane that fall outside `M` shall produce the value `0`. |
| FR6 | The system shall write one video frame per CSV row, in CSV order, to an MP4 file. |
| FR7 | Invalid planes shall be reported (row number and reason) and handled according to a documented policy (default: skip and warn; `--strict` aborts). |

### 2.2 Non-functional requirements

| ID | Requirement |
|----|-------------|
| NFR1 | Performance: projection of a 500^3 byte volume at >= 30 FPS (target >= 100 FPS) excluding video encoding. |
| NFR2 | Robustness: no crash on any plane position, including planes partly or fully outside the volume, or degenerate planes (reported as errors). |
| NFR3 | Memory: the volume is loaded once and never copied per frame. Output buffers are reused. |
| NFR4 | Design: clear separation between I/O, geometry, the projection algorithm and the demo, so each part can be tested and replaced (OOD). |
| NFR5 | Portability: Python 3.10+, runs on Linux / macOS / Windows. |

### 2.3 Inputs and outputs
- **Input**: `file.nrrd`, `file.csv`.
- **Output**: `file.mp4`, with one frame per plane.

### 2.4 CSV format
One plane per row, no header, comma separated:

```
# 3 points: ax,ay,az,bx,by,bz,cx,cy,cz
0,0,0, 100,0,0, 0,100,0
# 4 points: a1x,a1y,a1z,a2x,a2y,a2z,bx,by,bz,cx,cy,cz
```

Lines starting with `#` and blank lines are ignored.

---

## 3. Design Decisions (resolving the "not well-defined" parts)

| Topic | Decision |
|-------|----------|
| Coordinate system | Voxel index space: `M[x][y][z]`, `0 <= x < sx`, etc. Plane points are given in voxel coordinates (floats allowed). |
| Sampling | Nearest neighbour (round to nearest voxel) by default, for speed. Trilinear interpolation is an optional strategy. |
| Out-of-volume | Value `0`. |
| Output size | `E` has `round(Dab)` rows (y) and `round(Dbc)` columns (x), as in the assignment (`Dab * Dbc`). Pixels not covered by the triangle stay `0`. |
| Trapezoid (4 points) | Handled by generalizing the start points: `a1` starts at `a1_start`, `a2` at `a2_start` and step along `v(a1,b)` and `v(a2,c)`. The 3-point case is the special case `a1_start = a2_start = a`. |
| Value range for video | 16/32-bit data is linearly windowed to 8 bits for the video (global min/max, or `--window lo hi`). Raw projection values keep the original dtype. |
| NRRD axes | The NRRD array is reordered so index order is `[x][y][z]`. Spacing and orientation metadata are not applied (voxel space is used). |

### 3.1 Architecture

```
Ex1.py                  # demo main (CLI); loads the NRRD with pynrrd -> NumPy array
projection/
  io_csv.py             # CSV reader   -> list[Plane]
  geometry.py           # Vec3 helpers, Plane (validation, basis vectors, sizes)
  projector.py          # Projector (interface) + PlaneProjector (algorithm)
  sampler.py            # Sampler (interface): NearestSampler, TrilinearSampler
  video.py              # VideoWriter: frames -> mp4, intensity windowing
tests/                  # unit, property and performance tests
data/                   # sample NRRD + CSV files
```

Main classes:

- The volume is a plain NumPy 3D array (`M[x][y][z]`) loaded with the `pynrrd` library, so no custom volume or NRRD reader classes are needed. Shape and dtype come directly from the array.
- `Plane`: built from 3 or 4 points; validates (FR3) and exposes `start_left`, `start_right`, `dir_left`, `dir_right`, `Dab`, `Dbc`.
- `Sampler` (Strategy pattern): `sample(M, points) -> values` (`M` is the NumPy array). Allows swapping nearest and trilinear.
- `Projector`: `project(M, plane) -> ndarray` (the 2D image `E`). Reuses preallocated buffers.
- `VideoWriter`: Context manager that receives frames and writes the MP4.

---

## 4. The Algorithm

Input: volume `M`, plane given by `(a,b,c)` (or `(a1,a2,b,c)`).

1. `Dab = dist(a,b) = dist(a,c)`, `Dbc = dist(b,c)`.
2. `v_ab = (b - a) / Dab`, `v_ac = (c - a) / Dab`.
3. Allocate `E` as a zero matrix of size `Dab * Dbc` (rows = y, columns = x).
4. For each row `y` in `[0, Dab)`:
   1. `a1 = a + (y+1) * v_ab`, `a2 = a + (y+1) * v_ac`.
   2. `d12 = dist(a1, a2)`, `dp = (a2 - a1) / d12`.
   3. For `x1 = 0..floor(d12)`: `p = a1 + x1 * dp`, `v = M[round(p)]` (or `0` if outside), and write `E[y][ (Dbc/2) - (d12/2) + x1 ] = v` (the row is centred in the image).
5. Return `E`.

### 4.1 Efficient implementation (vectorized)
The per-pixel double loop is replaced by array operations over the whole image at once:

1. Compute row vectors `a1[y]`, `a2[y]`, `d12[y]`, `dp[y]` for all `y` (shape `(Dab, 3)`).
2. Build the index grid `x1 = 0..Dbc` and the points `P[y, x1] = a1[y] + x1 * dp[y]` by broadcasting (shape `(Dab, Dbc, 3)`).
3. Build the mask `x1 <= d12[y]` (pixels inside the triangle) and the in-bounds mask `0 <= P < (sx,sy,sz)`.
4. Round the points to integers, gather `M[xi, yi, zi]` with fancy indexing for valid pixels and write them to `E` at the centred column `x = Dbc/2 - d12/2 + x1`.

Complexity per frame: `O(Dab * Dbc)` time, `O(Dab * Dbc)` memory (reusable buffers). Because the geometry of a row depends only on the plane, per-plane precomputation is done once and the memory is reused between frames. If NumPy is not fast enough, the inner gather can be compiled with Numba (`@njit(parallel=True)`) without changing the interfaces.

### 4.2 Edge cases
- The plane partly or fully outside `M`: those pixels are `0`.
- Degenerate planes (`Dab = 0`, collinear points): rejected with a clear error.
- Non-integer distances: sizes are rounded, steps use the exact floating point values.
- Planes violating FR3: reported and skipped (or abort with `--strict`).
- Volumes with a dimension of size 1.

---

## 5. Data Set

| File | Description | Source |
|------|-------------|--------|
| `data/MRHead.nrrd` | MRI of a head (uint8/int16 depending on version), starting file from the assignment | 3D Slicer sample data |
| `data/synthetic_cube_64.nrrd` | Generated volume with known values, used for tests (e.g. value = `x + y + z`) | Generated by `tests/make_data.py` |
| `data/synthetic_500.nrrd` | 500^3 uint8 random/gradient volume for performance tests | Generated |
| `data/planes_axes.csv` | Planes parallel to the X, Y, Z sections | Hand made |
| `data/planes_rotation.csv` | Sequence of planes rotating around an axis (demo) | Generated |

---

## 6. Testing Strategy

| Level | What is tested |
|-------|----------------|
| Unit: geometry | Distances, normalization, plane validation (EPS boundary, 3 vs 4 points, coplanarity, degenerate input). |
| Unit: IO | NRRD loading for uint8/16/32, axis order, CSV parsing (comments, wrong column count, bad numbers). |
| Unit: sampler | Nearest rounding, out-of-bounds returns 0, corner and edge voxels. |
| Correctness | Axis-aligned planes must equal the corresponding NumPy slices (`M[k,:,:]`, `M[:,k,:]`, `M[:,:,k]`) on the synthetic cube. |
| Reference | The vectorized projector is compared against a direct, literal implementation of the pseudo-code (slow, loop based) on random planes. |
| Property | Rotating a plane by 90 degrees about the volume centre gives the rotated image; a plane completely outside gives an all zero image; output shape equals `Dab * Dbc`. |
| Robustness | Planes partly outside, huge planes, volumes of size 1 in some dimension. |
| Performance | 500^3 uint8: average FPS over 1000 random planes must be >= 30 (target >= 100). Run with `pytest -m perf`. |
| Demo / end to end | `python Ex1.py data/MRHead.nrrd data/planes_rotation.csv out.mp4` must finish, produce a playable MP4 with exactly one frame per valid CSV row, in order. A test checks the frame count and frame size. |

Run all tests:

```bash
pip install -r requirements.txt
pytest
pytest -m perf      # performance tests
```

---

## 7. Installation and Usage

Requirements: Python 3.10+, `numpy`, `pynrrd`, `opencv-python` (or `imageio[ffmpeg]`), `pytest`.

```bash
pip install -r requirements.txt
python Ex1.py <file.nrrd> <file.csv> <file.mp4>
```

Example:

```bash
python Ex1.py data/MRHead.nrrd data/planes_rotation.csv results/rotation.mp4
```

Optional flags:

| Flag | Meaning |
|------|---------|
| `--strict` | Abort on the first invalid plane. |
| `--interp {nearest,trilinear}` | Sampling method (default `nearest`). |
| `--window LO HI` | Intensity window for 16/32-bit data. |
| `--fps N` | Video frame rate (default 30). |

---

## 8. Experiment and Results

> Fill in after running the project.

**Setup**: machine / CPU, Python version, volume (`MRHead.nrrd`, size ...), CSV (`planes_rotation.csv`, N planes).

**Results**

| Metric | Value |
|--------|-------|
| Volume size | e.g. 256 x 256 x 130 |
| Frames | N |
| Average projection time per frame | ... ms |
| Projection FPS (500^3 synthetic) | ... |
| Total run time including video encoding | ... s |

**Output video**: `results/rotation.mp4`

**Observations**: do the axis-aligned planes match the 3D Slicer sections? What happens at the borders of the volume (black areas)? Difference between nearest and trilinear sampling in quality and speed.

---

## 9. Repository Layout and Workflow

- `README.md` (this file): problem, scope, SRS, design, algorithm, tests, results.
- `projection/`: source code. `tests/`: tests. `data/`: sample inputs. `results/`: output videos.
- Commit often with small, descriptive messages (e.g. `Add Plane validation`, `Vectorize projector`, `Add rotation demo`).

## 10. Notes
- The assignment text refers to the demo as `Ex0.py`; this project names it `Ex1.py`. Rename if your submission requires the other name.
