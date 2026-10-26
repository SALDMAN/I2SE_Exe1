# Software Requirements Specification (SRS)

**Project:** Ex1 - Plane Projection of a 3D Volume
**Course:** Introduction to Software Engineering (OOP / OOD)
**Version:** 1.0
**Status:** Draft

---

## 1. Introduction

### 1.1 Purpose
This document specifies the requirements of a program that computes 2D projections (cross-section images) of a 3D raster volume on arbitrary planes and renders a sequence of such projections as a video. It is intended for the developer, the course staff (reviewers and graders) and the test author.

### 1.2 Scope
The software, referred to as **PlaneProjector**, reads a 3D volume stored in the NRRD format, reads a list of planes from a CSV file, computes for each plane the 2D image `E` defined by the projection algorithm (section 5), and writes all images, in order, as frames of an MP4 video.

In scope: NRRD input, CSV input, plane validation, the projection algorithm, MP4 output, tests and a command line demo.
Out of scope: GUI, 3D rendering, volume rendering or MIP, non-isosceles planes, signed or floating point voxel types, and formats other than NRRD.

### 1.3 Definitions, acronyms, abbreviations

| Term | Meaning |
|------|---------|
| Volume (`M`) | 3D matrix of size `sx * sy * sz`, entry `M[x][y][z]` is an unsigned integer of 8, 16 or 32 bits. |
| Voxel | One element of the volume. |
| Plane (`p`) | A planar region defined by 3 points (isosceles triangle) or 4 points (isosceles trapezoid). |
| Projection image (`E`) | The 2D matrix obtained by sampling `M` over the plane. |
| NRRD | Nearly Raw Raster Data, a file format for N-dimensional raster data. |
| CSV | Comma separated values text file. |
| FPS | Frames (projection images) computed per second. |
| EPS | Numeric tolerance, `EPS = 0.001`. |
| `dist(u,v)` | Euclidean distance between points `u` and `v`. |

### 1.4 References
- Assignment text: "Ex1: Algorithm: Implementing a plane projection of 3D volume".
- NRRD file format specification and the 3D Slicer project (sample data).
- IEEE 830 / ISO/IEC/IEEE 29148 (structure of an SRS).

### 1.5 Overview
Section 2 gives the general description, section 3 the external interfaces, section 4 the functional requirements, section 5 the algorithm specification, section 6 the non-functional requirements, section 7 the data requirements, section 8 the acceptance and verification, and section 9 the traceability matrix.

---

## 2. Overall Description

### 2.1 Product perspective
PlaneProjector is a standalone command line program written in Python. It relies on third party libraries for NRRD reading (`pynrrd`), numerical work (`NumPy`) and video writing (`OpenCV` or `imageio` with ffmpeg).

```
 file.nrrd ---+
              +--> PlaneProjector --> file.mp4
 file.csv  ---+
```

### 2.2 Product functions
1. Load a 3D volume.
2. Load and validate a list of planes.
3. Compute the projection image of each plane.
4. Convert images to video frames and write the video.
5. Report errors and warnings.

### 2.3 User characteristics
Users are students and course staff who are comfortable with a command line. No knowledge of the internals is required.

### 2.4 Constraints
- Programming language: Python (3.10 or newer).
- Volume dimensions: `1 <= sx, sy, sz < 2^10`.
- Voxel type: unsigned integer, 8, 16 or 32 bits.
- Performance: >= 30 FPS is mandatory and >= 100 FPS is the target for a `500^3` byte volume.
- The design (structure, clarity, testability) has priority over the implementation details.

### 2.5 Assumptions and dependencies
- The NRRD file is valid and stores a 3D array.
- The system has enough memory for the volume (a `1023^3` `uint32` volume needs about 4.3 GB; the typical case is far smaller).
- Plane points are given in voxel coordinates, in the same index space as `M[x][y][z]`.
- The video encoder (OpenCV or ffmpeg) is installed.

---

## 3. External Interface Requirements

### 3.1 User interface (command line)

```
python Ex1.py <file.nrrd> <file.csv> <file.mp4> [options]
```

| Option | Meaning | Default |
|--------|---------|---------|
| `--strict` | Abort on the first invalid plane | off (skip with a warning) |
| `--interp {nearest,trilinear}` | Sampling method | `nearest` |
| `--window LO HI` | Intensity window for 16/32-bit data | global min / max |
| `--fps N` | Video frame rate | 30 |

Exit codes: `0` success; `1` invalid arguments or unreadable files; `2` invalid plane in `--strict` mode; `3` no valid plane found.

### 3.2 Input file interfaces

**NRRD (`file.nrrd`)**
- 3D data, type `uint8`, `uint16` or `uint32`.
- The array is used as `M[x][y][z]`. Orientation and spacing metadata are ignored (voxel index space).

**CSV (`file.csv`)**
- One plane per row, no header. Values are separated by commas; whitespace is ignored.
- 9 numbers = 3 points `a, b, c` (each point is `x, y, z`).
- 12 numbers = 4 points `a1, a2, b, c`.
- Blank lines and lines starting with `#` are ignored.
- Numbers may be integers or decimals.

### 3.3 Output file interface

**MP4 (`file.mp4`)**
- One frame for each valid plane, in CSV order.
- Frame size is derived from the plane size (section 5.4). If planes have different sizes, frames are padded with black to the size of the largest plane so that all frames have the same size.
- Pixel format: 8-bit grayscale (stored as 3-channel when the codec requires).

### 3.4 Software interfaces
`NumPy`, `pynrrd`, `OpenCV` (or `imageio[ffmpeg]`), `pytest` for tests.

---

## 4. Functional Requirements

Priority: **M** = mandatory, **S** = should, **C** = could.

### 4.1 Input

| ID | Requirement | Pri |
|----|-------------|-----|
| FR-1 | The system shall read a 3D volume from an NRRD file with element type `uint8`, `uint16` or `uint32`. | M |
| FR-2 | The system shall reject an NRRD file that is not 3-dimensional, has an unsupported type, or has a dimension outside `[1, 2^10)`, with a clear error message. | M |
| FR-3 | The system shall read a CSV file with planes of 3 points (9 values) or 4 points (12 values) per row. | M |
| FR-4 | The system shall report a malformed CSV row (wrong number of values, non-numeric values) with its line number. | M |
| FR-5 | The system shall ignore blank lines and comment lines in the CSV. | S |

### 4.2 Plane validation

| ID | Requirement | Pri |
|----|-------------|-----|
| FR-6 | For a plane `(a, b, c)` the system shall check `abs(dist(a,b) - dist(a,c)) < EPS`. | M |
| FR-7 | For a plane `(a1, a2, b, c)` the system shall check (1) `abs(dist(a1,b) - dist(a2,c)) < EPS`, (2) `dist(a1,a2) <= dist(b,c)`, and (3) that `a1->b` and `a2->c` lie on the same plane (coplanar within tolerance). | M |
| FR-8 | The system shall reject degenerate planes (zero length sides, or collinear points). | M |
| FR-9 | An invalid plane shall be reported with its row number and the violated rule. By default it is skipped; with `--strict` the program aborts. | M |

### 4.3 Projection

| ID | Requirement | Pri |
|----|-------------|-----|
| FR-10 | For each valid plane the system shall compute the projection image `E` using the algorithm of section 5. | M |
| FR-11 | The value of a pixel whose sample point lies outside the volume shall be `0`. | M |
| FR-12 | Pixels of `E` that are not covered by the triangle or trapezoid shall be `0`. | M |
| FR-13 | The default sampling shall be nearest neighbour (each sample point is rounded to the closest voxel). | M |
| FR-14 | The system shall offer trilinear interpolation as an alternative sampling method. | C |
| FR-15 | The computation shall not modify the volume. | M |
| FR-16 | Planes parallel to the X, Y or Z axes through integer coordinates shall return exactly the corresponding volume slice. | M |

### 4.4 Output

| ID | Requirement | Pri |
|----|-------------|-----|
| FR-17 | The system shall write one video frame for each valid plane, in the order of the CSV. | M |
| FR-18 | For 16/32-bit data the system shall map values linearly to 8 bits using the given window (`--window`) or the global minimum and maximum. | M |
| FR-19 | The system shall print a short summary at the end: number of planes read, projected, skipped, and the average projection time / FPS. | S |
| FR-20 | The system shall return a non-zero exit code on failure (section 3.1). | M |

---

## 5. Algorithm Specification

### 5.1 Inputs and outputs
Input: volume `M` of size `sx * sy * sz`, and a valid plane. Output: a 2D matrix `E` of the same element type as `M`.

### 5.2 Notation
For a 3-point plane `(a, b, c)`: `Dab = dist(a,b) = dist(a,c)`, `Dbc = dist(b,c)`, `v_ab = (b - a) / Dab`, `v_ac = (c - a) / Dab`.

For a 4-point plane `(a1, a2, b, c)`: the rows start on the segment `a1 - a2` and move along `v_1 = (b - a1) / D` and `v_2 = (c - a2) / D` where `D = dist(a1,b) = dist(a2,c)`. The 3-point case is the special case `a1 = a2 = a`.

### 5.3 Procedure
1. Allocate `E` of size `Dab * Dbc`, initialised to `0`.
2. Set `a1 = a`, `a2 = a`.
3. For each row `y = 0 .. Dab - 1`:
   1. `a1 = a1 + v_ab`, `a2 = a2 + v_ac`.
   2. `d12 = dist(a1, a2)`, `dp = (a2 - a1) / d12`, `p = a1`.
   3. For each `x1 = 0 .. floor(d12)`:
      1. `v = M[round(p)]`, or `0` if `round(p)` is outside `M`.
      2. `x = Dbc/2 - d12/2 + x1`.
      3. `E[y][x] = v`.
      4. `p = p + dp`.
4. Return `E`.

### 5.4 Output size
`E` has `round(Dab)` rows and `round(Dbc)` columns. For 4-point planes, `Dbc` is the distance `dist(b,c)`. Indices are clipped to the image bounds.

### 5.5 Special cases

| Case | Behaviour |
|------|-----------|
| `d12 = 0` (first row of a triangle) | The row contains a single sample at `a1`. |
| Plane entirely outside the volume | `E` is all zeros. |
| Plane partly outside the volume | Only the outside pixels are `0`. |
| Non-integer distances | Exact floating point values are used for steps; sizes are rounded. |
| Dimension of size 1 | Handled normally; samples with an index outside `[0, size)` are `0`. |

---

## 6. Non-Functional Requirements

| ID | Category | Requirement | Pri |
|----|----------|-------------|-----|
| NFR-1 | Performance | The projection of a `500^3` byte volume shall run at >= 30 FPS (mandatory) and >= 100 FPS (target), measured without video encoding. | M |
| NFR-2 | Performance | The volume shall be loaded once; per-frame work shall not copy the volume. | M |
| NFR-3 | Memory | Output and temporary buffers shall be reused between frames where the plane size is unchanged. | S |
| NFR-4 | Correctness | The result shall be deterministic: the same inputs give identical outputs. | M |
| NFR-5 | Robustness | The program shall not crash for any plane position or orientation; all failures shall produce an error message and an exit code. | M |
| NFR-6 | Maintainability | The code shall be split into independent modules (CSV reading, geometry, sampling, projection, video) with a documented interface, so each can be tested alone. | M |
| NFR-7 | Extensibility | A new sampling method shall be addable without changing the projector (Strategy pattern). | S |
| NFR-8 | Testability | All modules except video encoding shall be testable without any external file. | S |
| NFR-9 | Portability | The program shall run on Linux, macOS and Windows with Python 3.10+. | S |
| NFR-10 | Usability | Error messages shall name the file, the row or field, and the reason. | S |
| NFR-11 | Documentation | The repository shall contain a README, this SRS, and docstrings for all public classes and functions. | M |

---

## 7. Data Requirements

| File | Purpose |
|------|---------|
| `data/MRHead.nrrd` | Real MRI volume from 3D Slicer (the sample given in the assignment). |
| `data/synthetic_cube_64.nrrd` | Small generated volume with known values (e.g. `M[x][y][z] = x + y + z`), used for exact tests. |
| `data/synthetic_500.nrrd` | `500^3` `uint8` volume for performance tests. |
| `data/synthetic_u16.nrrd`, `data/synthetic_u32.nrrd` | Volumes of the other supported types. |
| `data/planes_axes.csv` | Planes parallel to the X, Y and Z axes. |
| `data/planes_rotation.csv` | A smooth sequence of rotating planes for the demo video. |
| `data/planes_invalid.csv` | Invalid and degenerate planes for error handling tests. |
| `data/planes_outside.csv` | Planes partly or entirely outside the volume. |

---

## 8. Verification and Acceptance

| Requirement | Method | Test |
|-------------|--------|------|
| FR-1, FR-2 | Unit test | Load valid files of each type; load a 2D file, a wrong type and a bad size. |
| FR-3 - FR-5 | Unit test | Parse valid, malformed, commented CSV files. |
| FR-6 - FR-9 | Unit test | Boundary of `EPS`, 3 vs 4 points, non-coplanar points, degenerate planes, `--strict`. |
| FR-10 - FR-12 | Reference test | The vectorized projector equals a literal loop implementation of section 5.3 on random planes. |
| FR-13 - FR-14 | Unit test | Rounding behaviour; trilinear on a linear volume gives exact values. |
| FR-16 | Correctness test | Axis aligned planes equal NumPy slices of the synthetic cube. |
| FR-17 - FR-20 | End to end test | Run the demo; check frame count, frame size, summary, and exit codes. |
| NFR-1 | Performance test | Average FPS over 1000 random planes on `synthetic_500.nrrd`. |
| NFR-4, NFR-5 | Property test | Repeated runs identical; random and extreme planes do not crash. |

**Acceptance criteria**
1. `python Ex1.py MRHead.nrrd planes_rotation.csv out.mp4` finishes without error and produces a playable video with one frame per valid plane.
2. All tests pass.
3. The performance test reaches at least 30 FPS (target 100 FPS).
4. README and SRS are present and consistent with the code.

---

## 9. Traceability Matrix

| Assignment item | Requirements |
|-----------------|--------------|
| 3D data, 8/16/32-bit, size limits | FR-1, FR-2 |
| Plane by 3 or 4 points, EPS rules (Q&A) | FR-3, FR-6 - FR-9 |
| Projection algorithm | FR-10 - FR-16, section 5 |
| Points outside M give 0 | FR-11 |
| Demo `python Ex1.py file.nrrd file.csv file.mp4` | section 3.1, FR-17 |
| Frame per plane in CSV order | FR-17 |
| >= 30 FPS (target 100) | NFR-1 |
| Design quality, robustness | NFR-5 - NFR-8 |
| Repository with README and SRS | NFR-11 |
| Testing strategy including demo | section 8 |

---

## 10. Open Issues / Assumptions to Confirm
- The assignment names the demo `Ex0.py` in one place; this SRS uses `Ex1.py`.
- Voxel coordinates are used for plane points (no physical spacing or orientation from the NRRD header). Confirm this with the course staff if physical units are expected.
- Frames of different sizes are padded with black to the largest size. Confirm whether the video should instead be resized.
