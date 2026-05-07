"""
BatchStack — FITS Batch Stacking Engine
========================================
Runs in a QThread so the GUI stays responsive.
Supports: Average, Maximum, Median, K-Sigma Regular/Median.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
from astropy.io import fits
from PySide6.QtCore import QThread, Signal


# ── helpers ────────────────────────────────────────────────────────────────────

def find_image_hdu(hdul: fits.HDUList) -> int:
    """Return the index of the first HDU that contains 2-D image data.

    Uses header NAXIS keywords instead of accessing .data directly,
    which avoids triggering BZERO/BSCALE errors on memmap'd files.
    """
    # Try primary first
    hdr0 = hdul[0].header
    if hdr0.get("NAXIS", 0) >= 2:
        return 0
    # Fall back to first image extension
    for idx, hdu in enumerate(hdul[1:], start=1):
        if hdu.header.get("NAXIS", 0) >= 2:
            return idx
    # Last resort: primary even if empty (will error later with a clear msg)
    return 0


def sort_files_by_header(file_list: list[Path]) -> list[Path]:
    """Sort FITS files by the DATE-OBS header keyword."""
    dated: list[tuple[str, Path]] = []
    for f in file_list:
        try:
            # Only read headers — no need to touch pixel data for sorting
            hdr = fits.getheader(f)
            date_obs = hdr.get("DATE-OBS", "") or ""
            dated.append((date_obs or f.name, f))
        except Exception:
            dated.append((f.name, f))
    dated.sort(key=lambda x: x[0])
    return [f for _, f in dated]


def k_sigma_combine(
    stack: np.ndarray,
    kappa: float = 2.0,
    iterations: int = 5,
    use_median: bool = False,
) -> np.ndarray:
    """
    Pixel-wise K-Sigma clipping for a 3-D stack (N, H, W).

    Regular mode uses mean/std for clipping and returns the clipped average.
    Median mode uses median/std for clipping and returns the clipped median.
    """
    mask = np.zeros_like(stack, dtype=bool)

    for _ in range(iterations):
        masked = np.ma.array(stack, mask=mask)
        center = np.ma.median(masked, axis=0) if use_median else masked.mean(axis=0)
        std = masked.std(axis=0)

        # Avoid division by zero
        std = np.where(std == 0, 1e-10, std)

        new_mask = mask | (np.abs(stack - center) > (kappa * std))

        if np.array_equal(new_mask, mask):
            break
        mask = new_mask

    masked = np.ma.array(stack, mask=mask)
    if use_median:
        result = np.ma.median(masked, axis=0).filled(np.nan)
    else:
        result = masked.mean(axis=0).filled(np.nan)

    return result.astype(np.float32)


# ── Stacking worker ───────────────────────────────────────────────────────────

class StackWorker(QThread):
    """
    Background worker that performs batch FITS stacking.

    Signals
    -------
    progress(current, total, message)
        Fired for each meaningful step; drives the progress bar + log.
    group_done(group_index, output_path)
        Fired after each group is successfully written.
    finished(total_groups, elapsed_seconds)
        Fired when all groups are done (or cancelled).
    error(message)
        Fired on unrecoverable error.
    """

    progress = Signal(int, int, str)
    group_done = Signal(int, str)
    finished = Signal(int, float)
    error = Signal(str)

    # Stacking method constants
    AVERAGE = "Average"
    MAXIMUM = "Maximum"
    MEDIAN = "Median"
    K_SIGMA_REGULAR = "K-Sigma (Regular)"
    K_SIGMA_MEDIAN = "K-Sigma (Median)"

    METHODS = [AVERAGE, MAXIMUM, MEDIAN, K_SIGMA_REGULAR, K_SIGMA_MEDIAN]

    def __init__(
        self,
        input_folder: str,
        output_folder: str,
        group_size: int = 3,
        method: str = AVERAGE,
        kappa: float = 2.0,
        iterations: int = 5,
        naming_style: str = "sequential",   # "sequential" | "range"
        prefix: str = "stack",
        include_incomplete: bool = False,
        sort_by: str = "filename",           # "filename" | "date-obs"
        dtype: str = "float32",              # "float32" | "float64"
        parent=None,
    ):
        super().__init__(parent)
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.group_size = group_size
        self.method = method
        self.kappa = kappa
        self.iterations = iterations
        self.naming_style = naming_style
        self.prefix = prefix
        self.include_incomplete = include_incomplete
        self.sort_by = sort_by
        self.dtype = np.float32 if dtype == "float32" else np.float64
        self._cancel = False

    # ── public API ─────────────────────────────────────────────────────────

    def cancel(self):
        """Request graceful cancellation."""
        self._cancel = True

    # ── main loop ──────────────────────────────────────────────────────────

    def run(self):  # noqa: C901 (complexity is fine for a worker)
        t0 = time.perf_counter()
        try:
            # Discover files
            files = list(self.input_folder.glob("*.fits")) + \
                    list(self.input_folder.glob("*.fit")) + \
                    list(self.input_folder.glob("*.fts")) + \
                    list(self.input_folder.glob("*.FITS")) + \
                    list(self.input_folder.glob("*.FIT")) + \
                    list(self.input_folder.glob("*.FTS"))

            # De-duplicate (case-insensitive filesystems)
            seen: set[str] = set()
            unique: list[Path] = []
            for f in files:
                key = str(f).lower()
                if key not in seen:
                    seen.add(key)
                    unique.append(f)
            files = unique

            if not files:
                self.error.emit("No FITS files found in the selected folder.")
                return

            # Sort
            if self.sort_by == "date-obs":
                self.progress.emit(0, 0, "Sorting by DATE-OBS header...")
                files = sort_files_by_header(files)
            else:
                files = sorted(files, key=lambda p: p.name.lower())

            total_files = len(files)
            gs = self.group_size

            # Build groups
            groups: list[list[Path]] = []
            for i in range(0, total_files, gs):
                grp = files[i : i + gs]
                if len(grp) < gs and not self.include_incomplete:
                    continue
                groups.append(grp)

            total_groups = len(groups)
            if total_groups == 0:
                self.error.emit("No complete groups to stack.")
                return

            self.output_folder.mkdir(parents=True, exist_ok=True)

            # Process each group
            completed_count = 0
            for g_idx, group in enumerate(groups):
                if self._cancel:
                    self.progress.emit(g_idx, total_groups, "Cancelled by user.")
                    break

                first_file_idx = g_idx * gs + 1
                last_file_idx = first_file_idx + len(group) - 1
                label = f"Group {g_idx + 1}/{total_groups} (files {first_file_idx}-{last_file_idx})"
                self.progress.emit(g_idx, total_groups, f"Reading {label}...")

                # Read group data
                datas: list[np.ndarray] = []
                expected_shape: tuple[int, ...] | None = None
                header: Optional[fits.Header] = None

                for fpath in group:
                    if self._cancel:
                        break
                    try:
                        # memmap=False required: many FITS files use
                        # BZERO/BSCALE for unsigned-int rescaling, which
                        # is incompatible with memory mapping.
                        with fits.open(fpath, memmap=False) as hdul:
                            hdu_idx = find_image_hdu(hdul)
                            data = hdul[hdu_idx].data
                            if data is None:
                                self.progress.emit(
                                    g_idx, total_groups,
                                    f"Warning: Skipped {fpath.name} -- no image data"
                                )
                                continue
                            if data.ndim != 2:
                                self.progress.emit(
                                    g_idx, total_groups,
                                    f"Warning: Skipped {fpath.name} -- expected 2-D image data"
                                )
                                continue
                            if expected_shape is None:
                                expected_shape = data.shape
                            elif data.shape != expected_shape:
                                self.progress.emit(
                                    g_idx, total_groups,
                                    f"Warning: Skipped {fpath.name} -- shape {data.shape} "
                                    f"does not match {expected_shape}"
                                )
                                continue
                            datas.append(data.astype(self.dtype, copy=False))
                            if header is None:
                                header = hdul[hdu_idx].header.copy()
                    except Exception as exc:
                        self.progress.emit(
                            g_idx, total_groups,
                            f"Warning: Error reading {fpath.name}: {exc}"
                        )
                        continue

                if self._cancel:
                    self.progress.emit(g_idx, total_groups, "Cancelled by user.")
                    break

                if len(datas) == 0:
                    self.progress.emit(
                        g_idx, total_groups,
                        f"Skipping {label} -- no readable frames"
                    )
                    continue

                if len(datas) < gs and not self.include_incomplete:
                    self.progress.emit(
                        g_idx, total_groups,
                        f"Skipping {label} -- only {len(datas)} readable frame(s)"
                    )
                    continue

                # Stack
                self.progress.emit(g_idx, total_groups, f"Stacking {label} ({self.method})...")
                frame_count = len(datas)
                stack = np.stack(datas, axis=0)
                del datas  # free list references early

                stacked = self._combine(stack)
                del stack

                # Build output filename
                outname = self._make_outname(g_idx, first_file_idx, last_file_idx)
                outpath = self.output_folder / outname

                # Add stacking metadata to header
                if header is not None:
                    header["STACKN"] = (frame_count, "Number of frames in stack")
                    header["STACKMT"] = (self.method, "Stacking method")
                    header["STACKSW"] = ("BatchStack", "Stacking software")

                fits.writeto(outpath, stacked, header, overwrite=True)
                del stacked

                completed_count += 1
                self.group_done.emit(g_idx + 1, str(outpath))
                self.progress.emit(
                    g_idx + 1, total_groups,
                    f"Saved {outname}"
                )

            elapsed = time.perf_counter() - t0
            self.finished.emit(completed_count, elapsed)

        except Exception as exc:
            self.error.emit(str(exc))

    # ── private ────────────────────────────────────────────────────────────

    def _combine(self, stack: np.ndarray) -> np.ndarray:
        """Apply the selected stacking method to a (N, H, W) array."""
        if self.method == self.AVERAGE:
            return np.mean(stack, axis=0).astype(np.float32)
        elif self.method == self.MAXIMUM:
            return np.max(stack, axis=0).astype(np.float32)
        elif self.method == self.MEDIAN:
            return np.median(stack, axis=0).astype(np.float32)
        elif self.method == self.K_SIGMA_REGULAR:
            return k_sigma_combine(
                stack, kappa=self.kappa, iterations=self.iterations, use_median=False
            )
        elif self.method == self.K_SIGMA_MEDIAN:
            return k_sigma_combine(
                stack, kappa=self.kappa, iterations=self.iterations, use_median=True
            )
        else:
            return np.mean(stack, axis=0).astype(np.float32)

    def _make_outname(self, g_idx: int, first: int, last: int) -> str:
        """Generate the output FITS filename."""
        if self.naming_style == "range":
            return f"{self.prefix}_{first:04d}-{last:04d}.fits"
        else:  # sequential
            return f"{self.prefix}_{g_idx + 1:04d}.fits"
