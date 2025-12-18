import os
import sys
import glob
import math
import shutil
import argparse
import re
from unicodedata import normalize
from multiprocessing import Process, JoinableQueue

import numpy as np
from PIL import Image, ImageFilter, ImageStat
import openslide
from openslide import open_slide, ImageSlide
from openslide.deepzoom import DeepZoomGenerator

Image.MAX_IMAGE_PIXELS = None
VIEWER_SLIDE_NAME = "slide"
def is_informative(tile, tile_size, threshold):
    """
    Edge-density based tissue filtering.
    """
    edge = tile.filter(ImageFilter.FIND_EDGES)
    edge_score = np.mean(ImageStat.Stat(edge).sum) / (tile_size ** 2)
    return edge_score > threshold
class TileWorker(Process):
    def __init__(self, queue, slidepath, tile_size, overlap,
                 limit_bounds, quality, threshold):
        super().__init__(daemon=True)
        self.queue = queue
        self.slidepath = slidepath
        self.tile_size = tile_size
        self.overlap = overlap
        self.limit_bounds = limit_bounds
        self.quality = quality
        self.threshold = threshold
        self.slide = None

    def run(self):
        self.slide = open_slide(self.slidepath)
        dz = None
        last_associated = None

        while True:
            data = self.queue.get()
            if data is None:
                self.queue.task_done()
                break

            associated, level, address, outfile = data
            if associated != last_associated:
                dz = self._get_dz(associated)
                last_associated = associated

            try:
                tile = dz.get_tile(level, address).convert("RGB")
                w, h = tile.size
                if is_informative(tile, self.tile_size, self.threshold):
                    if w != self.tile_size or h != self.tile_size:
                        tile = tile.resize((self.tile_size, self.tile_size))
                    tile.save(outfile, quality=self.quality)
            except Exception:
                pass

            self.queue.task_done()

    def _get_dz(self, associated=None):
        if associated is None:
            image = self.slide
        else:
            image = ImageSlide(self.slide.associated_images[associated])

        return DeepZoomGenerator(
            image,
            tile_size=self.tile_size,
            overlap=self.overlap,
            limit_bounds=self.limit_bounds
        )
class DeepZoomImageTiler:
    def __init__(self, dz, basename, target_levels,
                 base_mag, fmt, associated, queue):
        self.dz = dz
        self.basename = basename
        self.target_levels = target_levels
        self.base_mag = int(base_mag)
        self.format = fmt
        self.associated = associated
        self.queue = queue

    def run(self):
        dz_levels = [self.dz.level_count - i - 1 for i in self.target_levels]
        mag_levels = [int(self.base_mag / (2 ** i)) for i in self.target_levels]

        for mag, level in zip(mag_levels, dz_levels):
            out_dir = os.path.join(f"{self.basename}_files", str(mag))
            os.makedirs(out_dir, exist_ok=True)

            cols, rows = self.dz.level_tiles[level]
            for row in range(rows):
                for col in range(cols):
                    outfile = os.path.join(out_dir, f"{col}_{row}.{self.format}")
                    self.queue.put((self.associated, level, (col, row), outfile))
class DeepZoomStaticTiler:
    def __init__(self, slidepath, basename, mag_levels, base_mag,
                 objective, fmt, tile_size, overlap,
                 limit_bounds, quality, workers, threshold):

        self.slide = open_slide(slidepath)
        self.basename = basename
        self.mag_levels = mag_levels
        self.base_mag = base_mag
        self.objective = objective
        self.format = fmt
        self.tile_size = tile_size
        self.overlap = overlap
        self.limit_bounds = limit_bounds

        self.queue = JoinableQueue(2 * workers)
        for _ in range(workers):
            TileWorker(
                self.queue, slidepath, tile_size, overlap,
                limit_bounds, quality, threshold
            ).start()

    def run(self):
        self._run_slide()
        self._shutdown()

    def _run_slide(self):
        dz = DeepZoomGenerator(
            self.slide,
            self.tile_size,
            self.overlap,
            limit_bounds=self.limit_bounds
        )

        mag_base = self.slide.properties.get(
            openslide.PROPERTY_NAME_OBJECTIVE_POWER,
            self.objective
        )

        first_level = int(math.log2(float(mag_base) / self.base_mag))
        target_levels = [i + first_level for i in self.mag_levels][::-1]

        tiler = DeepZoomImageTiler(
            dz, self.basename, target_levels,
            mag_base, self.format, None, self.queue
        )
        tiler.run()

    def _shutdown(self):
        for _ in range(self.queue.qsize()):
            self.queue.put(None)
        self.queue.join()
def organize_patches(slide_path, out_base, ext="jpeg"):
    slide_name = os.path.splitext(os.path.basename(slide_path))[0]
    class_name = os.path.basename(os.path.dirname(slide_path))

    bag_dir = os.path.join(out_base, class_name, slide_name)
    os.makedirs(bag_dir, exist_ok=True)

    patches = glob.glob("WSI_temp_files/*/*." + ext)
    for p in patches:
        shutil.move(p, os.path.join(bag_dir, os.path.basename(p)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("WSI Patch Extraction (CMA-MIL)")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing class/slide.svs")
    parser.add_argument("--slide_format", type=str, default="svs")
    parser.add_argument("--tile_size", type=int, default=224)
    parser.add_argument("--base_mag", type=float, default=20)
    parser.add_argument("--magnifications", type=int, nargs="+", default=[0])
    parser.add_argument("--objective", type=float, default=20)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--quality", type=int, default=70)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=15)
    parser.add_argument("--out_dir", type=str, default="patches")
    args = parser.parse_args()

    slides = (
        glob.glob(os.path.join(args.data_root, "*/*." + args.slide_format)) +
        glob.glob(os.path.join(args.data_root, "*/*/*." + args.slide_format))
    )

    for idx, slide in enumerate(slides):
        print(f"[{idx+1}/{len(slides)}] Processing {slide}")

        DeepZoomStaticTiler(
            slide,
            "WSI_temp",
            tuple(sorted(args.magnifications)),
            args.base_mag,
            args.objective,
            "jpeg",
            args.tile_size,
            args.overlap,
            True,
            args.quality,
            args.workers,
            args.threshold
        ).run()

        organize_patches(slide, args.out_dir)
        shutil.rmtree("WSI_temp_files")

    print(f"Patch extraction completed for {len(slides)} slides.")
