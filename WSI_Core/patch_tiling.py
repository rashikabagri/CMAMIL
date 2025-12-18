import os
import openslide
from openslide.deepzoom import DeepZoomGenerator
from PIL import Image, ImageStat

def is_relevant_patch(patch, min_std, mean_thresh):
    gray = patch.convert("L")
    stat = ImageStat.Stat(gray)
    return stat.stddev[0] > min_std and stat.mean[0] < mean_thresh

def extract_patches_from_wsi(
    svs_path,
    save_dir,
    tile_size,
    resize,
    min_std,
    mean_thresh
):
    slide = openslide.OpenSlide(svs_path)
    tiles = DeepZoomGenerator(slide, tile_size=tile_size, overlap=0)

    os.makedirs(save_dir, exist_ok=True)

    for level in range(tiles.level_count):
        cols, rows = tiles.level_tiles[level]
        for i in range(cols):
            for j in range(rows):
                tile = tiles.get_tile(level, (i, j)).convert("RGB")
                if is_relevant_patch(tile, min_std, mean_thresh):
                    tile = tile.resize((resize, resize))
                    tile.save(os.path.join(save_dir, f"patch_{level}_{i}_{j}.jpg"))
