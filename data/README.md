# Dataset Overview

- **Total Volume:** 2,661 images
- **Classes (3):**
  - `none`: 1,164 images (bare/damaged skin, traffic signs, billboards, blank walls, noise)
  - `tats`: 802 images (permanent body ink, linework, pigmentation)
  - `street_art`: 695 images (murals, spray paint on urban surfaces)
- **Sources:** Roboflow Universe public datasets and original self-collected captures.
- **Splits (80 / 10 / 10):**
  - **Train:** 2,128 images (931 `none`, 641 `tats`, 556 `street_art`)
  - **Validation:** 265 images (116 `none`, 80 `tats`, 69 `street_art`)
  - **Test:** 268 images (117 `none`, 81 `tats`, 70 `street_art`)
- **Preprocessing:** All images normalized to ImageNet statistics and resized to 224x224 pixels.
