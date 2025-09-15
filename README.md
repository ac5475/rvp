# RVP Project - Head restoration after Stable Diffusion Inpainting

## Folder Structure
- `pipeline.py`: Main script
- `analysis-preparations.ipynb`: Notebook for image preparation, such as cropping, applying blending, etc.
- `qualitative-analysis.ipynb`: Notebook for qualitative analysis of restoration methods.
- `quantitative-analysis-results.ipynb`: Notebook for quantitative analysis of restoration methods.
- `quantitative-analysis-images.zip`: Compressed dataset of images for quantitative analysis.
    - `alpha_blend/` - Images generated using alpha blending.
    - `cropped/` - Cropped versions of the original images.
        - `alpha_blend/` - Cropped images generated using alpha blending.
        - `generated/` - Cropped generated images.
        - `mask_paste/` - Cropped images for baseline.
        - `poisson_blend/` - Cropped images generated using Poisson blending.
        - `pyramid_blend/` - Cropped images generated using pyramid blending.
    - `generated/` - Generated images.
    - `mask_paste/` - Images for baseline.
    - `poisson_blend/` - Images generated using Poisson blending.
    - `pyramid_blend/` - Images generated using pyramid blending.

- `qualitative-analysis-images/`: Contains images and masks for qualitative analysis.
  - `generated/`: Generated images.
  - `masks/`: Binary head masks from segmentation.
  - `originals/`: Original images.

