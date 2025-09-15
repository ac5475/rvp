"""
patches = {"1":"Torso", "2":"Torso", "3":"Right Hand", "4":"Left Hand", "5":"Left Foot", "6":"Right Foot", "7":"Upper Leg Right", "9":"Upper Leg Right",
   "8":"Upper Leg Left", "10":"Upper Leg Left", "11":"Lower Leg Right", "13":"Lower Leg Right", "12":"Lower Leg Left", "14":"Lower Leg Left",
   "15":"Upper Arm Left", "17":"Upper Arm Left", "16":"Upper Arm Right", "18":"Upper Arm Right", "19":"Lower Arm Left", "21":"Lower Arm Left",
   "20":"Lower Arm Right", "22":"Lower Arm Right", "23":"Head", "24":"Head"}
"""

import os
from typing import List, Union, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageOps
from torchvision import transforms
from diffusers import StableDiffusionInpaintPipeline

# Set to True to save intermediate results for debugging
DEBUG = False
DEBUG_OUTPUT_PATH = "output/tmp"


def square_pad(img, bg):
    width, height = img.size
    if width == height:
        return img
    elif width > height:
        result = Image.new(img.mode, (width, width), bg)
        result.paste(img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(img.mode, (height, height), bg)
        result.paste(img, ((height - width) // 2, 0))
        return result


def crop_top_left(img, bg=None):
    width, height = img.size
    min_dim = min(width, height)
    if width == height:
        return img
    else:
        return img.crop((0, 0, min_dim, min_dim))


def seg_from_densepose(dp, img_size):
    seg = Image.new("P", img_size, color=0)
    i = torch.argmax(dp["scores"]).item()
    xmin, ymin, _, _ = dp["pred_boxes_XYXY"][i]
    xmin, ymin = round(xmin.item()), round(ymin.item())
    labels = Image.fromarray(
        dp["pred_densepose"][0].labels.to("cpu").to(torch.uint8).numpy()
    ).convert("P")
    seg.paste(labels, (xmin, ymin))
    seg = np.asarray(seg)
    return seg


def load_data(img_path, dp_path):
    img = Image.open(img_path).convert("RGB")
    with open(dp_path, "rb") as fp:
        dp = torch.load(fp)
    seg = seg_from_densepose(dp, img.size)
    return img, seg


def generate_masks(seg, mask_padding=70, mask_erosion=3, head_erosion=5):
    seg_body = np.logical_and(1 <= seg, seg <= 22)
    seg_head = seg >= 23
    seg_head_hands_feet = np.logical_or(np.logical_and(3 <= seg, seg <= 6), seg >= 23)
    seg_body_hands_feet = np.logical_or(
        np.logical_and(1 <= seg, seg <= 2), np.logical_and(7 <= seg, seg <= 22)
    )

    if DEBUG:
        Image.fromarray(seg_body).save(os.path.join(DEBUG_OUTPUT_PATH, "body_segx.png"))
        Image.fromarray(seg_head).save(os.path.join(DEBUG_OUTPUT_PATH, "head_segx.png"))
        Image.fromarray(seg_head_hands_feet).save(
            os.path.join(DEBUG_OUTPUT_PATH, "head_hands_feet_segx.png")
        )
        Image.fromarray(seg_body_hands_feet).save(
            os.path.join(DEBUG_OUTPUT_PATH, "body_nohands_nofeet_segx.png")
        )

    mask_body = seg_body_hands_feet.astype(np.uint8)
    kernel_body = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (mask_padding + 1, mask_padding + 1)
    )
    mask_body = cv2.dilate(mask_body, kernel_body).astype(np.uint8) * 255

    if DEBUG:
        plt.imsave(
            os.path.join(DEBUG_OUTPUT_PATH, "circular_kernelx.png"),
            kernel_body,
            cmap="gray",
        )
        plt.imsave(
            os.path.join(DEBUG_OUTPUT_PATH, "mask_bodyx.png"), mask_body, cmap="gray"
        )

    mask_head = seg_head.astype(np.uint8)
    kernel_head = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (head_erosion * 2 + 1, head_erosion * 2 + 1)
    )
    mask_head = cv2.erode(mask_head, kernel_head).astype(np.uint8) * 255

    mask_head_hands_feet = seg_head_hands_feet.astype(np.uint8)
    kernel_sub = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (mask_erosion * 2 + 1, mask_erosion * 2 + 1)
    )
    mask_head_hands_feet = (
        cv2.erode(mask_head_hands_feet, kernel_sub).astype(np.uint8) * 255
    )

    mask_body_final = (
        np.logical_and(mask_body, np.logical_not(mask_head_hands_feet)).astype(np.uint8)
        * 255
    )
    if DEBUG:
        plt.imsave(os.path.join(DEBUG_OUTPUT_PATH, "maskx.png"), mask_body, cmap="gray")
        plt.imsave(
            os.path.join(DEBUG_OUTPUT_PATH, "mask_headx.png"), mask_head, cmap="gray"
        )
        plt.imsave(
            os.path.join(DEBUG_OUTPUT_PATH, "mask_head_hands_feetx.png"),
            mask_head_hands_feet,
            cmap="gray",
        )
        plt.imsave(
            os.path.join(DEBUG_OUTPUT_PATH, "mask_body_finalx.png"),
            mask_body_final,
            cmap="gray",
        )
    body = Image.fromarray(mask_body_final)
    head = Image.fromarray(mask_head)
    return body, head


def preprocess(img, body, head, square_fn=square_pad, size=512):
    if isinstance(img, Image.Image):
        bg = img.getpixel((0, 0))
    elif isinstance(img, torch.Tensor):
        bg = img[0, 0]
    else:
        raise Exception(f"Unknown image type {type(bg)}")
    img_square = square_fn(img, bg).resize((size, size))
    seg_square = square_fn(body, (0,)).resize(
        (size, size), resample=Image.Resampling.NEAREST
    )
    head_square = square_fn(head, (0,)).resize(
        (size, size), resample=Image.Resampling.NEAREST
    )

    if DEBUG:
        img_square.save(os.path.join(DEBUG_OUTPUT_PATH, "img_squarex.jpg"))
        seg_square.save(os.path.join(DEBUG_OUTPUT_PATH, "seg_squarex.png"))

    return img_square, seg_square, head_square


def run_model(
    model,
    img,
    seg,
    prompt,
    guidance_scale=20,
    num_images_per_prompt=1,
    num_inference_steps=100,
) -> Union[List[Image.Image], np.ndarray]:
    posprompt = "photograph, beautiful, detailed, detailed shoes, photorealism, detailed hands, detailed feet, detailed fingers, realistic lighting, natural lighting, crystal clear, detailed skin, ultra focus, sharp quality"
    negprompt = "disfigured, ugly, bad, cartoon, anime, 3d, painting, bad hands, bad feet, deformed hands, broken anatomy, deformed, unrealistic, missing body parts, unclear, blurry"
    prompt = prompt + posprompt

    # use the pipeline to generate the image
    images = model(
        prompt=prompt,
        image=img,
        mask_image=seg,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        negative_prompt=negprompt,
        num_inference_steps=num_inference_steps,
    ).images
    return images


def restore_head(img_orig: Image.Image, img_gen: Image.Image, head: Image.Image):
    """
    Paste the head region from the original image onto the generated image to correct distortions.
    Args:
        img_orig (Image.Image): The original input image.
        img_gen (Image.Image): The generated image to be corrected.
        head (Image.Image): The mask image indicating the head region.
    Returns:
        Image.Image: The generated image with the restored head region.
    """
    if DEBUG:
        img_gen.save(os.path.join(DEBUG_OUTPUT_PATH, "pre_restorationx.jpg"))
    img_gen.paste(img_orig.convert("RGB"), (0, 0), mask=head)
    if DEBUG:
        # Save debug images for inspection
        head.save(os.path.join(DEBUG_OUTPUT_PATH, "head_maskx.png"))
        blank = Image.new(mode=img_gen.mode, size=img_gen.size)
        blank.paste(img_orig.convert("RGB"), (0, 0), mask=head)
        blank.save(os.path.join(DEBUG_OUTPUT_PATH, "head_imagex.jpg"))
        img_gen.save(os.path.join(DEBUG_OUTPUT_PATH, "restored_headx.jpg"))
    return img_gen

def restore_head_alpha_blend(img_original: Image.Image, img_gen: Image.Image, head: Image.Image, alpha: float):
    """
    Alpha blend the head region from the original and generated images.
    Args:
        img_original (Image.Image): The original input image.
        img_gen (Image.Image): The generated image to be corrected.
        head (Image.Image): The mask image indicating the head region.
        alpha (float): The blending factor (0.0-1.0) for the head region.
    Returns:
        Image.Image: The generated image with the alpha-blended head region.
    """
    img = img_gen.copy()
    # Ensure both images are in RGB mode
    img_orig = img_original.copy().convert("RGB")
    img = img.convert("RGB")

    # Extract the head region from both images using the mask
    head_orig = Image.new(mode=img_gen.mode, size=img_gen.size)
    head_orig.paste(img_orig.convert("RGB"), (0, 0), mask=head)

    head_gen = Image.new(mode=img_gen.mode, size=img_gen.size)
    head_gen.paste(img_gen.convert("RGB"), (0, 0), mask=head)

    # Blend the head regions using the specified alpha
    head_blend = Image.blend(head_gen, head_orig, alpha)

    # Paste the blended head back onto the generated image
    img.paste(head_blend, (0, 0), mask=head)
    return img

def restore_head_poisson_blend(img_orig: Image.Image, img_gen: Image.Image, head: Image.Image, dilation_size=5):
    """
    Restore the head region using Poisson blending (OpenCV seamlessClone).
    Args:
        img_orig (Image.Image): Original image.
        img_gen (Image.Image): Generated image.
        head (Image.Image): Head mask.
        dilation_size (int): Amount to dilate the mask (controls blend area).
    Returns:
        Image.Image: The generated image with the Poisson-blended head region.
    """
    img_gen_copy = img_gen.copy()
    mask_head = np.array(head)

    # Dilate the mask to expand the blend area
    kernel_head = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilation_size * 2 + 1, dilation_size * 2 + 1)
    )
    mask_head_dilated = cv2.dilate(mask_head, kernel_head, iterations=1)

    # Convert images to OpenCV format
    img_orig_pois = cv2.cvtColor(np.array(img_orig), cv2.COLOR_RGB2BGR)
    img_gen_pois = cv2.cvtColor(np.array(img_gen_copy), cv2.COLOR_RGB2BGR)
    head_pois = mask_head_dilated  # Mask for blending

    # Find the center of the head region in the mask
    x, y, w, h = cv2.boundingRect(head_pois)
    center = (x + w // 2, y + h // 2)

    # Perform Poisson blending
    restored_img_pois = cv2.seamlessClone(
        img_orig_pois,            # Source image (original)
        img_gen_pois,             # Destination image (generated)
        head_pois,                # Mask defining the object to blend (head)
        center,                   # Center of the head region
        cv2.NORMAL_CLONE                # Blending mode
    )

    # Convert result back to PIL Image
    restored_img = Image.fromarray(cv2.cvtColor(restored_img_pois, cv2.COLOR_BGR2RGB))
    return restored_img


def restore_head_pyramid_blend(img_original: Image.Image, img_gen: Image.Image, head: Image.Image, levels: int):
    """
    Restore the head region using multi-resolution pyramid blending.
    Args:
        img_original (Image.Image): Original image.
        img_gen (Image.Image): Generated image.
        head (Image.Image): Head mask.
        levels (int): Number of pyramid levels.
    Returns:
        Image.Image: The generated image with the pyramid-blended head region.
    """
    # Convert PIL Images to normalized numpy arrays
    img_orig = img_original.copy()
    img_orig = np.array(img_orig).astype(np.float32) / 255.0
    img_gen = np.array(img_gen).astype(np.float32) / 255.0
    head = np.array(head).astype(np.float32) / 255.0

    # Smooth the mask to reduce edge artifacts
    head = cv2.GaussianBlur(head, (9, 9), sigmaX=3, sigmaY=3)
    head = np.clip(head, 0, 1)

    # Initialize Laplacian pyramids for both images and Gaussian pyramid for mask
    lp_orig = []
    lp_gen = []
    gp_mask = []

    # Start with initial Gaussian levels
    G_orig = img_orig.copy()
    G_gen = img_gen.copy()
    G_mask = head.copy()

    for i in range(levels - 1):
        # Downsample to next Gaussian level
        next_G_orig = cv2.pyrDown(G_orig)
        next_G_gen = cv2.pyrDown(G_gen)
        next_G_mask = cv2.pyrDown(G_mask)

        # Compute Laplacian for current level
        L_orig = G_orig - cv2.pyrUp(next_G_orig, dstsize=(G_orig.shape[1], G_orig.shape[0]))
        L_gen = G_gen - cv2.pyrUp(next_G_gen, dstsize=(G_gen.shape[1], G_gen.shape[0]))

        # Store Laplacians and mask
        lp_orig.append(L_orig)
        lp_gen.append(L_gen)
        gp_mask.append(G_mask)

        # Move to next level
        G_orig = next_G_orig
        G_gen = next_G_gen
        G_mask = next_G_mask

    # Add the smallest Gaussian level to Laplacian pyramids
    lp_orig.append(G_orig)
    lp_gen.append(G_gen)
    gp_mask.append(G_mask)

    # Ensure mask has 3 channels for blending and is normalized
    gp_mask = [np.repeat(np.clip(gm, 0, 1)[:, :, np.newaxis], 3, axis=2) for gm in gp_mask]

    # Blend Laplacians of both images using mask at each level
    L_blend = [lo * gm + lg * (1.0 - gm) for lo, lg, gm in zip(lp_orig, lp_gen, gp_mask)]

    # Reconstruct the final image from the blended pyramid
    restored_img = L_blend[-1]
    for i in range(levels - 2, -1, -1):
        restored_img = cv2.pyrUp(restored_img, dstsize=(L_blend[i].shape[1], L_blend[i].shape[0]))
        restored_img = cv2.add(restored_img, L_blend[i])

    # Clip values to valid range and convert back to PIL Image
    restored_img = np.clip(restored_img, 0, 1)
    restored_img = Image.fromarray((restored_img * 255).astype(np.uint8))
    return restored_img

def postprocess(img: Image.Image, orig_size: Tuple[int, int]):
    scaled_size = [round(d / max(orig_size) * 512) for d in orig_size]

    square_size = img.size

    left = max((square_size[0] - min(scaled_size[0], 512)) // 2, 0)
    top = max((square_size[1] - min(scaled_size[1], 512)) // 2, 0)
    right = left + scaled_size[0]
    bottom = top + scaled_size[1]
    return img.crop((left, top, right, bottom))


def design_garment(
    model,
    img: Image.Image,
    seg: Image.Image,
    prompt,
    mask_dilation=70,
    mask_erosion=3,
    head_erosion=5,
    guidance_scale=20,
    num_images_per_prompt=1,
    num_inference_steps=200,
    square_fn=square_pad,
    size=512,
    do_postprocess=True,
):
    img_size = img.size
    if DEBUG:
        plt.imsave(os.path.join(DEBUG_OUTPUT_PATH, "/segmentation.png"), seg)

    body, head = generate_masks(
        seg,
        mask_padding=mask_dilation,
        mask_erosion=mask_erosion,
        head_erosion=head_erosion,
    )
    if DEBUG:
        blank = Image.new(mode=img.mode, size=img.size)
        blank.paste(img.convert("RGB"), (0, 0), mask=body)
        blank.save(os.path.join(DEBUG_OUTPUT_PATH, "/masked_bodyx.jpg"))
        blank2 = Image.new(mode=img.mode, size=img.size)
        blank2.paste(img.convert("RGB"), (0, 0), mask=ImageOps.invert(body))
        blank2.save(os.path.join(DEBUG_OUTPUT_PATH, "/preserve_areax.jpg"))

    img, body, head = preprocess(
        img, body, head, square_fn=square_fn, size=size
    )

    imgs = run_model(
        model,
        img,
        body,
        prompt,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        num_inference_steps=num_inference_steps,
    )
    imgs_final = [restore_head(img, img_gen, head) for img_gen in imgs]
    if do_postprocess:
        imgs_final = [postprocess(img_fin, img_size) for img_fin in imgs_final]
    return imgs_final


def main():
    # Set paths
    images_dir = "sample_data/images"
    densepose_dir = "output/densepose"
    output_dir = "output/sample_output"

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Model input
    image_id = "000038_0"
    prompt = "Knee-length summer dress in blue with small pink flowers."

    # Load pipeline
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting", torch_dtype=torch.float16
    )
    pipeline = pipeline.to("cuda")

    # Load data
    image_path = os.path.join(images_dir, f"{image_id}.jpg")
    dp_path = os.path.join(densepose_dir, f"{image_id}.pkl")
    img, seg = load_data(image_path, dp_path)

    # Generate and save results
    imgs = design_garment(
        pipeline, img, seg, prompt, num_images_per_prompt=5
    )
    for i, img in enumerate(imgs):
        out_name = os.path.join(output_dir, f"{image_id}_{i}.jpg")
        img.save(out_name)
        print(f"Saved generated image to: {out_name}")


if __name__ == "__main__":
    main()
