import cv2
import os
import numpy as np
import pywt
from scipy.fftpack import dct, idct, fft2, ifft2, fftshift, ifftshift
from watermark_embedding import block_dct2d, block_idct2d
from skimage.metrics import structural_similarity as ssim
import argparse

def inverse_arnold_transform(image, original_shape, iterations=1):
    if image.shape[0] != image.shape[1]:
        raise ValueError("Image must be square for the inverse Arnold transform.")
    n = image.shape[0]
    unscrambled_image = np.copy(image)
    for _ in range(iterations):
        original_image = np.zeros_like(unscrambled_image)
        for x_prime in range(n):
            for y_prime in range(n):
                x = (x_prime - y_prime) % n
                y = (-x_prime + 2 * y_prime) % n
                original_image[x, y] = unscrambled_image[x_prime, y_prime]
        unscrambled_image = original_image
    return cv2.resize(unscrambled_image, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_AREA)



def smooth_watermark(img, kernel_size=5, sigma=5.0):
    smoothed_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    return smoothed_img
    
def extract_frequency_tensor(image_path, block_size=8):

    img = cv2.imread(image_path)

    channels = cv2.split(img)

    features = []

    for ch in channels:

        # DCT
        dct_img = block_dct2d(
            ch.astype(np.float32),
            block_size
        )

        # DWT
        cA, (cH, cV, cD) = pywt.dwt2(
            dct_img,
            'haar'
        )

        features.extend([
            cA, cH, cV, cD
        ])

    # [12, H, W]
    features = np.stack(features,
                        axis=0)

    # normalize
    features = (
        features - features.mean()
    ) / (
        features.std() + 1e-8
    )

    return torch.tensor(
        features,
        dtype=torch.float32
    )


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--cover", required=True)
    parser.add_argument("--watermark", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--watermarked_image_path", required=True)
    parser.add_argument("--scaling_factor", type=float, required=True)
    parser.add_argument("--arnold_iterations", type=int, required=True)

    args = parser.parse_args()

    original_cover_path = args.cover
    watermarked_image_path = args.watermarked_image_path
    output_path = args.output
    original_watermark_path = args.watermark
    scaling_factor = args.scaling_factor
    arnold_iterations = args.arnold_iterations

    try:
        # --- Extract watermark ---
        extracted_watermark = watermark_extraction_process(
            original_cover_path,
            watermarked_image_path,
            original_watermark_path,
            scaling_factor,
            arnold_iterations
        )

        output_path = f"{output_path}/extracted_watermark.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, extracted_watermark)
        print(f"Watermark extraction complete. Extracted watermark saved to {output_path}")

        # --- Load images ---
        original_watermark = cv2.imread(original_watermark_path, cv2.IMREAD_GRAYSCALE)
        cover = cv2.imread(original_cover_path)
        watermarked = cv2.imread(watermarked_image_path)

        # --- Debug checks ---
        print(f"Original watermark shape: {None if original_watermark is None else original_watermark.shape}")
        print(f"Extracted watermark shape: {None if extracted_watermark is None else extracted_watermark.shape}")
        print(f"Cover image shape: {None if cover is None else cover.shape}")
        print(f"Watermarked image shape: {None if watermarked is None else watermarked.shape}")

        # --- Compare watermarks ---
        if original_watermark is not None and extracted_watermark is not None:
            extracted_resized = cv2.resize(
                extracted_watermark,
                (original_watermark.shape[1], original_watermark.shape[0]),
                interpolation=cv2.INTER_AREA
            )

            _, original_watermark = cv2.threshold(original_watermark, 128, 255, cv2.THRESH_BINARY)

            # Convert both to uint8
            extracted_resized = extracted_resized.astype(np.uint8)
            original_watermark = original_watermark.astype(np.uint8)

            psnr_wm = cv2.PSNR(original_watermark, extracted_resized)
            ssim_wm, _ = ssim(original_watermark, extracted_resized, full=True)

            print(f"[Watermark]  PSNR: {psnr_wm:.2f} dB, SSIM: {ssim_wm:.4f}")
        else:
            print("[Watermark] Could not compute — one of the images is missing.")

        # --- Compare cover vs watermarked ---
        if cover is not None and watermarked is not None:
            cover = cover.astype(np.uint8)
            watermarked = watermarked.astype(np.uint8)

            if cover.shape == watermarked.shape:
                psnr_img = cv2.PSNR(cover, watermarked)
                ssim_img, _ = ssim(cover, watermarked, channel_axis=-1, full=True)
                print(f"[Images]     PSNR: {psnr_img:.2f} dB, SSIM: {ssim_img:.4f}")
            else:
                print("[Images] Shapes do not match — resizing watermarked image.")
                watermarked_resized = cv2.resize(watermarked, (cover.shape[1], cover.shape[0]))
                psnr_img = cv2.PSNR(cover, watermarked_resized)
                ssim_img, _ = ssim(cover, watermarked_resized, channel_axis=-1, full=True)
                print(f"[Images]     PSNR: {psnr_img:.2f} dB, SSIM: {ssim_img:.4f}")
        else:
            print("[Images] Could not compute — one of the images is missing.")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()