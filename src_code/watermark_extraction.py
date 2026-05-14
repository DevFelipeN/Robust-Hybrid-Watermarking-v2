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
    
def watermark_extraction_process(original_cover_path, watermarked_image_path, original_watermark_path, scaling_factor, arnold_iterations=1):
    # Load images
    I_O = cv2.imread(original_cover_path)
    if I_O is None:
        raise FileNotFoundError(f"Original cover image not found: {original_cover_path}")
    I_W = cv2.imread(watermarked_image_path)
    if I_W is None:
        raise FileNotFoundError(f"Watermarked image not found: {watermarked_image_path}")
    
    # --- GET ORIGINAL WATERMARK SHAPE FROM THE IMAGE FILE ---
    watermark_original = cv2.imread(original_watermark_path, cv2.IMREAD_GRAYSCALE) #Carrega a imagem da marca em tons de cinza
    if watermark_original is None:
        raise FileNotFoundError(f"Original watermark image not found: {original_watermark_path}")

    #Pega o tamanho da imagem da marca original    
    original_watermark_shape = watermark_original.shape
    original_h, original_w = original_watermark_shape
    # --- END ---

    block_size = 8 # tamanho dos blocos de aplicação do DCT

    # Separate images into B, G, R components and perform DCT
    B_O, G_O, R_O = cv2.split(I_O) # Imagem original
    B_W, G_W, R_W = cv2.split(I_W) # Imagem marcada

    # Aplicando DCT nos canais das duas imagens
    R_O_DCT = block_dct2d(R_O.astype(np.float32), block_size)
    G_O_DCT = block_dct2d(G_O.astype(np.float32), block_size)
    B_O_DCT = block_dct2d(B_O.astype(np.float32), block_size)
    R_W_DCT = block_dct2d(R_W.astype(np.float32), block_size)
    G_W_DCT = block_dct2d(G_W.astype(np.float32), block_size)
    B_W_DCT = block_dct2d(B_W.astype(np.float32), block_size)

    # Aplicando DWT nos resultados de DCT
    # Perform DWT on the DCT transformed color components
    coeffs_R_O_DCT_DWT = pywt.dwt2(R_O_DCT, 'haar') 
    coeffs_G_O_DCT_DWT = pywt.dwt2(G_O_DCT, 'haar')
    coeffs_B_O_DCT_DWT = pywt.dwt2(B_O_DCT, 'haar')
    coeffs_R_W_DCT_DWT = pywt.dwt2(R_W_DCT, 'haar')
    coeffs_G_W_DCT_DWT = pywt.dwt2(G_W_DCT, 'haar')
    coeffs_B_W_DCT_DWT = pywt.dwt2(B_W_DCT, 'haar')

    # -- ATÉ AQUI AS IMAGENS ORIGINAIS E MARCADAS SEPARADAS POR CANAIS, ESTÃO TRANFORMADAS PARA O DOMINIO DA FREQUÊNCIA --------

    def extract_channel_watermark(coeffs_O, coeffs_W, scaling_factor, original_watermark_shape, block_size=8):
        # pega as 4 bandas da DWT: LL, LH, HL, HH
        cA_O, (cH_O, cV_O, cD_O) = coeffs_O
        cA_W, (cH_W, cV_W, cD_W) = coeffs_W

        # Extract the watermark DCT coefficients (inverting embedding formula)
        W_A = (cA_W - cA_O) / scaling_factor 
        W_H = (cH_W - cH_O) / scaling_factor
        W_V = (cV_W - cV_O) / scaling_factor
        W_D = (cD_W - cD_O) / scaling_factor

        # 'Cria uma imagem vazia do tamanho da imagem original da marca
        # Resize all components to original watermark size using frequency-domain alignment
        target_h, target_w = original_watermark_shape #
        recon_h, recon_w = target_h, target_w
        W_ext_DCT = np.zeros((recon_h, recon_w), dtype=np.float32)

        # Use DCT-based resizing (frequency-domain alignment)
        def dct_resize(block, out_shape):
            # 1. Perform full DCT
            block_dct = dct(dct(block.T, norm='ortho').T, norm='ortho')
            # 2. Zero-pad or crop in frequency domain
            h_out, w_out = out_shape
            h_in, w_in = block_dct.shape
            resized_dct = np.zeros((h_out, w_out), dtype=np.float32)
            h_min = min(h_in, h_out)
            w_min = min(w_in, w_out)
            resized_dct[:h_min, :w_min] = block_dct[:h_min, :w_min]
            # 3. Inverse DCT
            resized_block = idct(idct(resized_dct.T, norm='ortho').T, norm='ortho')
            return resized_block

        mid_h, mid_w = recon_h // 2, recon_w // 2 # Define o centro da imagem da marca
        W_ext_DCT[:mid_h, :mid_w] = dct_resize(W_A, (mid_h, mid_w))
        W_ext_DCT[:mid_h, mid_w:] = dct_resize(W_H, (mid_h, recon_w - mid_w))
        W_ext_DCT[mid_h:, :mid_w] = dct_resize(W_V, (recon_h - mid_h, mid_w))
        W_ext_DCT[mid_h:, mid_w:] = dct_resize(W_D, (recon_h - mid_h, recon_w - mid_w))

        # Invert block-wise DCT
        W_ext = block_idct2d(W_ext_DCT, block_size)
        return W_ext

    
    # Extract watermark from each channel using the helper function
    W_R_ext = extract_channel_watermark(coeffs_R_O_DCT_DWT, coeffs_R_W_DCT_DWT, scaling_factor, original_watermark_shape)
    W_G_ext = extract_channel_watermark(coeffs_G_O_DCT_DWT, coeffs_G_W_DCT_DWT, scaling_factor, original_watermark_shape)
    W_B_ext = extract_channel_watermark(coeffs_B_O_DCT_DWT, coeffs_B_W_DCT_DWT, scaling_factor, original_watermark_shape)

    # Average the three extracted channels to get a single, combined result
    W_ext_combined = (W_R_ext + W_G_ext + W_B_ext) / 3.0

    # Normalize and binarize the single-channel combined result
    W_norm = cv2.normalize(W_ext_combined, None, 0, 255, cv2.NORM_MINMAX)
    W_norm_uint8 = W_norm.astype(np.uint8)
    
    # Use the threshold you found to be effective, e.g., 99
    _, W_binary = cv2.threshold(W_norm_uint8, 128, 255, cv2.THRESH_BINARY)
    
    # Apply inverse Arnold transform to the clean binary watermark
    extracted_watermark = inverse_arnold_transform(W_binary, original_watermark_shape, arnold_iterations)
    #extracted_watermark = smooth_watermark(extracted_watermark)

    return extracted_watermark


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