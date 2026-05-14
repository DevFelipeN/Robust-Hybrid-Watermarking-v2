#importing the relevant libraries
import cv2
import numpy as np 
import pywt
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
import os
from skimage.metrics import structural_similarity as ssim
import argparse

#padd and crop function for images that are not sqare images
def make_square(img):
    height, width = img.shape[:2]
    size = min(height, width)
    top = (height - size) // 2
    left = (width - size) // 2
    return img[top:top+size, left:left+size]


#arnold transform

def arnold_transform(image,iterations=1):
    if image.shape[0] != image.shape[1]:
        '''
        print("WARNING: Not a sqaure image. Has been croppped.")
        image = make_square(image)
        '''
        raise ValueError("Arnold Transform requires a square image")

    n = image.shape[0]
    transformed_image = np.zeros_like(image)
    for _ in range(iterations):
        for x in range(n):
            for y in range(n):
                new_x = (2*x + y) % n
                new_y = (x + y) % n
                transformed_image[new_x,new_y] = image[x,y]
        image = transformed_image.copy()
    return transformed_image


def covert_to_binary(image_gray,threshold=128):
    #converts a grayscale image to a binary image
    _, binary_img = cv2.threshold(image_gray, 128, 255, cv2.THRESH_BINARY)
    
    return binary_img
'''
def covert_to_binary(image_gray,threshold=128):
    #converts a grayscale image to a binary image

    return (image_gray>threshold).astype(np.uint8)
'''
def block_dct2d(image, block_size=8):
    h, w = image.shape
    dct_coeffs = np.zeros_like(image, dtype=np.float32)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:min(i+block_size, h), j:min(j+block_size, w)]
            padded_block = np.zeros((block_size, block_size), dtype=np.float32)
            padded_block[:block.shape[0], :block.shape[1]] = block
            dct_block = dct(dct(padded_block.T, norm='ortho').T, norm='ortho')
            dct_coeffs[i:min(i+block_size, h), j:min(j+block_size, w)] = dct_block[:block.shape[0], :block.shape[1]]
    return dct_coeffs


def block_idct2d(dct_coeffs, block_size=8):
    h, w = dct_coeffs.shape
    image_reconstructed = np.zeros_like(dct_coeffs, dtype=np.float32)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = dct_coeffs[i:min(i+block_size, h), j:min(j+block_size, w)]
            padded_block = np.zeros((block_size, block_size), dtype=np.float32)
            padded_block[:block.shape[0], :block.shape[1]] = block
            idct_block = idct(idct(padded_block.T, norm='ortho').T, norm='ortho')
            image_reconstructed[i:min(i+block_size, h), j:min(j+block_size, w)] = idct_block[:block.shape[0], :block.shape[1]]
    return image_reconstructed


def embed_watermark_dwt(dwt_coeffs_original, watermark_sub_block, scaling_factor):
    cA, (cH,cV,cD) = dwt_coeffs_original
    if cA.shape != watermark_sub_block.shape:
        print(f"WARNINIG: DWT cA shape {cA.shape} and watermark sub-block shape {watermark_sub_block.shape} mismatch. Resizing watermak.")
        watermark_sub_block_resized = cv2.resize(watermark_sub_block.astype(np.float32), (cA.shape[1], cA.shape[0]), interpolation=cv2.INTER_LINEAR)
    else:
        watermark_sub_block_resized = watermark_sub_block.astype(np.float32)

    #additive embedding
    watermarked_cA = cA + scaling_factor * watermark_sub_block_resized

    # Reconstruct the DWT coefficients with the watermarked cA
    watermarked_dwt_coeffs = (watermarked_cA, (cH, cV, cD))
    return watermarked_dwt_coeffs

def watermark_embedding_process(cover_image_path,watermark_image_path,scaling_factor,output_path,arnold_iterations=1):
    #load image cover
    I_O = cv2.imread(cover_image_path)
    if I_O is None:
        raise FileNotFoundError(f"Watermark image not found: {watermark_image_path}")

    #load image watermark
    W = cv2.imread(watermark_image_path,cv2.IMREAD_GRAYSCALE)
    if W is None:
        raise FileNotFoundError(f"Watermark image not found: {watermark_image_path}")

    #separating the cover image into R, G and B color components with DCT transforms
    B,G,R = cv2.split(I_O)

    block_size = 8 

    #DCT to the three images
    R_DCT = block_dct2d(R.astype(np.float32),block_size)
    G_DCT = block_dct2d(G.astype(np.float32),block_size)
    B_DCT = block_dct2d(B.astype(np.float32),block_size)

    #DWT to the DCT tranformed color components 
    coeff_R_DCT_DWT = pywt.dwt2(R_DCT,'haar')
    coeff_G_DCT_DWT = pywt.dwt2(G_DCT,'haar')
    coeff_B_DCT_DWT = pywt.dwt2(B_DCT,'haar') 

    w_size = max(W.shape) #Pega a maior dimensão da imagem da marca

    if W.shape[0] != W.shape[1]:
        print(f"WARNING: Watermark is not square ({W.shape}). Resizing to {w_size}")
        W_resized = cv2.resize(W,(w_size,w_size),interpolation=cv2.INTER_AREA)
    else:
        W_resized = W.copy()

    #Salva a imagem da marca binarizada 
    watermarkpath = f"{output_path}/embed_watermark.png"
    os.makedirs(os.path.dirname(watermarkpath), exist_ok=True)
    binary = covert_to_binary(W_resized)
    cv2.imwrite(watermarkpath, binary)

    # Embaralha a imagem da marca
    W_scrambled = arnold_transform(W_resized,arnold_iterations)
    
    #convert the scrambled watermark to binary  watermark
    W_binary = covert_to_binary(W_scrambled)

    h_w, w_w = W_binary.shape

    #dividing the watermark into three parts
    W_binary_R_part = W_binary
    W_binary_G_part = W_binary
    W_binary_B_part = W_binary

    #perform DCT to each watermark block to get W_R_DCT, W_G_DCT and  W_B_DCT
    W_R_DCT = block_dct2d(W_binary_R_part.astype(np.float32),block_size)
    W_G_DCT = block_dct2d(W_binary_G_part.astype(np.float32),block_size)
    W_B_DCT = block_dct2d(W_binary_B_part.astype(np.float32),block_size)

    #embed the W_R_DCT
    h_wr, w_wr = W_R_DCT.shape  #get the height and width of the brand

    cA_R_shape = coeff_R_DCT_DWT[0].shape # get the sub-banda LL
    W_R_DCT_resized = cv2.resize(W_R_DCT, 
    (cA_R_shape[1] * 2, cA_R_shape[0] * 2), 
    interpolation=cv2.INTER_LINEAR)   

    #Defines the center of the watermark image.   
    mid_h, mid_w = W_R_DCT_resized.shape[0] // 2, W_R_DCT_resized.shape[1] // 2

    #Divides the image into four parts
    W_RA = W_R_DCT_resized[0:mid_h, 0:mid_w]
    W_RB = W_R_DCT_resized[0:mid_h, mid_w:]
    W_RC = W_R_DCT_resized[mid_h:, 0:mid_w]
    W_RD = W_R_DCT_resized[mid_h:, mid_w:]

    #Extracts each sub-band from the DWT transform.
    cA_R, (cH_R, cV_R, cD_R) = coeff_R_DCT_DWT 

    # Need to resize watermark parts to match DWT band sizes
    W_RA_resized = cv2.resize(W_RA.astype(np.float32), (cA_R.shape[1], cA_R.shape[0]), interpolation=cv2.INTER_LINEAR)
    W_RB_resized = cv2.resize(W_RB.astype(np.float32), (cH_R.shape[1], cH_R.shape[0]), interpolation=cv2.INTER_LINEAR)
    W_RC_resized = cv2.resize(W_RC.astype(np.float32), (cV_R.shape[1], cV_R.shape[0]), interpolation=cv2.INTER_LINEAR)
    W_RD_resized = cv2.resize(W_RD.astype(np.float32), (cD_R.shape[1], cD_R.shape[0]), interpolation=cv2.INTER_LINEAR)

    #Adds each watermark part into the DWT sub-bands of the R channel. 
    LL_R_W = cA_R + scaling_factor * W_RA_resized
    LH_R_W = cH_R + scaling_factor * W_RB_resized
    HL_R_W = cV_R + scaling_factor * W_RC_resized
    HH_R_W = cD_R + scaling_factor * W_RD_resized

    watermarked_coeffs_R_DCT_DWT = (LL_R_W, (LH_R_W, HL_R_W, HH_R_W))

    #IDWT to watermarked DWT bands
    I_R_DCT_watermarked = pywt.idwt2(watermarked_coeffs_R_DCT_DWT, 'haar')
    I_R_DCT_watermarked = I_R_DCT_watermarked[:R_DCT.shape[0], :R_DCT.shape[1]] # Ajustando o tamanho da imagem marcada para mesmo tamanho da R_DCT

    #IDCT to obtain the watermarked 
    I_R_W = block_idct2d(I_R_DCT_watermarked, block_size)
    I_R_W = np.clip(I_R_W, 0, 255).astype(np.uint8) # Clip and convert to uint8


    #embed the W_G_DCT
    h_wr, w_wr = W_G_DCT.shape #dividing into sub-blocks
    cA_G_shape = coeff_G_DCT_DWT[0].shape
    W_G_DCT_resized = cv2.resize(W_G_DCT, (cA_G_shape[1] * 2, cA_R_shape[0] * 2), interpolation=cv2.INTER_LINEAR)
    mid_h, mid_w = W_G_DCT_resized.shape[0] // 2, W_G_DCT_resized.shape[1] // 2
    W_GA = W_G_DCT_resized[0:mid_h, 0:mid_w]
    W_GB = W_G_DCT_resized[0:mid_h, mid_w:]
    W_GC = W_G_DCT_resized[mid_h:, 0:mid_w]
    W_GD = W_G_DCT_resized[mid_h:, mid_w:]

    cA_G, (cH_G, cV_G, cD_G) = coeff_G_DCT_DWT

    # Need to resize watermark parts to match DWT band sizes
    W_GA_resized = cv2.resize(W_GA.astype(np.float32), (cA_G.shape[1], cA_G.shape[0]), interpolation=cv2.INTER_LINEAR)
    W_GB_resized = cv2.resize(W_GB.astype(np.float32), (cH_G.shape[1], cH_G.shape[0]), interpolation=cv2.INTER_LINEAR)
    W_GC_resized = cv2.resize(W_GC.astype(np.float32), (cV_G.shape[1], cV_G.shape[0]), interpolation=cv2.INTER_LINEAR)
    W_GD_resized = cv2.resize(W_GD.astype(np.float32), (cD_G.shape[1], cD_G.shape[0]), interpolation=cv2.INTER_LINEAR)

    LL_G_W = cA_G + scaling_factor * W_GA_resized
    LH_G_W = cH_G + scaling_factor * W_GB_resized
    HL_G_W = cV_G + scaling_factor * W_GC_resized
    HH_G_W = cD_G + scaling_factor * W_GD_resized

    watermarked_coeffs_G_DCT_DWT = (LL_G_W, (LH_G_W, HL_G_W, HH_G_W))

    #IDWT to watermarked DWT bands
    I_G_DCT_watermarked = pywt.idwt2(watermarked_coeffs_G_DCT_DWT, 'haar')
    I_G_DCT_watermarked = I_G_DCT_watermarked[:G_DCT.shape[0], :G_DCT.shape[1]]

    #IDCT to obtain the watermarked 
    I_G_W = block_idct2d(I_G_DCT_watermarked, block_size)
    I_G_W = np.clip(I_G_W, 0, 255).astype(np.uint8) # Clip and convert to uint8

    #embed the W_B_DCT
    h_wr, w_wr = W_B_DCT.shape #dividing into sub-blocks
    cA_B_shape = coeff_B_DCT_DWT[0].shape
    W_B_DCT_resized = cv2.resize(W_B_DCT, (cA_B_shape[1] * 2, cA_B_shape[0] * 2), interpolation=cv2.INTER_LINEAR)
    mid_h, mid_w = W_B_DCT_resized.shape[0] // 2, W_B_DCT_resized.shape[1] // 2
    W_BA = W_B_DCT_resized[0:mid_h, 0:mid_w]
    W_BB = W_B_DCT_resized[0:mid_h, mid_w:]
    W_BC = W_B_DCT_resized[mid_h:, 0:mid_w]
    W_BD = W_B_DCT_resized[mid_h:, mid_w:]

    cA_B, (cH_B, cV_B, cD_B) = coeff_B_DCT_DWT

    # Need to resize watermark parts to match DWT band sizes
    W_BA_resized = cv2.resize(W_BA.astype(np.float32), (cA_B.shape[1], cA_B.shape[0]), interpolation=cv2.INTER_LINEAR)
    W_BB_resized = cv2.resize(W_BB.astype(np.float32), (cH_B.shape[1], cH_B.shape[0]), interpolation=cv2.INTER_LINEAR)
    W_BC_resized = cv2.resize(W_BC.astype(np.float32), (cV_B.shape[1], cV_B.shape[0]), interpolation=cv2.INTER_LINEAR)
    W_BD_resized = cv2.resize(W_BD.astype(np.float32), (cD_B.shape[1], cD_B.shape[0]), interpolation=cv2.INTER_LINEAR)

    LL_B_W = cA_B + scaling_factor * W_BA_resized
    LH_B_W = cH_B + scaling_factor * W_BB_resized
    HL_B_W = cV_B + scaling_factor * W_BC_resized
    HH_B_W = cD_B + scaling_factor * W_BD_resized

    watermarked_coeffs_B_DCT_DWT = (LL_B_W, (LH_B_W, HL_B_W, HH_B_W))

    #IDWT to watermarked DWT bands
    I_B_DCT_watermarked = pywt.idwt2(watermarked_coeffs_B_DCT_DWT, 'haar')
    I_B_DCT_watermarked = I_B_DCT_watermarked[:B_DCT.shape[0], :B_DCT.shape[1]]

    #IDCT to obtain the watermarked 
    I_B_W = block_idct2d(I_B_DCT_watermarked, block_size)
    I_B_W = np.clip(I_B_W, 0, 255).astype(np.uint8) # Clip and convert to uint8
    
    I_W = cv2.merge([I_B_W, I_G_W, I_R_W]) # Merge in BGR order for OpenCV display/save

    return I_W


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cover", required=True)
    parser.add_argument("--watermark", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scaling_factor", type=float, required=True)
    parser.add_argument("--arnold_iterations", type=int, required=True)

    args = parser.parse_args()

    cover_image_path = args.cover
    watermark_image_path = args.watermark
    output_path = args.output
    scaling_factor = args.scaling_factor
    arnold_iterations = args.arnold_iterations

    try:
        watermarked_image = watermark_embedding_process(
            cover_image_path,
            watermark_image_path,
            scaling_factor,
            output_path,
            arnold_iterations
        )

        watermarked_path = f"{output_path}/watermarked.png"
        os.makedirs(os.path.dirname(watermarked_path), exist_ok=True)
        cv2.imwrite(watermarked_path, watermarked_image) 
        print(f"Watermark embedding complete. Saved to {watermarked_path}")

        cover = cv2.imread(cover_image_path)
        watermark = cv2.imread(watermark_image_path, cv2.IMREAD_GRAYSCALE)

        psnr_value = cv2.PSNR(cover, watermarked_image)

        cover_gray = cv2.cvtColor(cover, cv2.COLOR_BGR2GRAY)
        watermarked_gray = cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2GRAY)
        ssim_value, _ = ssim(cover_gray, watermarked_gray, full=True)

        print(f"PSNR: {psnr_value:.2f} dB")
        print(f"SSIM: {ssim_value:.4f}")

        cv2.imwrite("output_preview.png", watermarked_image)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
