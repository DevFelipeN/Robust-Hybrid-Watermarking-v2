import cv2
import numpy as np

def add_gaussian_noise(image, mean=0, std=10):
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy = cv2.add(image.astype(np.float32), noise)
    return np.clip(noisy, 0, 255).astype(np.uint8)

def apply_filter(image, kernel_size=3):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def jpeg_compression(image, quality=50):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', image, encode_param)
    return cv2.imdecode(encimg, 1)

def rotate_image(image, angle=30):
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def translate_image(image, tx=20, ty=20):
    h, w = image.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def scale_image(image, fx=1.2, fy=1.2):
    h, w = image.shape[:2]
    scaled = cv2.resize(image, None, fx=fx, fy=fy)
    # center crop to original size
    center_x, center_y = scaled.shape[1]//2, scaled.shape[0]//2
    start_x, start_y = center_x - w//2, center_y - h//2
    return scaled[start_y:start_y+h, start_x:start_x+w]

def shear_image(image, shear_factor=0.2):
    h, w = image.shape[:2]
    M = np.float32([[1, shear_factor, 0],
                    [shear_factor, 1, 0]])
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def affine_transform(image):
    h, w = image.shape[:2]
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[60, 70], [190, 60], [70, 210]])
    M = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def run_all_attacks(image_path):
    img = cv2.imread(image_path)
    out = {}

    out['filtered']     = apply_filter(img)
    out['noisy']        = add_gaussian_noise(img)
    out['jpeg']         = jpeg_compression(img, quality=30)
    out['rotated']      = rotate_image(img, angle=45)
    out['translated']   = translate_image(img, tx=40, ty=30)
    out['scaled']       = scale_image(img, fx=1.4, fy=1.4)
    out['sheared']      = shear_image(img, shear_factor=0.3)
    out['affine']       = affine_transform(img)

    return out

if __name__ == "__main__":
    import os

    image_path = "/home/chinasa/python_projects/watermark/output/watermarked_image.png"  # change as needed
    attacks = run_all_attacks(image_path)

    os.makedirs("outputs", exist_ok=True)
    for name, img in attacks.items():
        if name == "jpeg":
            cv2.imwrite(f"outputs/{name}.jpg", img)
        else:
            cv2.imwrite(f"outputs/{name}.png", img)


    print("Saved all attacked images in the 'outputs/' folder.")


