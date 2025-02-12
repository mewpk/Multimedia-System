import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct

# ฟังก์ชันคำนวณ 2D DCT แบบ 'ortho' สำหรับบล็อก 8x8
def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

# ฟังก์ชันคำนวณ 2D inverse DCT แบบ 'ortho' สำหรับบล็อก 8x8
def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

# ตาราง Quantization มาตรฐานสำหรับ Luminance (ภาพสีเทา) ตามมาตรฐาน JPEG
quant_table_std = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float32)

def get_quantization_table(quality):
    """
    ปรับตาราง Quantization ตามค่า quality ที่รับเข้ามา โดยใช้สูตร:
      - ถ้า quality < 50  => scale = 5000 / quality
      - ถ้า quality >= 50 => scale = 200 - 2 * quality
    จากนั้นคำนวณ quant_table ใหม่โดยใช้สูตร:
      quant_table = floor((standard_table * scale + 50) / 100)
    และกำหนดค่าให้มีค่าไม่น้อยกว่า 1
    """
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality
    quant_table = np.floor((quant_table_std * scale + 50) / 100)
    quant_table[quant_table < 1] = 1
    return quant_table

def compress_image(image, quality):
    """
    บีบอัดภาพ grayscale โดยใช้กระบวนการ DCT แบบแบ่งบล็อกขนาด 8x8
      - image: อาเรย์ 2 มิติของภาพ (ค่า 0-255)
      - quality: ค่า quality (10, 20, …, 90)
    คืนค่า: ภาพที่บีบอัดแล้ว (uint8)
    """
    h, w = image.shape

    # ตรวจสอบขนาดของภาพ หากขนาดไม่ใช่พหุของ 8 ให้ pad ด้วยค่า 0
    pad_h = (8 - h % 8) if h % 8 != 0 else 0
    pad_w = (8 - w % 8) if w % 8 != 0 else 0
    padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    h_pad, w_pad = padded.shape

    # คำนวณตาราง Quantization ตาม quality ที่กำหนด
    quant_table = get_quantization_table(quality)

    # สร้างอาเรย์สำหรับเก็บภาพที่บีบอัดแล้ว
    compressed = np.zeros_like(padded, dtype=np.float32)

    # ประมวลผลทีละบล็อกขนาด 8x8
    for i in range(0, h_pad, 8):
        for j in range(0, w_pad, 8):
            block = padded[i:i+8, j:j+8].astype(np.float32)
            # ย้ายค่าให้มีค่าเฉลี่ยศูนย์ (subtract 128) ตามมาตรฐาน JPEG
            block = block - 128
            # คำนวณ DCT ของบล็อก
            dct_block = dct2(block)
            # Quantization: หารด้วยตาราง quantizationแล้วปัดเศษ
            quantized = np.round(dct_block / quant_table)
            # Dequantization: คูณกลับด้วยตาราง quantization
            dequantized = quantized * quant_table
            # คำนวณ inverse DCT เพื่อคืนค่าบล็อกในเชิงพื้นที่
            idct_block = idct2(dequantized)
            # คืนค่าบล็อกด้วยการบวก 128 และจำกัดค่าให้อยู่ในช่วง 0-255
            block_recon = idct_block + 128
            block_recon = np.clip(block_recon, 0, 255)
            compressed[i:i+8, j:j+8] = block_recon

    # ตัดส่วน padding ออกให้ได้ขนาดเดิม
    compressed = compressed[:h, :w]
    return compressed.astype(np.uint8)

def main():
    # เปลี่ยน filepath ให้เป็นภาพของนักศึกษาเอง
    filepath = './65010731.jpg'
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("ไม่สามารถโหลดภาพได้ กรุณาตรวจสอบ filepath")
        return

    # กำหนดระดับ quality ที่ต้องการทดสอบ (10, 20, …, 90)
    quality_values = list(range(10, 100, 10))
    compressed_images = []

    for quality in quality_values:
        comp_img = compress_image(image, quality)
        compressed_images.append(comp_img)

    # แสดงผลลัพธ์: ภาพต้นฉบับและภาพที่บีบอัดสำหรับแต่ละค่า quality
    num_plots = len(quality_values) + 1
    plt.figure(figsize=(15, 8))

    # แสดงภาพต้นฉบับ
    plt.subplot(2, 5, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    # แสดงภาพที่บีบอัดตามระดับ quality ต่าง ๆ
    for idx, (q, comp_img) in enumerate(zip(quality_values, compressed_images)):
        plt.subplot(2, 5, idx + 2)
        plt.imshow(comp_img, cmap='gray')
        plt.title(f"Quality = {q}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
