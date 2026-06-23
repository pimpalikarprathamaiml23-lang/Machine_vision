"""
================================================================================
LAB PRACTICAL REPORT / PROJECT DOCUMENTATION
================================================================================
PROJECT TITLE: Nighttime CCTV Surveillance Image Enhancement System
CLIENT: Security Agency
OBJECTIVE: Develop an automated image processing pipeline to improve visibility,
           reduce low-light sensor grain, and maximize detail extraction from
           poor-quality nighttime surveillance footage.

CORE METHODOLOGIES IMPLEMENTED:
1. NOISE REMOVAL & FILTERING: 
   - Bilateral Filtering (cv2.bilateralFilter) is chosen over Gaussian blur. It
     smooths out high-frequency sensor grain noise in dark areas while actively
     preserving sharp structural edges (crucial for facial/license plate details).
     
2. HISTOGRAM EQUALIZATION:
   - Contrast Limited Adaptive Histogram Equalization (CLAHE) is used instead of
     Global Histogram Equalization. CLAHE divides images into contextual tiles 
     (8x8 grid), enhancing localized details without amplifying ambient glare 
     or over-exposing streetlights.

3. BRIGHTNESS ENHANCEMENT:
   - Gamma Correction via Look-Up Tables (LUT). Non-linearly maps intensity values
     to brighten up midtones and reveal objects hidden in deep shadows without 
     washing out dark gray tones into solid white.
================================================================================
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def process_and_enhance(image_path):
    """
    Core Pipeline Function: Implements the sequential image enhancement steps.
    """
    # Load image in GrayScale (Nighttime CCTV is usually monochrome or heavily discolored)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    # -------------------------------------------------------------------------
    # STEP 1: NOISE REMOVAL & FILTERING (Edge-Preserving Smoothing)
    # -------------------------------------------------------------------------
    # d=9: Pixel neighborhood diameter
    # sigmaColor/Space=75: Higher value smooths larger areas of similar dark colors
    denoised = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    # -------------------------------------------------------------------------
    # STEP 2: HISTOGRAM EQUALIZATION (Localized Contrast Boost)
    # -------------------------------------------------------------------------
    # clipLimit=3.0: Prevents the amplification of residual noise in dark patches
    # tileGridSize=(8, 8): Breaks image into 64 zones to process contrast locally
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    equalized = clahe.apply(denoised)

    # -------------------------------------------------------------------------
    # STEP 3: BRIGHTNESS ENHANCEMENT (Non-linear Gamma Mapping)
    # -------------------------------------------------------------------------
    # gamma = 1.2: Expands the dynamic range of lower-end shadow values
    gamma = 1.2
    invGamma = 1.0 / gamma
    # Build a lookup table mapping input pixels [0-255] to corrected brightened values
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    enhanced = cv2.LUT(equalized, table)
    
    return img, enhanced

def auto_enhance_downloads():
    """
    Automation & Presentation Layer: Scans directory, processes inputs, and renders charts.
    """
    downloads_dir = r"C:\Users\STUDENTS\Downloads"
    
    # Automate searching for target forensic artifacts (.jpg / .jpeg format)
    jpg_files = glob.glob(os.path.join(downloads_dir, "*.jpg")) + glob.glob(os.path.join(downloads_dir, "*.jpeg"))
    
    # Remove any previously enhanced images from our source list to avoid self-processing
    jpg_files = [f for f in jpg_files if "_enhanced" not in f]

    if len(jpg_files) < 2:
        print(f"System Error: Insufficient material. Found only {len(jpg_files)} raw JPG file(s).")
        print("Files detected:", [os.path.basename(f) for f in jpg_files])
        return

    # Select the first two surveillance frames found
    path1 = jpg_files[0]
    path2 = jpg_files[1]
    
    print(f"[STATUS] Processing Evidence 1: {os.path.basename(path1)}")
    print(f"[STATUS] Processing Evidence 2: {os.path.basename(path2)}")

    # Execute processing pipeline
    result1 = process_and_enhance(path1)
    result2 = process_and_enhance(path2)

    if result1 is None or result2 is None:
        print("Error: Encountered an issue reading one of the found files.")
        return

    orig1, enh1 = result1
    orig2, enh2 = result2

    # -------------------------------------------------------------------------
    # STEP 4: ANALYSIS AND VISUALIZATION (Comparative Layout)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(12, 10))

    # Row 1: Evidence Frame 1 Comparison
    plt.subplot(2, 2, 1)
    plt.title(f"Original 1: {os.path.basename(path1)}")
    plt.imshow(orig1, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title("Enhanced Forensic Result 1")
    plt.imshow(enh1, cmap='gray')
    plt.axis('off')

    # Row 2: Evidence Frame 2 Comparison
    plt.subplot(2, 2, 3)
    plt.title(f"Original 2: {os.path.basename(path2)}")
    plt.imshow(orig2, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title("Enhanced Forensic Result 2")
    plt.imshow(enh2, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Save output artifacts to disk for security records
    out1 = path1.replace('.jpg', '_enhanced.jpg').replace('.jpeg', '_enhanced.jpeg')
    out2 = path2.replace('.jpg', '_enhanced.jpg').replace('.jpeg', '_enhanced.jpeg')
    cv2.imwrite(out1, enh1)
    cv2.imwrite(out2, enh2)
    print(f"[SUCCESS] Saved output logs:\n -> {out1}\n -> {out2}")

# Execute the security application
auto_enhance_downloads()