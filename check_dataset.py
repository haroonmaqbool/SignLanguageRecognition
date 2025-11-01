"""
Dataset Structure Checker
========================

This script helps you verify that your ASL alphabet dataset is properly organized
and ready for preprocessing.

Author: AI Coding Assistant
Date: 2024
"""

import os
import glob

def check_dataset_structure():
    """
    Check if the dataset is properly structured for the preprocessing pipeline.
    """
    print("=" * 60)
    print("ASL Alphabet Dataset Structure Checker")
    print("=" * 60)
    
    # Common dataset directory names
    possible_paths = [
        "archive/asl_alphabet_train/asl_alphabet_train",  # Your dataset location
        "asl_alphabet_train",
        "asl_alphabet_train/asl_alphabet_train", 
        "dataset/asl_alphabet_train",
        "data/asl_alphabet_train",
        "ASL_Alphabet_Dataset/asl_alphabet_train"
    ]
    
    print("Searching for dataset in common locations...")
    
    train_path = None
    for path in possible_paths:
        if os.path.exists(path):
            train_path = path
            print(f"[SUCCESS] Found dataset at: {path}")
            break
    
    if train_path is None:
        print("[ERROR] Dataset not found in common locations!")
        print("\nPlease check if your dataset is in one of these locations:")
        for path in possible_paths:
            print(f"   - {path}")
        print("\nIf your dataset is elsewhere, please move it to one of these locations")
        print("   or update the 'train_path' variable in preprocessing.py")
        return False
    
    # Check dataset structure
    print(f"\nAnalyzing dataset structure at: {train_path}")
    
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    found_letters = []
    missing_letters = []
    total_images = 0
    
    for letter in alphabet:
        letter_path = os.path.join(train_path, letter)
        if os.path.exists(letter_path):
            # Count images in this letter folder
            image_files = glob.glob(os.path.join(letter_path, "*.jpg")) + \
                         glob.glob(os.path.join(letter_path, "*.jpeg")) + \
                         glob.glob(os.path.join(letter_path, "*.png"))
            
            image_count = len(image_files)
            total_images += image_count
            found_letters.append(letter)
            
            print(f"   [OK] {letter}: {image_count} images")
        else:
            missing_letters.append(letter)
            print(f"   [MISSING] {letter}: Missing folder")
    
    print(f"\nDataset Summary:")
    print(f"   - Found letters: {len(found_letters)}/26")
    print(f"   - Missing letters: {len(missing_letters)}")
    print(f"   - Total images: {total_images}")
    
    if missing_letters:
        print(f"   - Missing letters: {', '.join(missing_letters)}")
    
    # Check if dataset is ready for preprocessing
    if len(found_letters) >= 20:  # At least 20 letters
        print(f"\n[SUCCESS] Dataset is ready for preprocessing!")
        print(f"   - You have {len(found_letters)} letters which is sufficient")
        print(f"   - Total of {total_images} images available")
        print(f"\nNext steps:")
        print(f"   1. Run: python preprocessing.py")
        print(f"   2. Run: python train_model.py")
        print(f"   3. Run: python evaluate_model.py")
        return True
    else:
        print(f"\n[WARNING] Dataset may not be complete enough for training")
        print(f"   - You only have {len(found_letters)} letters")
        print(f"   - Consider downloading a more complete dataset")
        return False

def suggest_dataset_organization():
    """
    Provide suggestions for organizing the dataset.
    """
    print(f"\nExpected Dataset Structure:")
    print(f"   dataset_folder/")
    print(f"   +-- asl_alphabet_train/")
    print(f"       +-- A/")
    print(f"       |   +-- A1.jpg")
    print(f"       |   +-- A2.jpg")
    print(f"       |   +-- ...")
    print(f"       +-- B/")
    print(f"       |   +-- B1.jpg")
    print(f"       |   +-- B2.jpg")
    print(f"       |   +-- ...")
    print(f"       +-- C/")
    print(f"       +-- ... (up to Z)")
    print(f"       +-- ...")
    
    print(f"\nIf your dataset structure is different:")
    print(f"   1. Rename folders to match A, B, C, ..., Z")
    print(f"   2. Ensure images are in JPG, JPEG, or PNG format")
    print(f"   3. Place the main folder in your project directory")

if __name__ == "__main__":
    try:
        is_ready = check_dataset_structure()
        if not is_ready:
            suggest_dataset_organization()
    except Exception as e:
        print(f"[ERROR] Error checking dataset: {e}")
        suggest_dataset_organization()
