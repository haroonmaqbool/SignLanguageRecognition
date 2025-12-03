"""
Dataset Structure Checker
========================
This script helps you verify that your ASL alphabet dataset is properly organized
and ready for preprocessing.
"""

import os
from pathlib import Path

def check_dataset_structure():
    """Check if the dataset is properly structured for the preprocessing pipeline."""
    print("=" * 60)
    print("ASL Alphabet Dataset Structure Checker")
    print("=" * 60)
    
    # Common dataset directory names
    possible_paths = [
        "archive/asl_alphabet_train/asl_alphabet_train",
        "asl_alphabet_train",
        "asl_alphabet_train/asl_alphabet_train", 
        "dataset/asl_alphabet_train",
        "data/asl_alphabet_train",
        "ASL_Alphabet_Dataset/asl_alphabet_train"
    ]
    
    print("Searching for dataset in common locations...")
    
    # Find dataset path
    train_path = next((p for p in possible_paths if os.path.exists(p)), None)
    
    if not train_path:
        print("[ERROR] Dataset not found in common locations!")
        print("\nPlease check if your dataset is in one of these locations:")
        print("\n".join(f"   - {p}" for p in possible_paths))
        print("\nIf your dataset is elsewhere, please move it to one of these locations")
        print("   or update the 'train_path' variable in preprocessing.py")
        return False
    
    print(f"[SUCCESS] Found dataset at: {train_path}")
    print(f"\nAnalyzing dataset structure at: {train_path}")
    
    found_letters = []
    missing_letters = []
    found_special = []
    missing_special = []
    total_images = 0
    
    # Check A-Z letters
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        letter_path = Path(train_path) / letter
        
        if letter_path.exists():
            # Count images using pathlib
            image_count = sum(1 for _ in letter_path.glob("*.[jJ][pP][gG]")) + \
                         sum(1 for _ in letter_path.glob("*.[jJ][pP][eE][gG]")) + \
                         sum(1 for _ in letter_path.glob("*.[pP][nN][gG]"))
            
            total_images += image_count
            found_letters.append(letter)
            print(f"   [OK] {letter}: {image_count} images")
        else:
            missing_letters.append(letter)
            print(f"   [MISSING] {letter}: Missing folder")
    
    # Check special classes: space, del, nothing
    print(f"\nChecking special classes (space, del, nothing)...")
    special_classes = ["space", "del", "nothing"]
    for special in special_classes:
        special_path = Path(train_path) / special
        
        if special_path.exists():
            image_count = sum(1 for _ in special_path.glob("*.[jJ][pP][gG]")) + \
                         sum(1 for _ in special_path.glob("*.[jJ][pP][eE][gG]")) + \
                         sum(1 for _ in special_path.glob("*.[pP][nN][gG]"))
            
            total_images += image_count
            found_special.append(special)
            print(f"   [OK] {special}: {image_count} images")
        else:
            missing_special.append(special)
            print(f"   [MISSING] {special}: Missing folder")
    
    # Print summary
    print(f"\nDataset Summary:")
    print(f"   - Found letters (A-Z): {len(found_letters)}/26")
    if missing_letters:
        print(f"   - Missing letters: {', '.join(missing_letters)}")
    print(f"   - Found special classes: {len(found_special)}/3 (space, del, nothing)")
    if missing_special:
        print(f"   - Missing special classes: {', '.join(missing_special)}")
    print(f"   - Total classes found: {len(found_letters) + len(found_special)}/29")
    print(f"   - Total images: {total_images}")
    
    if missing_letters:
        print(f"   - Missing letters: {', '.join(missing_letters)}")
    
    # Check if dataset is ready
    total_classes = len(found_letters) + len(found_special)
    is_ready = len(found_letters) >= 20  
    
    if is_ready:
        print(f"\n[SUCCESS] Dataset is ready for preprocessing!")
        print(f"   - You have {len(found_letters)} letters (A-Z) which is sufficient")
        if found_special:
            print(f"   - You have {len(found_special)} special classes: {', '.join(found_special)}")
        print(f"   - Total of {total_images} images available")
        print(f"   - Total classes: {total_classes} (will train on {total_classes} classes)")
        print(f"\nNext steps:")
        print(f"   1. Run: python preprocessing.py")
        print(f"   2. Run: python train_model.py")
        print(f"   3. Run: python evaluate_model.py")
    else:
        print(f"\n[WARNING] Dataset may not be complete enough for training")
        print(f"   - You only have {len(found_letters)} letters")
        print(f"   - Consider downloading a more complete dataset")
    
    return is_ready

def suggest_dataset_organization():
    """Provide suggestions for organizing the dataset."""
    structure = """
Expected Dataset Structure:
   dataset_folder/
   +-- asl_alphabet_train/
       +-- A/
       |   +-- A1.jpg
       |   +-- A2.jpg
       |   +-- ...
       +-- B/
       |   +-- B1.jpg
       |   +-- B2.jpg
       |   +-- ...
       +-- C/
       +-- ... (up to Z)
       +-- space/
       |   +-- space1.jpg
       |   +-- space2.jpg
       |   +-- ...
       +-- del/
       |   +-- del1.jpg
       |   +-- del2.jpg
       |   +-- ...
       +-- nothing/
       |   +-- nothing1.jpg
       |   +-- nothing2.jpg
       |   +-- ...

If your dataset structure is different:
   1. Rename folders to match A, B, C, ..., Z
   2. Ensure images are in JPG, JPEG, or PNG format
   3. Place the main folder in your project directory
"""
    print(structure)

if __name__ == "__main__":
    try:
        if not check_dataset_structure():
            suggest_dataset_organization()
    except Exception as e:
        print(f"[ERROR] Error checking dataset: {e}")
        suggest_dataset_organization()