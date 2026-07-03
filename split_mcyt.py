import os
import shutil

MCYT_ROOT = r"C:\Users\DELL LATITUDE 5420\Downloads\MCYT"

GENUINE_SRC = os.path.join(MCYT_ROOT, "MCYT_Signature_100originalConvertedGenuine")
FORGED_SRC = os.path.join(MCYT_ROOT, "MCYT_Signature_100originalConvertedForged")

TRAIN_GENUINE = "dataset/MCYT/train/genuine"
TEST_GENUINE = "dataset/MCYT/test/genuine"
TEST_FORGED = "dataset/MCYT/test/forged"

os.makedirs(TRAIN_GENUINE, exist_ok=True)
os.makedirs(TEST_GENUINE, exist_ok=True)
os.makedirs(TEST_FORGED, exist_ok=True)

for user in os.listdir(GENUINE_SRC):

    genuine_user_path = os.path.join(GENUINE_SRC, user)
    forged_user_path = os.path.join(FORGED_SRC, user)

    if not os.path.isdir(genuine_user_path):
        continue

    train_user_dir = os.path.join(TRAIN_GENUINE, user)
    test_user_dir = os.path.join(TEST_GENUINE, user)
    test_forged_dir = os.path.join(TEST_FORGED, user)

    os.makedirs(train_user_dir, exist_ok=True)
    os.makedirs(test_user_dir, exist_ok=True)
    os.makedirs(test_forged_dir, exist_ok=True)

    genuine_images = sorted(os.listdir(genuine_user_path))

    # 5-shot training
    train_imgs = genuine_images[:5]
    test_imgs = genuine_images[5:]

    for img in train_imgs:
        shutil.copy(
            os.path.join(genuine_user_path, img),
            train_user_dir
        )

    for img in test_imgs:
        shutil.copy(
            os.path.join(genuine_user_path, img),
            test_user_dir
        )

    # Copy all forged to test
    if os.path.isdir(forged_user_path):
        for img in os.listdir(forged_user_path):
            shutil.copy(
                os.path.join(forged_user_path, img),
                test_forged_dir
            )

print("✅ MCYT 5-shot split completed.")