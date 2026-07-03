import os
import shutil

GPDS_ROOT = r"C:\Users\DELL LATITUDE 5420\OneDrive\Documents\GPDSOriginal"

GENUINE_SRC = os.path.join(GPDS_ROOT, "GPDS300genuine")
FORGED_SRC = os.path.join(GPDS_ROOT, "GPDS300forged")

TRAIN_ROOT = "dataset/GPDS/train/genuine"
TEST_GENUINE_ROOT = "dataset/GPDS/test/genuine"
TEST_FORGED_ROOT = "dataset/GPDS/test/forged"

os.makedirs(TRAIN_ROOT, exist_ok=True)
os.makedirs(TEST_GENUINE_ROOT, exist_ok=True)
os.makedirs(TEST_FORGED_ROOT, exist_ok=True)

TRAIN_SAMPLES_PER_USER = 5   # 🔥 CHANGED FROM 5 TO 10

# ---------- SPLIT GENUINE ----------
for user in os.listdir(GENUINE_SRC):
    user_path = os.path.join(GENUINE_SRC, user)

    if not os.path.isdir(user_path):
        continue

    train_user_dir = os.path.join(TRAIN_ROOT, user)
    test_user_dir = os.path.join(TEST_GENUINE_ROOT, user)

    os.makedirs(train_user_dir, exist_ok=True)
    os.makedirs(test_user_dir, exist_ok=True)

    images = sorted(os.listdir(user_path))

    train_imgs = images[:TRAIN_SAMPLES_PER_USER]
    test_imgs = images[TRAIN_SAMPLES_PER_USER:]

    for img in train_imgs:
        shutil.copy(os.path.join(user_path, img), train_user_dir)

    for img in test_imgs:
        shutil.copy(os.path.join(user_path, img), test_user_dir)

# ---------- SPLIT FORGED ----------
for user in os.listdir(FORGED_SRC):
    user_path = os.path.join(FORGED_SRC, user)

    if not os.path.isdir(user_path):
        continue

    test_user_dir = os.path.join(TEST_FORGED_ROOT, user)
    os.makedirs(test_user_dir, exist_ok=True)

    for img in os.listdir(user_path):
        shutil.copy(os.path.join(user_path, img), test_user_dir)

print("✅ GPDS-300 dataset split with 10-shot training completed.")
