import os
import shutil

CEDAR_ROOT = r"C:\Users\DELL LATITUDE 5420\Downloads\CEDAR\signatures"

GENUINE_SRC = os.path.join(CEDAR_ROOT, "full_org")
FORGED_SRC = os.path.join(CEDAR_ROOT, "full_forg")

TRAIN_GENUINE = "dataset/CEDAR/train/genuine"
TRAIN_FORGED = "dataset/CEDAR/train/forged"
TEST_GENUINE = "dataset/CEDAR/test/genuine"
TEST_FORGED = "dataset/CEDAR/test/forged"

os.makedirs(TRAIN_GENUINE, exist_ok=True)
os.makedirs(TRAIN_FORGED, exist_ok=True)
os.makedirs(TEST_GENUINE, exist_ok=True)
os.makedirs(TEST_FORGED, exist_ok=True)

# -------- GROUP GENUINE --------
genuine_users = {}
for file in os.listdir(GENUINE_SRC):
    if file.lower().endswith((".png", ".jpg", ".jpeg")):
        user_id = file.split("_")[1]
        genuine_users.setdefault(user_id, []).append(file)

# -------- GROUP FORGED --------
forged_users = {}
for file in os.listdir(FORGED_SRC):
    if file.lower().endswith((".png", ".jpg", ".jpeg")):
        user_id = file.split("_")[1]
        forged_users.setdefault(user_id, []).append(file)

# -------- SPLIT --------
for user in genuine_users.keys():

    train_g_user = os.path.join(TRAIN_GENUINE, user)
    train_f_user = os.path.join(TRAIN_FORGED, user)
    test_g_user = os.path.join(TEST_GENUINE, user)
    test_f_user = os.path.join(TEST_FORGED, user)

    os.makedirs(train_g_user, exist_ok=True)
    os.makedirs(train_f_user, exist_ok=True)
    os.makedirs(test_g_user, exist_ok=True)
    os.makedirs(test_f_user, exist_ok=True)

    genuine_files = sorted(genuine_users[user])
    forged_files = sorted(forged_users.get(user, []))

    # 5-shot genuine
    for f in genuine_files[:5]:
        shutil.copy(os.path.join(GENUINE_SRC, f), train_g_user)

    for f in genuine_files[5:]:
        shutil.copy(os.path.join(GENUINE_SRC, f), test_g_user)

    # 5 forged for training
    for f in forged_files[:5]:
        shutil.copy(os.path.join(FORGED_SRC, f), train_f_user)

    for f in forged_files[5:]:
        shutil.copy(os.path.join(FORGED_SRC, f), test_f_user)

print("✅ Proper CEDAR 5-shot split completed.")
