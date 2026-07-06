from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import traceback
import os

from siamese_network import SiameseNetwork

# -----------------------------
# App Setup
# -----------------------------
app = Flask(__name__)
CORS(app)

# -----------------------------
# Load Model
# -----------------------------
print("Loading model...")

model = SiameseNetwork()

state_dict = torch.load(
    "models/siamese_bestgpds.pth",
    map_location=torch.device("cpu")
)

model_dict = model.state_dict()

filtered_dict = {
    k: v
    for k, v in state_dict.items()
    if k in model_dict and v.shape == model_dict[k].shape
}

model_dict.update(filtered_dict)
model.load_state_dict(model_dict)

model.eval()

print("✅ Model loaded successfully")

# -----------------------------
# Image Preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((120, 160)),
    transforms.ToTensor()
])

# -----------------------------
# Embedding Function
# -----------------------------
def get_embedding(img):
    with torch.no_grad():
        x = model.conv(img.unsqueeze(0))
        x = x.view(x.size(0), -1)
        emb = F.normalize(x.squeeze(0), p=2, dim=0)
    return emb

# -----------------------------
# Home Route
# -----------------------------
@app.route("/")
def home():
    return jsonify({
        "status": "online",
        "project": "Few-Shot Offline Signature Verification API",
        "message": "API is running successfully.",
        "endpoint": "/verify"
    })

# -----------------------------
# Verify Route
# -----------------------------
@app.route("/verify", methods=["POST"])
def verify():

    print("\n==============================")
    print("VERIFY REQUEST RECEIVED")
    print("==============================")

    try:

        print("Step 1 : Reading uploaded files")

        ref_files = request.files.getlist("reference")
        query_file = request.files["query"]

        print("Reference Files :", len(ref_files))
        print("Query File Received")

        if len(ref_files) != 5:
            return jsonify({
                "error": "Upload exactly 5 reference signatures"
            }), 400

        print("Step 2 : Creating reference embeddings")

        embeddings = []

        for idx, file in enumerate(ref_files):

            file.stream.seek(0)

            img = Image.open(file).convert("L")
            img = transform(img)

            embeddings.append(get_embedding(img))

            print(f"Reference {idx+1} embedding created")

        ref_embeddings = torch.stack(embeddings)

        print("All reference embeddings created")

        print("Step 3 : Computing prototype")

        prototype = ref_embeddings.mean(dim=0)

        print("Prototype computed")

        print("Step 4 : Computing adaptive threshold")

        intra_distances = []

        for i in range(len(ref_embeddings)):
            for j in range(i + 1, len(ref_embeddings)):
                intra_distances.append(
                    torch.norm(
                        ref_embeddings[i] - ref_embeddings[j]
                    ).item()
                )

        avg_intra_dist = np.median(intra_distances)

        threshold = avg_intra_dist * 1.1

        print("Threshold :", threshold)

        print("Step 5 : Processing query image")

        query_file.stream.seek(0)

        query_img = Image.open(query_file).convert("L")
        query_img = transform(query_img)

        query_emb = get_embedding(query_img)

        print("Query embedding created")

        query_dist = torch.norm(query_emb - prototype).item()

        pairwise = np.array([
            torch.norm(query_emb - ref).item()
            for ref in ref_embeddings
        ])

        mean_pairwise = np.mean(pairwise)
        std_pairwise = np.std(pairwise)
        max_pairwise = np.max(pairwise)

        print(f"Mean : {mean_pairwise:.4f}")
        print(f"Std  : {std_pairwise:.4f}")
        print(f"Max  : {max_pairwise:.4f}")

        print("Step 6 : Decision")

        is_borderline = False

        if std_pairwise < 0.015:
            result = "Forged"

        elif (max_pairwise - mean_pairwise) < 0.02:
            result = "Forged"

        elif mean_pairwise > threshold:
            result = "Forged"

        else:

            lower = threshold * 0.9

            if mean_pairwise < lower:
                result = "Genuine"

            else:

                is_borderline = True

                cond1 = query_dist < threshold * 1.1
                cond2 = std_pairwise > 0.01
                cond3 = max_pairwise < threshold * 1.2

                score = sum([cond1, cond2, cond3])

                result = "Genuine" if score >= 2 else "Forged"

        print("Result :", result)

        print("Returning response")

        return jsonify({
            "result": result,
            "distance": round(query_dist, 4),
            "threshold": round(threshold, 4),
            "is_borderline": is_borderline
        })

    except Exception as e:

        print("\n❌ EXCEPTION OCCURRED")
        traceback.print_exc()

        return jsonify({
            "error": str(e)
        }), 500


# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)