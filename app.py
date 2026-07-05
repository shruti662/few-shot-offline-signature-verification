from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

from siamese_network import SiameseNetwork

# -----------------------------
# App Setup
# -----------------------------
app = Flask(__name__)
CORS(app)

# -----------------------------
# Load Model
# -----------------------------
model = SiameseNetwork()

state_dict = torch.load("models/siamese_bestgpds.pth", map_location=torch.device('cpu'))

model_dict = model.state_dict()

filtered_dict = {
    k: v for k, v in state_dict.items()
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
# API Route
# -----------------------------
@app.route("/verify", methods=["POST"])
def verify():
    print("========== VERIFY REQUEST RECEIVED ==========")
    try:
        print("Entered try block")
        ref_files = request.files.getlist("reference")
        query_file = request.files["query"]

        if len(ref_files) != 5:
            return jsonify({"error": "Upload exactly 5 reference signatures"}), 400

        # -------------------------
        # Reference embeddings
        # -------------------------
        embeddings = []
        for file in ref_files:
            img = Image.open(file).convert("L")
            img = transform(img)
            embeddings.append(get_embedding(img))

        ref_embeddings = torch.stack(embeddings)

        # -------------------------
        # Prototype
        # -------------------------
        prototype = ref_embeddings.mean(dim=0)

        # -------------------------
        # Intra-class distance
        # -------------------------
        intra_distances = []

        for i in range(len(ref_embeddings)):
            for j in range(i + 1, len(ref_embeddings)):
                intra_distances.append(
                    torch.norm(ref_embeddings[i] - ref_embeddings[j]).item()
                )

        avg_intra_dist = np.median(intra_distances)

        # 🔥 Adaptive threshold
        threshold = avg_intra_dist * 1.1

        # -------------------------
        # Query embedding
        # -------------------------
        query_img = Image.open(query_file).convert("L")
        query_img = transform(query_img)
        query_emb = get_embedding(query_img)

        query_dist = torch.norm(query_emb - prototype).item()

        pairwise = np.array([
            torch.norm(query_emb - ref).item()
            for ref in ref_embeddings
        ])

        mean_pairwise = np.mean(pairwise)
        std_pairwise = np.std(pairwise)
        max_pairwise = np.max(pairwise)

        print(f"Mean: {mean_pairwise:.4f} | Threshold: {threshold:.4f}")
        print(f"Std: {std_pairwise:.4f} | Max: {max_pairwise:.4f}")

        # -------------------------
        # 🔥 FINAL DECISION LOGIC
        # -------------------------
        is_borderline = False

        # 🔴 Rule 1: Too consistent → likely forged
        if std_pairwise < 0.015:
            result = "Forged"

        # 🔴 Rule 2: Flat distribution → forged
        elif (max_pairwise - mean_pairwise) < 0.02:
            result = "Forged"

        # 🔴 Rule 3: Outside cluster
        elif mean_pairwise > threshold:
            result = "Forged"

        else:
            lower = threshold * 0.9

            # Clearly genuine
            if mean_pairwise < lower:
                result = "Genuine"

            # Borderline
            else:
                is_borderline = True

                cond1 = query_dist < (threshold * 1.1)
                cond2 = std_pairwise > 0.01   # ensure variation
                cond3 = max_pairwise < (threshold * 1.2)

                score = sum([cond1, cond2, cond3])

                result = "Genuine" if score >= 2 else "Forged"

        return jsonify({
            "result": result,
            "distance": round(query_dist, 4),
            "threshold": round(threshold, 4),
            "is_borderline": is_borderline
        })

    except Exception as e:
        print("❌ Error:", str(e))
        return jsonify({"error": str(e)}), 500


# -----------------------------
# Run Server
# -----------------------------
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)