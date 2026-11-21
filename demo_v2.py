import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image, ImageTk

import torch
import torch.nn as nn
from torchvision import transforms, models
from transformers import AutoTokenizer, AutoModel


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "IU_Xray"

IMAGE_DIR = os.path.join(
    DATA_DIR,
    "NLMCXR_png"
)

CHECKPOINT_PATH = "multimodal_v2_best.pth"

TEXT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"

IMAGE_SIZE = 224
MAX_TEXT_LENGTH = 128

TARGET_COLUMNS = [
    "cardiomegaly",
    "atelectasis",
    "pleural_effusion",
    "emphysema",
    "pulmonary_congestion",
    "opacity",
    "granulomatous_disease"
]

DISPLAY_NAMES = {
    "cardiomegaly": "Cardiomegaly",
    "atelectasis": "Atelectasis",
    "pleural_effusion": "Pleural Effusion",
    "emphysema": "Emphysema",
    "pulmonary_congestion": "Pulmonary Congestion",
    "opacity": "Opacity",
    "granulomatous_disease": "Granulomatous Disease"
}


# ============================================================
# VALIDATION-OPTIMIZED THRESHOLDS
# ============================================================

THRESHOLDS = {
    "cardiomegaly": 0.62,
    "atelectasis": 0.70,
    "pleural_effusion": 0.75,
    "emphysema": 0.79,
    "pulmonary_congestion": 0.76,
    "opacity": 0.51,
    "granulomatous_disease": 0.39
}


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("MEDICAL IMAGE + CLINICAL TEXT DIAGNOSIS")
print("MULTIMODAL V2 LIVE DEMO")
print("=" * 70)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print()


# ============================================================
# IMAGE TRANSFORM
# EXACTLY MATCHES EVALUATION
# ============================================================

image_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# MODEL
# ============================================================

class MultimodalV2(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # IMAGE ENCODER
        # ----------------------------------------------------

        self.image_encoder = models.efficientnet_b0(
            weights=None
        )

        self.image_encoder.classifier = nn.Identity()

        self.image_projection = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU()
        )

        # ----------------------------------------------------
        # TEXT ENCODER
        # ----------------------------------------------------

        self.text_encoder = AutoModel.from_pretrained(
            TEXT_MODEL
        )

        self.text_projection = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU()
        )

        # ----------------------------------------------------
        # FUSION
        # ----------------------------------------------------

        self.fusion = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU()
        )

        # ----------------------------------------------------
        # CLASSIFIER
        # ----------------------------------------------------

        self.classifier = nn.Linear(
            128,
            len(TARGET_COLUMNS)
        )


    def forward(
        self,
        image,
        input_ids,
        attention_mask
    ):

        # ====================================================
        # IMAGE BRANCH
        # ====================================================

        image_features = self.image_encoder(image)

        image_features = self.image_projection(
            image_features
        )

        # ====================================================
        # TEXT BRANCH
        # ====================================================

        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # CLS TOKEN REPRESENTATION
        text_features = text_output.last_hidden_state[:, 0]

        text_features = self.text_projection(
            text_features
        )

        # ====================================================
        # FUSION
        # ====================================================

        fused_features = torch.cat(
            [
                image_features,
                text_features
            ],
            dim=1
        )

        fused_features = self.fusion(
            fused_features
        )

        # ====================================================
        # CLASSIFICATION
        # ====================================================

        logits = self.classifier(
            fused_features
        )

        return logits


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("Loading BioClinicalBERT tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    TEXT_MODEL
)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading V2 model...")

model = MultimodalV2()

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)
model.eval()


print(
    f"Checkpoint epoch: "
    f"{checkpoint.get('epoch', 'N/A')}"
)

print(
    f"Validation loss: "
    f"{checkpoint.get('val_loss', 'N/A')}"
)

print("Model loaded successfully.")
print()


# ============================================================
# COUNT PARAMETERS
# ============================================================

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    f"Total parameters: {total_params:,}"
)

print(
    f"Trainable parameters: {trainable_params:,}"
)

print()


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict(image_path, indication):

    print("\n" + "=" * 70)
    print("RUNNING MULTIMODAL INFERENCE")
    print("=" * 70)

    print(f"Image: {image_path}")
    print(f"Indication: {indication}")

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = image_transform(
        image
    )

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    print(
        f"Image tensor: "
        f"{tuple(image_tensor.shape)}"
    )

    # --------------------------------------------------------
    # TEXT
    # --------------------------------------------------------

    encoded = tokenizer(
        indication,
        padding="max_length",
        truncation=True,
        max_length=MAX_TEXT_LENGTH,
        return_tensors="pt"
    )

    input_ids = encoded[
        "input_ids"
    ].to(device)

    attention_mask = encoded[
        "attention_mask"
    ].to(device)

    print(
        f"Text tokens: "
        f"{input_ids.shape[1]}"
    )

    # --------------------------------------------------------
    # MODEL FORWARD PASS
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(
            image_tensor,
            input_ids,
            attention_mask
        )

        probabilities = torch.sigmoid(
            logits
        )

    probabilities = (
        probabilities
        .cpu()
        .numpy()[0]
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    results = []

    print("\nMODEL OUTPUT")
    print("-" * 70)

    for i, target in enumerate(
        TARGET_COLUMNS
    ):

        probability = float(
            probabilities[i]
        )

        threshold = THRESHOLDS[
            target
        ]

        prediction = (
            probability >= threshold
        )

        results.append({
            "target": target,
            "probability": probability,
            "threshold": threshold,
            "prediction": prediction
        })

        print(
            f"{DISPLAY_NAMES[target]:25s}"
            f" Probability: {probability:.4f}"
            f" | Threshold: {threshold:.2f}"
            f" | Prediction: "
            f"{'POSITIVE' if prediction else 'NEGATIVE'}"
        )

    print("=" * 70)

    return results


# ============================================================
# GUI
# ============================================================

class DemoApp:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "Multimodal Medical Image + Clinical Text Demo"
        )

        self.root.geometry(
            "1150x750"
        )

        self.root.configure(
            bg="#f5f5f5"
        )

        self.image_path = None

        self.create_interface()


    # ========================================================
    # UI
    # ========================================================

    def create_interface(self):

        # ----------------------------------------------------
        # TITLE
        # ----------------------------------------------------

        title = tk.Label(
            self.root,
            text="Multimodal Medical Finding Classification",
            font=("Arial", 24, "bold"),
            bg="#f5f5f5"
        )

        title.pack(
            pady=(20, 5)
        )

        subtitle = tk.Label(
            self.root,
            text=(
                "EfficientNet-B0 + BioClinicalBERT "
                "| Multimodal V2"
            ),
            font=("Arial", 12),
            bg="#f5f5f5"
        )

        subtitle.pack(
            pady=(0, 20)
        )

        # ----------------------------------------------------
        # MAIN FRAME
        # ----------------------------------------------------

        main = tk.Frame(
            self.root,
            bg="#f5f5f5"
        )

        main.pack(
            fill="both",
            expand=True,
            padx=30,
            pady=10
        )

        # ----------------------------------------------------
        # LEFT PANEL
        # ----------------------------------------------------

        left = tk.Frame(
            main,
            bg="white",
            bd=1,
            relief="solid"
        )

        left.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(0, 15)
        )

        tk.Label(
            left,
            text="Input",
            font=("Arial", 17, "bold"),
            bg="white"
        ).pack(
            pady=15
        )

        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        self.image_label = tk.Label(
            left,
            text="No X-ray selected",
            bg="#eeeeee",
            width=55,
            height=18
        )

        self.image_label.pack(
            padx=20,
            pady=10
        )

        tk.Button(
            left,
            text="Select Chest X-ray",
            command=self.select_image,
            font=("Arial", 11, "bold"),
            padx=15,
            pady=8
        ).pack(
            pady=10
        )

        # ----------------------------------------------------
        # INDICATION
        # ----------------------------------------------------

        tk.Label(
            left,
            text="Clinical Indication",
            font=("Arial", 12, "bold"),
            bg="white"
        ).pack(
            anchor="w",
            padx=25,
            pady=(15, 5)
        )

        self.indication_box = tk.Text(
            left,
            height=5,
            width=55,
            font=("Arial", 11)
        )

        self.indication_box.pack(
            padx=25
        )

        self.indication_box.insert(
            "1.0",
            "Chest pain"
        )

        # ----------------------------------------------------
        # PREDICT BUTTON
        # ----------------------------------------------------

        tk.Button(
            left,
            text="RUN PREDICTION",
            command=self.run_prediction,
            font=("Arial", 13, "bold"),
            padx=20,
            pady=12
        ).pack(
            pady=25
        )

        # ----------------------------------------------------
        # RIGHT PANEL
        # ----------------------------------------------------

        right = tk.Frame(
            main,
            bg="white",
            bd=1,
            relief="solid"
        )

        right.pack(
            side="right",
            fill="both",
            expand=True
        )

        tk.Label(
            right,
            text="Model Output",
            font=("Arial", 17, "bold"),
            bg="white"
        ).pack(
            pady=15
        )

        self.results_frame = tk.Frame(
            right,
            bg="white"
        )

        self.results_frame.pack(
            fill="both",
            expand=True,
            padx=25
        )

        # ----------------------------------------------------
        # MODEL INFO
        # ----------------------------------------------------

        info = (
            "MODEL\n"
            "EfficientNet-B0 + BioClinicalBERT\n\n"
            "IMAGE INPUT\n"
            "224 × 224\n\n"
            "TEXT INPUT\n"
            "Maximum 128 tokens\n\n"
            "FUSION\n"
            "256-D Image + 256-D Text\n"
            "→ 512-D representation\n\n"
            "CHECKPOINT\n"
            "V2 | Epoch 5\n\n"
            "DECISION\n"
            "Validation-optimized thresholds"
        )

        tk.Label(
            right,
            text=info,
            font=("Arial", 10),
            justify="left",
            bg="#fafafa",
            anchor="w",
            padx=15,
            pady=15
        ).pack(
            fill="x",
            padx=20,
            pady=20
        )


    # ========================================================
    # SELECT IMAGE
    # ========================================================

    def select_image(self):

        path = filedialog.askopenfilename(
            title="Select Chest X-ray",
            filetypes=[
                (
                    "Image files",
                    "*.png *.jpg *.jpeg"
                )
            ]
        )

        if not path:
            return

        self.image_path = path

        image = Image.open(
            path
        ).convert("RGB")

        image.thumbnail(
            (500, 350)
        )

        self.tk_image = ImageTk.PhotoImage(
            image
        )

        self.image_label.configure(
            image=self.tk_image,
            text=""
        )


    # ========================================================
    # RUN PREDICTION
    # ========================================================

    def run_prediction(self):

        if self.image_path is None:

            messagebox.showwarning(
                "Missing image",
                "Please select a chest X-ray first."
            )

            return

        indication = (
            self.indication_box
            .get("1.0", "end")
            .strip()
        )

        if not indication:

            messagebox.showwarning(
                "Missing indication",
                "Please enter a clinical indication."
            )

            return

        try:

            results = predict(
                self.image_path,
                indication
            )

            self.display_results(
                results
            )

        except Exception as e:

            messagebox.showerror(
                "Prediction Error",
                str(e)
            )

            print(
                "\nERROR:",
                e
            )


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    def display_results(
        self,
        results
    ):

        # Clear previous results

        for widget in self.results_frame.winfo_children():

            widget.destroy()

        # Header

        tk.Label(
            self.results_frame,
            text="Finding",
            font=("Arial", 11, "bold"),
            bg="white",
            width=24,
            anchor="w"
        ).grid(
            row=0,
            column=0,
            pady=8
        )

        tk.Label(
            self.results_frame,
            text="Probability",
            font=("Arial", 11, "bold"),
            bg="white"
        ).grid(
            row=0,
            column=1,
            padx=10
        )

        tk.Label(
            self.results_frame,
            text="Prediction",
            font=("Arial", 11, "bold"),
            bg="white"
        ).grid(
            row=0,
            column=2,
            padx=10
        )

        # Results

        for row, result in enumerate(
            results,
            start=1
        ):

            name = DISPLAY_NAMES[
                result["target"]
            ]

            probability = (
                result["probability"]
            )

            prediction = (
                result["prediction"]
            )

            tk.Label(
                self.results_frame,
                text=name,
                bg="white",
                font=("Arial", 10),
                anchor="w",
                width=24
            ).grid(
                row=row,
                column=0,
                sticky="w",
                pady=7
            )

            tk.Label(
                self.results_frame,
                text=f"{probability:.2%}",
                bg="white",
                font=("Arial", 10)
            ).grid(
                row=row,
                column=1
            )

            result_text = (
                "POSITIVE"
                if prediction
                else "NEGATIVE"
            )

            result_label = tk.Label(
                self.results_frame,
                text=result_text,
                font=("Arial", 10, "bold"),
                bg="white"
            )

            result_label.grid(
                row=row,
                column=2,
                padx=10
            )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = DemoApp(root)

    root.mainloop()