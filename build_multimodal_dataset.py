import os
import xml.etree.ElementTree as ET
import pandas as pd


REPORT_DIR = "IU_Xray/NLMCXR_reports/ecgen-radiology"
IMAGE_DIR = "IU_Xray/NLMCXR_png"

OUTPUT_FILE = "IU_Xray/multimodal_dataset.csv"


# ============================================================
# TARGET CATEGORIES
# ============================================================

TARGET_PATTERNS = {
    "cardiomegaly": [
        "Cardiomegaly",
        "Cardiac Shadow/enlarged"
    ],

    "atelectasis": [
        "Pulmonary Atelectasis"
    ],

    "pleural_effusion": [
        "Pleural Effusion"
    ],

    "emphysema": [
        "Emphysema",
        "Pulmonary Emphysema"
    ],

    "pulmonary_congestion": [
        "Pulmonary Congestion"
    ],

    "opacity": [
        "Opacity/lung"
    ],

    "granulomatous_disease": [
        "Granulomatous Disease",
        "Calcified Granuloma"
    ]
}


records = []

total_reports = 0
total_images = 0
matched_images = 0


# ============================================================
# PROCESS XML REPORTS
# ============================================================

for filename in os.listdir(REPORT_DIR):

    if not filename.endswith(".xml"):
        continue

    total_reports += 1

    xml_path = os.path.join(REPORT_DIR, filename)

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

    except Exception as e:
        print(f"Error reading {filename}: {e}")
        continue


    # --------------------------------------------------------
    # REPORT ID
    # --------------------------------------------------------

    iuxr = root.find("IUXRId")

    if iuxr is not None:
        report_id = iuxr.get("id")
    else:
        report_id = filename.replace(".xml", "")


    # --------------------------------------------------------
    # CLINICAL TEXT
    # --------------------------------------------------------

    indication = ""
    findings = ""
    impression = ""

    for item in root.findall(".//AbstractText"):

        label = item.get("Label", "").upper()
        text = (item.text or "").strip()

        if label == "INDICATION":
            indication = text

        elif label == "FINDINGS":
            findings = text

        elif label == "IMPRESSION":
            impression = text


    clinical_text_parts = []

    if indication:
        clinical_text_parts.append(
            f"Indication: {indication}"
        )

    if findings:
        clinical_text_parts.append(
            f"Findings: {findings}"
        )

    if impression:
        clinical_text_parts.append(
            f"Impression: {impression}"
        )

    clinical_text = " ".join(clinical_text_parts)


    # --------------------------------------------------------
    # EXTRACT ALL MESH TERMS
    # --------------------------------------------------------

    mesh_terms = []

    for major in root.findall(".//MeSH/major"):

        if major.text:

            mesh_terms.append(
                major.text.strip()
            )


    # --------------------------------------------------------
    # CREATE BINARY TARGETS
    # --------------------------------------------------------

    targets = {}

    for category, patterns in TARGET_PATTERNS.items():

        targets[category] = int(
            any(
                term.startswith(pattern)
                for term in mesh_terms
                for pattern in patterns
            )
        )


    # --------------------------------------------------------
    # FIND ASSOCIATED IMAGES
    # --------------------------------------------------------

    parent_images = root.findall(".//parentImage")

    for image in parent_images:

        total_images += 1

        image_id = image.get("id")

        if not image_id:
            continue

        image_filename = image_id + ".png"

        image_path = os.path.join(
            IMAGE_DIR,
            image_filename
        )

        if not os.path.exists(image_path):
            continue

        matched_images += 1

        record = {
            "report_id": report_id,
            "image": image_filename,
            "clinical_text": clinical_text
        }

        record.update(targets)

        records.append(record)


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(records)


# ============================================================
# SAVE DATASET
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# REPORT
# ============================================================

print("\n==============================")
print("MULTIMODAL DATASET CREATED")
print("==============================")

print(f"Total XML reports: {total_reports}")
print(f"Total image references: {total_images}")
print(f"Matched PNG images: {matched_images}")
print(f"Dataset rows: {len(df)}")

print("\nTarget distribution:")

target_columns = list(TARGET_PATTERNS.keys())

for column in target_columns:

    positives = df[column].sum()

    print(
        f"{column:25} "
        f"{positives}"
    )

print("\nDataset saved to:")
print(OUTPUT_FILE)