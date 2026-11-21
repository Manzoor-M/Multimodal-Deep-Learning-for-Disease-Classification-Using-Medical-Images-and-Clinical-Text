import os
import xml.etree.ElementTree as ET
import pandas as pd


# =========================
# PATHS
# =========================

IMAGE_DIR = "IU_Xray/NLMCXR_png"
REPORT_DIR = "IU_Xray/NLMCXR_reports/ecgen-radiology"

OUTPUT_FILE = "IU_Xray/dataset.csv"


# =========================
# STORAGE
# =========================

records = []

total_reports = 0
total_images = 0
matched_images = 0


# =========================
# READ XML FILES
# =========================

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


    # -------------------------
    # Report ID
    # -------------------------

    iuxr = root.find("IUXRId")

    if iuxr is not None:
        report_id = iuxr.get("id")
    else:
        report_id = ""


    # -------------------------
    # Clinical text
    # -------------------------

    indication = ""
    findings = ""
    impression = ""

    abstract_texts = root.findall(
        ".//AbstractText"
    )

    for item in abstract_texts:

        label = item.get("Label", "").upper()
        text = item.text or ""

        if label == "INDICATION":
            indication = text

        elif label == "FINDINGS":
            findings = text

        elif label == "IMPRESSION":
            impression = text


    # -------------------------
    # MeSH label
    # -------------------------

    mesh = root.find(".//MeSH/major")

    if mesh is not None:
        label = mesh.text or ""
    else:
        label = ""


    # -------------------------
    # Images
    # -------------------------

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

        # Check if corresponding image exists

        if os.path.exists(image_path):

            matched_images += 1

            records.append({
                "report_id": report_id,
                "image": image_filename,
                "indication": indication,
                "findings": findings,
                "impression": impression,
                "label": label
            })


# =========================
# CREATE DATAFRAME
# =========================

df = pd.DataFrame(records)


# =========================
# SAVE CSV
# =========================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================
# SUMMARY
# =========================

print("\n==============================")
print("DATASET PREPARATION COMPLETE")
print("==============================")

print(f"Total XML reports: {total_reports}")
print(f"Total image references: {total_images}")
print(f"Matched PNG images: {matched_images}")
print(f"Dataset rows: {len(df)}")

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nDataset saved to:")
print(OUTPUT_FILE)