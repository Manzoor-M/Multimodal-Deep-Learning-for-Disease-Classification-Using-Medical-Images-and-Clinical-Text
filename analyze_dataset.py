import os
import xml.etree.ElementTree as ET
from collections import Counter
import pandas as pd


REPORT_DIR = "IU_Xray/NLMCXR_reports/ecgen-radiology"


# Broad categories we want to investigate
TARGET_CATEGORIES = {
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


# Store one set of categories per report
report_categories = {}

# Count raw terms
all_terms = Counter()

# Duplicate report IDs
report_ids = []


for filename in os.listdir(REPORT_DIR):

    if not filename.endswith(".xml"):
        continue

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
        report_id = filename

    report_ids.append(report_id)


    # -------------------------
    # All MeSH terms
    # -------------------------

    terms = []

    for major in root.findall(".//MeSH/major"):

        if major.text:

            term = major.text.strip()

            terms.append(term)
            all_terms[term] += 1


    # -------------------------
    # Map to broad categories
    # -------------------------

    categories = set()

    for term in terms:

        for category, patterns in TARGET_CATEGORIES.items():

            for pattern in patterns:

                if term.startswith(pattern):
                    categories.add(category)


    report_categories[report_id] = categories


# =====================================
# REPORT ID ANALYSIS
# =====================================

id_counts = Counter(report_ids)

duplicate_ids = {
    report_id: count
    for report_id, count in id_counts.items()
    if count > 1
}


print("\n==============================")
print("REPORT ID ANALYSIS")
print("==============================")

print(f"Total XML files: {len(report_ids)}")
print(f"Unique report IDs: {len(id_counts)}")
print(f"Duplicate report IDs: {len(duplicate_ids)}")

if duplicate_ids:

    print("\nDuplicate report IDs:")

    for report_id, count in list(duplicate_ids.items())[:20]:
        print(f"{report_id}: {count}")


# =====================================
# CATEGORY COUNTS
# =====================================

category_counts = Counter()

for categories in report_categories.values():

    for category in categories:
        category_counts[category] += 1


print("\n==============================")
print("BROAD CATEGORY COUNTS")
print("==============================")

for category, count in category_counts.most_common():

    print(f"{category:25} {count}")


# =====================================
# CATEGORY CO-OCCURRENCE
# =====================================

category_names = list(TARGET_CATEGORIES.keys())

co_occurrence = []

for report_id, categories in report_categories.items():

    categories = sorted(categories)

    if len(categories) >= 2:

        for i in range(len(categories)):

            for j in range(i + 1, len(categories)):

                co_occurrence.append(
                    (categories[i], categories[j])
                )


pair_counts = Counter(co_occurrence)


print("\n==============================")
print("CATEGORY CO-OCCURRENCE")
print("==============================")

for (a, b), count in pair_counts.most_common():

    print(f"{a:25} + {b:25} {count}")