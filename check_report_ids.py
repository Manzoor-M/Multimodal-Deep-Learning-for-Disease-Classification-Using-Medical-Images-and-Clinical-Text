import pandas as pd

df = pd.read_csv("IU_Xray/multimodal_dataset.csv")

counts = df.groupby("report_id")["image"].count()

print("\n==============================")
print("REPORT / IMAGE DISTRIBUTION")
print("==============================")

print(f"Unique report IDs: {len(counts)}")
print(f"Total images: {len(df)}")

print("\nImages per report:")
print(counts.value_counts().sort_index())

print("\nReports with more than 2 images:")
print((counts > 2).sum())

print("\nMaximum images for one report:")
print(counts.max())