"""
inspect_data.py

Quick exploratory look at the Business Entity Resolution Challenge data.
Run this from inside student_resource/ after activating your venv:

    python inspect_data.py

It loads the three train sources + ground truth, prints shapes, sample
rows, null counts, and country distributions so you can see the real
noise patterns before writing normalization code.
"""

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


def load_data():
    s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t")
    s2 = pd.read_csv("dataset/train/train_source2.tsv", sep="\t")
    s3 = pd.read_csv("dataset/train/train_source3.tsv", sep="\t")
    gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t")
    return s1, s2, s3, gt


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():
    s1, s2, s3, gt = load_data()

    print_section("SHAPES")
    print(f"source1: {s1.shape}")
    print(f"source2: {s2.shape}")
    print(f"source3: {s3.shape}")
    print(f"ground_truth: {gt.shape}")

    print_section("SOURCE 1 SAMPLE (20 rows)")
    print(s1.head(20).to_string())

    print_section("SOURCE 2 SAMPLE (20 rows)")
    print(s2.head(20).to_string())

    print_section("SOURCE 3 SAMPLE (20 rows)")
    print(s3.head(20).to_string())

    print_section("GROUND TRUTH SAMPLE (20 rows)")
    print(gt.head(20).to_string())

    print_section("NULL COUNTS")
    print("source1:\n", s1.isna().sum())
    print("\nsource2:\n", s2.isna().sum())
    print("\nsource3:\n", s3.isna().sum())

    print_section("COUNTRY DISTRIBUTION")
    print("source1:\n", s1["country"].value_counts(dropna=False))
    print("\nsource2:\n", s2["country"].value_counts(dropna=False))
    print("\nsource3:\n", s3["country"].value_counts(dropna=False))

    print_section("SINGLETON STATS (S1 entities with no matches)")
    empty_mask = gt["matched_entity_ids"].isna() | (
        gt["matched_entity_ids"].astype(str).str.strip() == ""
    )
    n_singletons = empty_mask.sum()
    print(f"Singletons: {n_singletons} / {len(gt)} ({n_singletons / len(gt):.1%})")

    print_section("MATCH COUNT DISTRIBUTION (non-singleton entities)")
    match_counts = gt.loc[~empty_mask, "matched_entity_ids"].astype(str).apply(
        lambda x: len(x.split(","))
    )
    print(match_counts.value_counts().sort_index())

    print_section("SAMPLE MATCHED RECORDS (side-by-side name/address comparison)")
    s1_idx = s1.set_index("entity_id")
    s2_idx = s2.set_index("entity_id")
    s3_idx = s3.set_index("entity_id")

    shown = 0
    for _, row in gt.loc[~empty_mask].iterrows():
        if shown >= 8:
            break
        s1_id = row["source1_entity_id"]
        if s1_id not in s1_idx.index:
            continue
        print(f"\n--- {s1_id} ---")
        print(f"  S1 name/addr: {s1_idx.loc[s1_id, 'business_name']!r} | "
              f"{s1_idx.loc[s1_id, 'business_address']!r}")
        for match_id in str(row["matched_entity_ids"]).split(","):
            match_id = match_id.strip()
            if match_id.startswith("S2-") and match_id in s2_idx.index:
                r = s2_idx.loc[match_id]
            elif match_id.startswith("S3-") and match_id in s3_idx.index:
                r = s3_idx.loc[match_id]
            else:
                continue
            print(f"  {match_id} name/addr: {r['business_name']!r} | {r['business_address']!r}")
        shown += 1


if __name__ == "__main__":
    main()
