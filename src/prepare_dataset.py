#!/usr/bin/env python3
"""
Dataset Preparation Pipeline
=============================
Cyberbullying Type Classification on Indonesian Text Using TF-IDF

This script executes Phases 1–8 of the dataset preparation pipeline:
  Phase 1: Dataset Inspection
  Phase 2: Schema Standardization
  Phase 3: Label Analysis
  Phase 4: Label Mapping
  Phase 5: Dataset Merge
  Phase 6: Dataset Cleaning
  Phase 7: Dataset Validation
  Phase 8: Research Readiness

Usage:
    python3 src/prepare_dataset.py
"""

import csv
import os
import re
import ast
import json
from collections import Counter, defaultdict
from datetime import datetime

# ─── Configuration ───────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "dataset", "raw")
INTERIM_DIR = os.path.join(BASE_DIR, "dataset", "interim")
PROCESSED_DIR = os.path.join(BASE_DIR, "dataset", "processed")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

# Ensure directories exist
for d in [INTERIM_DIR, PROCESSED_DIR, DOCS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Utility Functions ──────────────────────────────────────────────────────

def read_csv(filepath):
    """Read a CSV and return (headers, rows)."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = [row for row in reader]
    return headers, rows


def write_csv(filepath, headers, rows):
    """Write headers + rows to CSV."""
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"  ✓ Saved {filepath}  ({len(rows)} rows)")


def write_md(filepath, content):
    """Write markdown content to file."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  ✓ Saved {filepath}")


def is_valid_text(text):
    """Check if text is valid (non-empty, not just whitespace/symbols)."""
    if not text or not text.strip():
        return False
    # Remove all punctuation/symbols — if nothing remains, invalid
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", text).strip()
    return len(cleaned) > 0


def text_stats(texts):
    """Return text length statistics."""
    lengths = [len(t.split()) for t in texts if t.strip()]
    if not lengths:
        return {"avg": 0, "min": 0, "max": 0, "total_words": 0}
    return {
        "avg": round(sum(lengths) / len(lengths), 2),
        "min": min(lengths),
        "max": max(lengths),
        "total_words": sum(lengths),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 1 — Dataset Inspection
# ═══════════════════════════════════════════════════════════════════════════

def phase1_inspect():
    """Inspect all datasets and generate dataset-analysis.md."""
    print("\n" + "=" * 70)
    print("  PHASE 1 — Dataset Inspection")
    print("=" * 70)

    datasets_info = []

    # --- Dataset A: cyberbullying_cleaned_indo.csv ---
    filepath_a = os.path.join(RAW_DIR, "cyberbullying_cleaned_indo.csv")
    h_a, r_a = read_csv(filepath_a)
    label_idx = h_a.index("cyberbullying_type")
    text_idx = h_a.index("clean_text")
    labels_a = Counter(row[label_idx] for row in r_a)
    missing_text = sum(1 for row in r_a if text_idx >= len(row) or not row[text_idx].strip())
    dup_a = len(r_a) - len(set(tuple(row) for row in r_a))

    datasets_info.append({
        "name": "cyberbullying_cleaned_indo.csv",
        "language": "Indonesian (machine-translated)",
        "rows": len(r_a),
        "text_col": "clean_text",
        "label_col": "cyberbullying_type",
        "num_classes": len(labels_a),
        "labels": labels_a,
        "missing": missing_text,
        "duplicates": dup_a,
        "suitable": "Suitable",
        "reason": "Has multi-class cyberbullying type labels that directly match the research requirements. Text appears machine-translated from English but labels are ideal (age, ethnicity, gender, religion, other_cyberbullying, not_cyberbullying)."
    })

    # --- Dataset B: hatespeech & abusive/data.csv ---
    filepath_b = os.path.join(RAW_DIR, "hatespeech & abusive", "data.csv")
    h_b, r_b = read_csv(filepath_b)
    tweet_idx = h_b.index("Tweet")
    hs_idx = h_b.index("HS")
    ab_idx = h_b.index("Abusive")
    missing_b = sum(1 for row in r_b if tweet_idx >= len(row) or not row[tweet_idx].strip())
    dup_b = len(r_b) - len(set(tuple(row) for row in r_b))

    # Derive class labels from HS + subcategories
    derived_labels_b = Counter()
    for row in r_b:
        hs = row[hs_idx]
        ab = row[ab_idx]
        if hs == "1":
            derived_labels_b["hate_speech (derived)"] += 1
        elif ab == "1":
            derived_labels_b["insult (derived)"] += 1
        else:
            derived_labels_b["normal (derived)"] += 1

    datasets_info.append({
        "name": "hatespeech & abusive/data.csv",
        "language": "Indonesian (native)",
        "rows": len(r_b),
        "text_col": "Tweet",
        "label_col": "HS, Abusive, HS_* subcategories",
        "num_classes": f"2 binary (HS, Abusive) + 7 subcategories → {len(derived_labels_b)} derived classes",
        "labels": derived_labels_b,
        "missing": missing_b,
        "duplicates": dup_b,
        "suitable": "Suitable",
        "reason": "Native Indonesian tweets with detailed hate speech subcategories (Religion, Race, Physical, Gender, Other). Can be mapped to cyberbullying types. Large dataset (13,169 rows). Has 125 duplicate rows and 146 text duplicates that need cleaning."
    })

    # --- Dataset C: indotoxic2024_annotated_data_v2_final.csv ---
    filepath_c = os.path.join(RAW_DIR, "indotoxic2024_annotated_data_v2_final.csv")
    h_c, r_c = read_csv(filepath_c)
    text_idx_c = h_c.index("text")
    missing_c = sum(1 for row in r_c if text_idx_c >= len(row) or not row[text_idx_c].strip())
    dup_c = len(r_c) - len(set(tuple(row) for row in r_c))

    # Sample annotation resolution
    annot_cols = ["toxicity", "insults", "threat_incitement_to_violence",
                  "identity_attack", "profanity_obscenity", "sexually_explicit"]
    annot_summary = {}
    for col in annot_cols:
        if col in h_c:
            idx = h_c.index(col)
            positive_count = 0
            for row in r_c:
                if idx < len(row):
                    try:
                        vals = ast.literal_eval(row[idx])
                        if isinstance(vals, list) and any(int(v) == 1 for v in vals):
                            positive_count += 1
                    except (ValueError, SyntaxError):
                        pass
            annot_summary[col] = positive_count

    datasets_info.append({
        "name": "indotoxic2024_annotated_data_v2_final.csv",
        "language": "Indonesian (native)",
        "rows": len(r_c),
        "text_col": "text",
        "label_col": "Multi-annotator binary (toxicity, insults, threat, identity_attack, profanity, sexually_explicit)",
        "num_classes": f"6 annotation columns → multi-class derived",
        "labels": annot_summary,
        "missing": missing_c,
        "duplicates": dup_c,
        "suitable": "Partially Suitable",
        "reason": "Authentic Indonesian text from social media with 28,448 rows. Uses multi-annotator binary labeling across 6 toxicity categories. Requires annotator disagreement resolution and multi-label to single-label conversion. Labels can be mapped to cyberbullying types. Very large dataset but complex transformation needed."
    })

    # --- Support files ---
    support_files = [
        {
            "name": "hatespeech & abusive/abusive.csv",
            "rows": 125,
            "description": "Abusive word list (125 words). Support file for text analysis, NOT a training dataset.",
            "suitable": "Not Suitable (Support File)"
        },
        {
            "name": "hatespeech & abusive/new_kamusalay.csv",
            "rows": 15166,
            "description": "Slang/alay normalization dictionary (15,166 entries). Support file for preprocessing, NOT a training dataset.",
            "suitable": "Not Suitable (Support File)"
        },
        {
            "name": "kamus_singkatan.csv",
            "rows": 1503,
            "description": "Indonesian abbreviation dictionary (1,503 entries). Support file for preprocessing, NOT a training dataset.",
            "suitable": "Not Suitable (Support File)"
        },
    ]

    # --- Deleted files ---
    deleted_files = [
        {
            "name": "combined_dataset.csv (DELETED)",
            "rows": 2066,
            "description": "Sentiment analysis dataset with labels: positif, negatif, positive, negative, Bullying, Non-bullying. These are sentiment polarity labels, NOT cyberbullying type labels. Deleted per user approval.",
            "suitable": "Not Suitable (Deleted)"
        },
        {
            "name": "hatespeech & abusive/re_dataset.csv (DELETED)",
            "rows": 13169,
            "description": "Exact duplicate of data.csv (byte-for-byte identical). Deleted to avoid confusion.",
            "suitable": "Not Suitable (Deleted)"
        },
    ]

    # Generate dataset-analysis.md
    md = "# Dataset Analysis Report\n\n"
    md += f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md += "---\n\n"
    md += "## Training Dataset Candidates\n\n"
    md += "| Dataset | Language | Rows | Text Column | Label Column | Classes | Missing Values | Duplicate Rows | Suitable? |\n"
    md += "|---------|----------|------|-------------|--------------|---------|----------------|----------------|-----------|\n"

    for d in datasets_info:
        labels_str = str(d["num_classes"])
        md += f"| {d['name']} | {d['language']} | {d['rows']:,} | `{d['text_col']}` | `{d['label_col']}` | {labels_str} | {d['missing']} | {d['duplicates']} | **{d['suitable']}** |\n"

    md += "\n---\n\n"

    # Detailed suitability
    md += "## Suitability Analysis\n\n"
    for d in datasets_info:
        md += f"### {d['name']}\n\n"
        md += f"**Status**: {d['suitable']}\n\n"
        md += f"**Reason**: {d['reason']}\n\n"
        md += f"**Label Distribution**:\n\n"
        if isinstance(d["labels"], Counter):
            for label, count in d["labels"].most_common():
                md += f"- `{label}`: {count:,}\n"
        elif isinstance(d["labels"], dict):
            for label, count in d["labels"].items():
                md += f"- `{label}` (at least 1 annotator positive): {count:,}\n"
        md += "\n---\n\n"

    # Support files
    md += "## Support Files (Not Training Datasets)\n\n"
    md += "| File | Rows | Description | Status |\n"
    md += "|------|------|-------------|--------|\n"
    for s in support_files:
        md += f"| {s['name']} | {s['rows']:,} | {s['description']} | {s['suitable']} |\n"

    md += "\n---\n\n"

    # Deleted files
    md += "## Deleted Files\n\n"
    md += "| File | Original Rows | Reason | Status |\n"
    md += "|------|---------------|--------|--------|\n"
    for d in deleted_files:
        md += f"| {d['name']} | {d['rows']:,} | {d['description']} | {d['suitable']} |\n"

    md += "\n---\n\n"
    md += "## Summary\n\n"
    md += "- **3 datasets** are suitable for training (after transformation)\n"
    md += "- **3 files** are support/preprocessing resources (kept for later use)\n"
    md += "- **2 files** were deleted (sentiment dataset + duplicate)\n"

    write_md(os.path.join(DOCS_DIR, "dataset-analysis.md"), md)
    return datasets_info


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 2 — Schema Standardization
# ═══════════════════════════════════════════════════════════════════════════

def phase2_standardize():
    """Standardize each dataset to (text, label) schema."""
    print("\n" + "=" * 70)
    print("  PHASE 2 — Schema Standardization")
    print("=" * 70)

    standardized = {}

    # --- Dataset A: cyberbullying_cleaned_indo.csv ---
    print("\n  [A] cyberbullying_cleaned_indo.csv")
    filepath_a = os.path.join(RAW_DIR, "cyberbullying_cleaned_indo.csv")
    h_a, r_a = read_csv(filepath_a)
    text_idx = h_a.index("clean_text")
    label_idx = h_a.index("cyberbullying_type")

    rows_a = []
    skipped_a = 0
    for row in r_a:
        text = row[text_idx].strip() if text_idx < len(row) else ""
        label = row[label_idx].strip() if label_idx < len(row) else ""
        if text and label:
            rows_a.append([text, label])
        else:
            skipped_a += 1
    print(f"    Skipped {skipped_a} rows with empty text/label")

    out_a = os.path.join(INTERIM_DIR, "cyberbullying_cleaned_indo_standardized.csv")
    write_csv(out_a, ["text", "label"], rows_a)
    standardized["A"] = {"path": out_a, "rows": rows_a, "name": "cyberbullying_cleaned_indo"}

    # --- Dataset B: hatespeech & abusive/data.csv ---
    print("\n  [B] hatespeech & abusive/data.csv")
    filepath_b = os.path.join(RAW_DIR, "hatespeech & abusive", "data.csv")
    h_b, r_b = read_csv(filepath_b)
    tweet_idx = h_b.index("Tweet")
    hs_idx = h_b.index("HS")
    ab_idx = h_b.index("Abusive")

    # Subcategory indices
    sub_indices = {}
    for col in ["HS_Individual", "HS_Group", "HS_Religion", "HS_Race",
                "HS_Physical", "HS_Gender", "HS_Other"]:
        if col in h_b:
            sub_indices[col] = h_b.index(col)

    rows_b = []
    skipped_b = 0
    label_stats_b = Counter()
    for row in r_b:
        text = row[tweet_idx].strip() if tweet_idx < len(row) else ""
        if not text:
            skipped_b += 1
            continue

        hs = row[hs_idx].strip()
        ab = row[ab_idx].strip()

        if hs == "1":
            # Check subcategories for specificity
            religion = row[sub_indices.get("HS_Religion", -1)].strip() == "1" if "HS_Religion" in sub_indices else False
            race = row[sub_indices.get("HS_Race", -1)].strip() == "1" if "HS_Race" in sub_indices else False
            physical = row[sub_indices.get("HS_Physical", -1)].strip() == "1" if "HS_Physical" in sub_indices else False
            gender = row[sub_indices.get("HS_Gender", -1)].strip() == "1" if "HS_Gender" in sub_indices else False

            if religion or race:
                label = "hate_speech"
            elif physical:
                label = "threat"
            elif gender:
                label = "harassment"
            else:
                label = "hate_speech"  # Default HS → hate_speech
        elif ab == "1":
            label = "insult"
        else:
            label = "normal"

        rows_b.append([text, label])
        label_stats_b[label] += 1

    print(f"    Skipped {skipped_b} rows with empty text")
    print(f"    Label distribution: {dict(label_stats_b.most_common())}")

    out_b = os.path.join(INTERIM_DIR, "hatespeech_abusive_standardized.csv")
    write_csv(out_b, ["text", "label"], rows_b)
    standardized["B"] = {"path": out_b, "rows": rows_b, "name": "hatespeech_abusive"}

    # --- Dataset C: indotoxic2024_annotated_data_v2_final.csv ---
    print("\n  [C] indotoxic2024_annotated_data_v2_final.csv")
    filepath_c = os.path.join(RAW_DIR, "indotoxic2024_annotated_data_v2_final.csv")
    h_c, r_c = read_csv(filepath_c)
    text_idx_c = h_c.index("text")

    annot_cols = {
        "threat_incitement_to_violence": "threat",
        "identity_attack": "hate_speech",
        "insults": "insult",
        "sexually_explicit": "sexually_explicit",
        "profanity_obscenity": "harassment",
    }
    annot_indices = {}
    for col in annot_cols:
        if col in h_c:
            annot_indices[col] = h_c.index(col)

    # Priority order for multi-label → single-label
    priority = ["threat_incitement_to_violence", "identity_attack", "insults",
                "sexually_explicit", "profanity_obscenity"]

    rows_c = []
    skipped_c = 0
    parse_errors_c = 0
    label_stats_c = Counter()

    for row in r_c:
        text = row[text_idx_c].strip() if text_idx_c < len(row) else ""
        if not text:
            skipped_c += 1
            continue

        # Resolve annotations via majority vote
        resolved = {}
        valid_row = True
        for col, target_label in annot_cols.items():
            if col in annot_indices:
                idx = annot_indices[col]
                if idx < len(row):
                    try:
                        vals = ast.literal_eval(row[idx])
                        if isinstance(vals, list):
                            # Majority vote: if at least half say 1, it's positive
                            positive = sum(1 for v in vals if int(v) == 1)
                            resolved[col] = 1 if positive >= len(vals) / 2 else 0
                        else:
                            resolved[col] = 0
                    except (ValueError, SyntaxError):
                        parse_errors_c += 1
                        resolved[col] = 0
                else:
                    resolved[col] = 0

        # Assign label by priority
        label = "normal"
        for p in priority:
            if resolved.get(p, 0) == 1:
                label = annot_cols[p]
                break

        rows_c.append([text, label])
        label_stats_c[label] += 1

    print(f"    Skipped {skipped_c} rows with empty text")
    print(f"    Parse errors: {parse_errors_c}")
    print(f"    Label distribution: {dict(label_stats_c.most_common())}")

    out_c = os.path.join(INTERIM_DIR, "indotoxic2024_standardized.csv")
    write_csv(out_c, ["text", "label"], rows_c)
    standardized["C"] = {"path": out_c, "rows": rows_c, "name": "indotoxic2024"}

    return standardized


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 3 — Label Analysis
# ═══════════════════════════════════════════════════════════════════════════

def phase3_label_analysis(standardized):
    """Analyze all labels and generate label-analysis.md."""
    print("\n" + "=" * 70)
    print("  PHASE 3 — Label Analysis")
    print("=" * 70)

    # Collect all original labels per dataset
    label_mappings = []

    # Dataset A original labels
    a_mappings = [
        ("age", "cyberbullying_cleaned_indo", "Cyberbullying based on age (ageism, age-based harassment)", "harassment"),
        ("ethnicity", "cyberbullying_cleaned_indo", "Cyberbullying based on ethnicity/race", "hate_speech"),
        ("gender", "cyberbullying_cleaned_indo", "Cyberbullying based on gender identity", "harassment"),
        ("not_cyberbullying", "cyberbullying_cleaned_indo", "Text that does not contain cyberbullying", "normal"),
        ("other_cyberbullying", "cyberbullying_cleaned_indo", "Other forms of cyberbullying not categorized above", "insult"),
        ("religion", "cyberbullying_cleaned_indo", "Cyberbullying based on religion", "hate_speech"),
    ]
    label_mappings.extend(a_mappings)

    # Dataset B derived labels
    b_mappings = [
        ("HS=0, Abusive=0", "hatespeech_abusive", "No hate speech and no abusive language — normal text", "normal"),
        ("HS=0, Abusive=1", "hatespeech_abusive", "Abusive language without hate speech — insult/profanity", "insult"),
        ("HS=1, HS_Religion=1 or HS_Race=1", "hatespeech_abusive", "Hate speech targeting religion or race", "hate_speech"),
        ("HS=1, HS_Physical=1", "hatespeech_abusive", "Hate speech with physical targeting — threat/intimidation", "threat"),
        ("HS=1, HS_Gender=1", "hatespeech_abusive", "Hate speech targeting gender", "harassment"),
        ("HS=1, HS_Other=1", "hatespeech_abusive", "Other forms of hate speech", "hate_speech"),
    ]
    label_mappings.extend(b_mappings)

    # Dataset C derived labels
    c_mappings = [
        ("threat_incitement_to_violence=1", "indotoxic2024", "Text contains threats or incitement to violence (highest priority)", "threat"),
        ("identity_attack=1", "indotoxic2024", "Text attacks based on identity (race, religion, etc.)", "hate_speech"),
        ("insults=1", "indotoxic2024", "Text contains insults/personal attacks", "insult"),
        ("sexually_explicit=1", "indotoxic2024", "Text contains sexually explicit content", "sexually_explicit"),
        ("profanity_obscenity=1", "indotoxic2024", "Text contains profanity/obscenity without other specific category", "harassment"),
        ("all annotations=0", "indotoxic2024", "No toxic annotations — normal text", "normal"),
    ]
    label_mappings.extend(c_mappings)

    # Generate label-analysis.md
    md = "# Label Analysis Report\n\n"
    md += f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md += "---\n\n"
    md += "## Label Mapping Table\n\n"
    md += "| Original Label | Dataset | Meaning | Recommended Final Label |\n"
    md += "|----------------|---------|---------|------------------------|\n"
    for orig, ds, meaning, final in label_mappings:
        md += f"| `{orig}` | {ds} | {meaning} | **{final}** |\n"

    md += "\n---\n\n"
    md += "## Final Label Schema\n\n"
    md += "| Final Label | Description | Source Categories |\n"
    md += "|-------------|-------------|-------------------|\n"
    md += "| `normal` | Text does not contain cyberbullying | not_cyberbullying, HS=0+Abusive=0, all annotations=0 |\n"
    md += "| `insult` | Insulting, demeaning, or abusive language | other_cyberbullying, HS=0+Abusive=1, insults=1 |\n"
    md += "| `hate_speech` | Identity-based hate speech (religion, race, ethnicity) | ethnicity, religion, HS_Religion, HS_Race, HS_Other, identity_attack=1 |\n"
    md += "| `threat` | Threats, intimidation, or incitement to violence | HS_Physical, threat_incitement=1 |\n"
    md += "| `harassment` | General harassment (age, gender, profanity-based) | age, gender, HS_Gender, profanity_obscenity=1 |\n"
    md += "| `sexually_explicit` | Sexually explicit content | sexually_explicit=1 |\n"

    md += "\n---\n\n"
    md += "## Mapping Justifications\n\n"

    md += "### `age` → `harassment`\n\n"
    md += "Age-based cyberbullying (ageism) is a form of general harassment. It does not fit hate_speech "
    md += "(which focuses on immutable identity characteristics like race/religion in Indonesian legal context) "
    md += "nor insult (which is non-targeted abusive language).\n\n"

    md += "### `ethnicity` → `hate_speech`\n\n"
    md += "Ethnicity-based cyberbullying is targeting someone based on their ethnic/racial background. "
    md += "This aligns with the definition of hate speech (ujaran kebencian) in Indonesian context.\n\n"

    md += "### `gender` → `harassment`\n\n"
    md += "Gender-based cyberbullying is a form of harassment. In the DRD, harassment is defined as "
    md += "'pelecehan verbal' (verbal harassment), which includes gender-based targeting.\n\n"

    md += "### `not_cyberbullying` → `normal`\n\n"
    md += "Directly maps to the non-cyberbullying class.\n\n"

    md += "### `other_cyberbullying` → `insult`\n\n"
    md += "The 'other' category from the cyberbullying dataset captures generic cyberbullying that doesn't "
    md += "fall into specific categories. This most closely aligns with 'insult' (general abusive language).\n\n"

    md += "### `religion` → `hate_speech`\n\n"
    md += "Religion-based cyberbullying targets religious identity. This is hate speech (ujaran kebencian) "
    md += "in Indonesian legal and social context.\n\n"

    md += "### `HS=1, HS_Physical=1` → `threat`\n\n"
    md += "Physical targeting in hate speech context implies threats or intimidation against physical safety.\n\n"

    md += "### `sexually_explicit=1` → `sexually_explicit`\n\n"
    md += "Sexually explicit content is kept as its own category as it represents a distinct type of cyberbullying.\n\n"

    md += "---\n\n"
    md += "## Notes\n\n"
    md += "- The `sexually_explicit` class may have very few samples. If it has fewer than 50 samples, "
    md += "it may need to be merged into `harassment` during model training to avoid class imbalance issues.\n"
    md += "- All mappings are based on semantic equivalence between source labels and the target schema.\n"
    md += "- No labels were fabricated; all derive from existing annotations.\n"

    write_md(os.path.join(DOCS_DIR, "label-analysis.md"), md)
    return label_mappings


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 4 — Label Mapping
# ═══════════════════════════════════════════════════════════════════════════

def phase4_label_mapping(standardized):
    """Apply label mapping and generate label_mapping.csv."""
    print("\n" + "=" * 70)
    print("  PHASE 4 — Label Mapping")
    print("=" * 70)

    # Mapping for Dataset A (cyberbullying_cleaned_indo)
    mapping_a = {
        "age": "harassment",
        "ethnicity": "hate_speech",
        "gender": "harassment",
        "not_cyberbullying": "normal",
        "other_cyberbullying": "insult",
        "religion": "hate_speech",
    }

    # Dataset B and C already have final labels from Phase 2
    # (they were derived during standardization)

    # Apply mapping to Dataset A
    rows_a = standardized["A"]["rows"]
    mapped_a = []
    for row in rows_a:
        text = row[0]
        original_label = row[1]
        final_label = mapping_a.get(original_label, original_label)
        mapped_a.append([text, final_label])
    standardized["A"]["rows"] = mapped_a

    # Generate label_mapping.csv
    mapping_rows = []
    for orig, final in mapping_a.items():
        mapping_rows.append([orig, final])

    # Add Dataset B mappings
    mapping_rows.extend([
        ["HS=0 & Abusive=0", "normal"],
        ["HS=0 & Abusive=1", "insult"],
        ["HS=1 & (HS_Religion=1 | HS_Race=1)", "hate_speech"],
        ["HS=1 & HS_Physical=1", "threat"],
        ["HS=1 & HS_Gender=1", "harassment"],
        ["HS=1 & HS_Other=1 (default)", "hate_speech"],
    ])

    # Add Dataset C mappings
    mapping_rows.extend([
        ["threat_incitement_to_violence=1", "threat"],
        ["identity_attack=1", "hate_speech"],
        ["insults=1", "insult"],
        ["sexually_explicit=1", "sexually_explicit"],
        ["profanity_obscenity=1", "harassment"],
        ["all annotations=0", "normal"],
    ])

    out_path = os.path.join(INTERIM_DIR, "label_mapping.csv")
    write_csv(out_path, ["Original Label", "Final Label"], mapping_rows)

    # Show final label distribution per dataset
    for key, ds in standardized.items():
        labels = Counter(row[1] for row in ds["rows"])
        print(f"  [{key}] {ds['name']}: {dict(labels.most_common())}")

    return standardized


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 5 — Dataset Merge
# ═══════════════════════════════════════════════════════════════════════════

def phase5_merge(standardized):
    """Merge compatible datasets and generate merge-report.md."""
    print("\n" + "=" * 70)
    print("  PHASE 5 — Dataset Merge")
    print("=" * 70)

    # Merge all 3 datasets
    merged_rows = []
    per_dataset_counts = {}

    for key, ds in standardized.items():
        count_before = len(ds["rows"])
        merged_rows.extend(ds["rows"])
        per_dataset_counts[ds["name"]] = count_before
        print(f"  Added {count_before:,} rows from {ds['name']}")

    total_merged = len(merged_rows)
    print(f"\n  Total merged: {total_merged:,} rows")

    # Save merged dataset
    out_path = os.path.join(INTERIM_DIR, "merged_dataset.csv")
    write_csv(out_path, ["text", "label"], merged_rows)

    # Label distribution after merge
    merged_labels = Counter(row[1] for row in merged_rows)

    # Generate merge-report.md
    md = "# Merge Report\n\n"
    md += f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md += "---\n\n"

    md += "## Merged Datasets\n\n"
    md += "| Dataset | Rows | Status |\n"
    md += "|---------|------|--------|\n"
    for name, count in per_dataset_counts.items():
        md += f"| {name} | {count:,} | ✅ Merged |\n"
    md += f"| **Total** | **{total_merged:,}** | |\n"

    md += "\n---\n\n"
    md += "## Excluded Datasets\n\n"
    md += "| Dataset | Original Rows | Reason |\n"
    md += "|---------|---------------|--------|\n"
    md += "| combined_dataset.csv | 2,066 | Sentiment analysis labels (positif/negatif/positive/negative). Not cyberbullying type classification. **Deleted.** |\n"
    md += "| re_dataset.csv | 13,169 | Exact duplicate of data.csv (byte-identical). **Deleted.** |\n"
    md += "| abusive.csv | 125 | Abusive word list (support file, not training data) |\n"
    md += "| new_kamusalay.csv | 15,166 | Slang normalization dictionary (support file) |\n"
    md += "| kamus_singkatan.csv | 1,503 | Abbreviation dictionary (support file) |\n"

    md += "\n---\n\n"
    md += "## Merge Statistics\n\n"
    md += f"- **Total samples before merge**: {sum(per_dataset_counts.values()):,} (across {len(per_dataset_counts)} datasets)\n"
    md += f"- **Total samples after merge**: {total_merged:,}\n"
    md += f"- **Total classes**: {len(merged_labels)}\n\n"

    md += "### Label Distribution After Merge\n\n"
    md += "| Label | Count | Percentage |\n"
    md += "|-------|-------|------------|\n"
    for label, count in merged_labels.most_common():
        pct = round(count / total_merged * 100, 2)
        md += f"| `{label}` | {count:,} | {pct}% |\n"

    md += "\n---\n\n"
    md += "## Notes\n\n"
    md += "- Merged dataset has NOT been deduplicated yet (Phase 6).\n"
    md += "- Label mapping was applied before merging.\n"
    md += "- The `sexually_explicit` class may have very few samples.\n"

    write_md(os.path.join(DOCS_DIR, "merge-report.md"), md)
    return merged_rows, merged_labels


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 6 — Dataset Cleaning
# ═══════════════════════════════════════════════════════════════════════════

def phase6_clean(merged_rows):
    """Clean the merged dataset."""
    print("\n" + "=" * 70)
    print("  PHASE 6 — Dataset Cleaning")
    print("=" * 70)

    valid_labels = {"normal", "insult", "hate_speech", "threat", "harassment", "sexually_explicit"}
    initial_count = len(merged_rows)

    # Step 1: Remove empty/null text
    step1 = [row for row in merged_rows if row[0] and row[0].strip()]
    removed_empty = initial_count - len(step1)
    print(f"  Removed {removed_empty} empty text rows")

    # Step 2: Remove invalid labels
    step2 = [row for row in step1 if row[1] in valid_labels]
    removed_invalid = len(step1) - len(step2)
    print(f"  Removed {removed_invalid} invalid label rows")

    # Step 3: Remove corrupted text (only symbols/numbers, too short)
    step3 = [row for row in step2 if is_valid_text(row[0])]
    removed_corrupted = len(step2) - len(step3)
    print(f"  Removed {removed_corrupted} corrupted text rows")

    # Step 4: Remove duplicates (text-based)
    seen_texts = set()
    step4 = []
    for row in step3:
        text_normalized = row[0].strip().lower()
        if text_normalized not in seen_texts:
            seen_texts.add(text_normalized)
            step4.append(row)
    removed_dups = len(step3) - len(step4)
    print(f"  Removed {removed_dups} duplicate rows (text-based)")

    final_count = len(step4)
    print(f"\n  Cleaning summary: {initial_count:,} → {final_count:,} ({initial_count - final_count:,} removed)")

    # Check sexually_explicit count
    se_count = sum(1 for row in step4 if row[1] == "sexually_explicit")
    if se_count < 50:
        print(f"\n  ⚠ sexually_explicit has only {se_count} samples (< 50).")
        print(f"    Merging sexually_explicit → harassment for statistical reliability.")
        step4 = [[row[0], "harassment" if row[1] == "sexually_explicit" else row[1]] for row in step4]

    # Save final dataset
    out_path = os.path.join(PROCESSED_DIR, "final_dataset.csv")
    write_csv(out_path, ["text", "label"], step4)

    # Also save to dataset/cleaned/ per TRD structure
    cleaned_path = os.path.join(BASE_DIR, "dataset", "cleaned", "final_dataset.csv")
    os.makedirs(os.path.dirname(cleaned_path), exist_ok=True)
    write_csv(cleaned_path, ["text", "label"], step4)

    return step4, {
        "initial": initial_count,
        "removed_empty": removed_empty,
        "removed_invalid": removed_invalid,
        "removed_corrupted": removed_corrupted,
        "removed_duplicates": removed_dups,
        "final": final_count,
        "se_merged": se_count < 50,
        "se_count": se_count,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 7 — Dataset Validation
# ═══════════════════════════════════════════════════════════════════════════

def phase7_validate(final_rows, cleaning_stats):
    """Validate final dataset and generate dataset-statistics.md."""
    print("\n" + "=" * 70)
    print("  PHASE 7 — Dataset Validation")
    print("=" * 70)

    total_rows = len(final_rows)
    texts = [row[0] for row in final_rows]
    labels = [row[1] for row in final_rows]
    label_counts = Counter(labels)

    # Statistics
    missing_text = sum(1 for t in texts if not t or not t.strip())
    missing_label = sum(1 for l in labels if not l or not l.strip())
    dup_texts = total_rows - len(set(t.strip().lower() for t in texts))

    stats = text_stats(texts)

    # Vocabulary (unique words)
    all_words = set()
    for t in texts:
        all_words.update(t.lower().split())
    vocab_size = len(all_words)

    # Find longest and shortest text
    longest_text = max(texts, key=lambda t: len(t.split()))
    shortest_text = min(texts, key=lambda t: len(t.split()) if t.strip() else float("inf"))

    print(f"  Total rows: {total_rows:,}")
    print(f"  Total classes: {len(label_counts)}")
    print(f"  Missing text: {missing_text}")
    print(f"  Missing labels: {missing_label}")
    print(f"  Duplicate texts: {dup_texts}")
    print(f"  Vocabulary size: {vocab_size:,}")

    # Generate dataset-statistics.md
    md = "# Dataset Statistics Report\n\n"
    md += f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md += "---\n\n"

    md += "## Overview\n\n"
    md += f"| Metric | Value |\n"
    md += f"|--------|-------|\n"
    md += f"| Total Rows | {total_rows:,} |\n"
    md += f"| Total Classes | {len(label_counts)} |\n"
    md += f"| Missing Text Values | {missing_text} |\n"
    md += f"| Missing Label Values | {missing_label} |\n"
    md += f"| Duplicate Rows (text-based) | {dup_texts} |\n"
    md += f"| Average Text Length (words) | {stats['avg']} |\n"
    md += f"| Shortest Text (words) | {stats['min']} |\n"
    md += f"| Longest Text (words) | {stats['max']} |\n"
    md += f"| Vocabulary Size (unique words) | {vocab_size:,} |\n"

    md += "\n---\n\n"
    md += "## Samples per Class\n\n"
    md += "| Label | Count | Percentage |\n"
    md += "|-------|-------|------------|\n"
    for label, count in label_counts.most_common():
        pct = round(count / total_rows * 100, 2)
        md += f"| `{label}` | {count:,} | {pct}% |\n"

    md += "\n---\n\n"
    md += "## Cleaning Summary\n\n"
    md += "| Step | Rows Removed |\n"
    md += "|------|--------------|\n"
    md += f"| Empty text | {cleaning_stats['removed_empty']} |\n"
    md += f"| Invalid labels | {cleaning_stats['removed_invalid']} |\n"
    md += f"| Corrupted text | {cleaning_stats['removed_corrupted']} |\n"
    md += f"| Duplicate rows | {cleaning_stats['removed_duplicates']} |\n"
    md += f"| **Total removed** | **{cleaning_stats['initial'] - cleaning_stats['final']:,}** |\n"
    md += f"| **Initial count** | {cleaning_stats['initial']:,} |\n"
    md += f"| **Final count** | {cleaning_stats['final']:,} |\n"

    if cleaning_stats.get("se_merged"):
        md += f"\n> **Note**: `sexually_explicit` class had only {cleaning_stats['se_count']} samples and was merged into `harassment` for statistical reliability.\n"

    md += "\n---\n\n"
    md += "## Text Length Distribution\n\n"
    md += "| Range (words) | Count |\n"
    md += "|---------------|-------|\n"
    len_ranges = [(1, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, 200), (201, float("inf"))]
    for lo, hi in len_ranges:
        count = sum(1 for t in texts if lo <= len(t.split()) <= hi)
        hi_str = f"{int(hi)}" if hi != float("inf") else "∞"
        md += f"| {lo}–{hi_str} | {count:,} |\n"

    md += "\n---\n\n"

    # Phase 8 — Research Readiness
    md += "## Research Readiness Assessment\n\n"
    md += "### Dataset Readiness for ML Pipeline\n\n"
    md += "| Requirement | Status | Notes |\n"
    md += "|-------------|--------|-------|\n"

    # Check minimum size
    min_ok = total_rows >= 5000
    md += f"| Minimum 5,000 samples | {'✅ Pass' if min_ok else '⚠️ Below target'} | {total_rows:,} samples |\n"

    # Check multi-class
    mc_ok = len(label_counts) >= 3
    md += f"| Multi-class (≥3 classes) | {'✅ Pass' if mc_ok else '❌ Fail'} | {len(label_counts)} classes |\n"

    # Check no missing values
    no_missing = missing_text == 0 and missing_label == 0
    md += f"| No missing values | {'✅ Pass' if no_missing else '❌ Fail'} | text={missing_text}, label={missing_label} |\n"

    # Check no duplicates
    no_dups = dup_texts == 0
    md += f"| No duplicate texts | {'✅ Pass' if no_dups else '⚠️ Has duplicates'} | {dup_texts} duplicates |\n"

    # Check label format
    md += f"| Consistent label format | ✅ Pass | All labels are lowercase snake_case |\n"
    md += f"| Text column valid | ✅ Pass | All entries are non-empty strings |\n"

    md += "\n---\n\n"
    md += "### Readiness for Specific Algorithms\n\n"
    md += "| Algorithm | Ready? | Notes |\n"
    md += "|-----------|--------|-------|\n"
    md += "| **TF-IDF** | ✅ Ready | Text data suitable for bag-of-words representation |\n"
    md += "| **Naive Bayes** | ✅ Ready | Works well with TF-IDF features, handles multi-class |\n"
    md += "| **Logistic Regression** | ✅ Ready | Handles high-dimensional sparse TF-IDF features well |\n"
    md += "| **Support Vector Machine** | ✅ Ready | Effective with TF-IDF, good for text classification |\n"

    md += "\n---\n\n"
    md += "### Remaining Considerations\n\n"

    # Check class imbalance
    max_count = max(label_counts.values())
    min_count = min(label_counts.values())
    imbalance_ratio = round(max_count / min_count, 2)
    md += f"1. **Class Imbalance**: Ratio {imbalance_ratio}:1 (largest/smallest class). "
    if imbalance_ratio > 10:
        md += "Significant imbalance detected. Consider using stratified split, class weights, or oversampling/undersampling.\n"
    elif imbalance_ratio > 3:
        md += "Moderate imbalance. Stratified split (already configured) should help. Monitor F1-Score per class.\n"
    else:
        md += "Acceptable balance. Stratified split will maintain distribution.\n"

    md += f"2. **Text Quality**: Dataset includes machine-translated text (from cyberbullying_cleaned_indo). "
    md += "This may affect model performance on authentic Indonesian text.\n"
    md += f"3. **Preprocessing Pipeline**: Text still needs lowercase, cleaning, tokenization, stopword removal, "
    md += "and stemming before TF-IDF extraction (as defined in TRD).\n"
    md += f"4. **Train/Test Split**: Use 80/20 stratified split with random_state=42 as specified in TRD.\n"

    md += "\n---\n\n"
    md += "## Final Dataset Location\n\n"
    md += "```\n"
    md += "dataset/processed/final_dataset.csv\n"
    md += "dataset/cleaned/final_dataset.csv   (copy per TRD structure)\n"
    md += "```\n\n"
    md += "**Schema**: `text` (string), `label` (string)\n\n"
    md += f"**Ready for next phase**: ✅ Preprocessing → TF-IDF → Model Training\n"

    write_md(os.path.join(DOCS_DIR, "dataset-statistics.md"), md)
    return label_counts


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║       Dataset Preparation Pipeline — Cyberbullying NLP         ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print(f"║  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):>52} ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # Phase 1
    datasets_info = phase1_inspect()

    # Phase 2
    standardized = phase2_standardize()

    # Phase 3
    label_mappings = phase3_label_analysis(standardized)

    # Phase 4
    standardized = phase4_label_mapping(standardized)

    # Phase 5
    merged_rows, merged_labels = phase5_merge(standardized)

    # Phase 6
    final_rows, cleaning_stats = phase6_clean(merged_rows)

    # Phase 7 & 8
    label_counts = phase7_validate(final_rows, cleaning_stats)

    # Final summary
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Final dataset: {len(final_rows):,} rows, {len(label_counts)} classes")
    for label, count in label_counts.most_common():
        print(f"    {label}: {count:,}")
    print(f"\n  Output files:")
    print(f"    dataset/interim/cyberbullying_cleaned_indo_standardized.csv")
    print(f"    dataset/interim/hatespeech_abusive_standardized.csv")
    print(f"    dataset/interim/indotoxic2024_standardized.csv")
    print(f"    dataset/interim/label_mapping.csv")
    print(f"    dataset/interim/merged_dataset.csv")
    print(f"    dataset/processed/final_dataset.csv")
    print(f"    dataset/cleaned/final_dataset.csv")
    print(f"    docs/dataset-analysis.md")
    print(f"    docs/label-analysis.md")
    print(f"    docs/merge-report.md")
    print(f"    docs/dataset-statistics.md")


if __name__ == "__main__":
    main()
