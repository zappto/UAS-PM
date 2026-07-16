# Label Analysis Report

**Generated**: 2026-07-15 03:21:49

---

## Label Mapping Table

| Original Label | Dataset | Meaning | Recommended Final Label |
|----------------|---------|---------|------------------------|
| `age` | cyberbullying_cleaned_indo | Cyberbullying based on age (ageism, age-based harassment) | **harassment** |
| `ethnicity` | cyberbullying_cleaned_indo | Cyberbullying based on ethnicity/race | **hate_speech** |
| `gender` | cyberbullying_cleaned_indo | Cyberbullying based on gender identity | **harassment** |
| `not_cyberbullying` | cyberbullying_cleaned_indo | Text that does not contain cyberbullying | **normal** |
| `other_cyberbullying` | cyberbullying_cleaned_indo | Other forms of cyberbullying not categorized above | **insult** |
| `religion` | cyberbullying_cleaned_indo | Cyberbullying based on religion | **hate_speech** |
| `HS=0, Abusive=0` | hatespeech_abusive | No hate speech and no abusive language — normal text | **normal** |
| `HS=0, Abusive=1` | hatespeech_abusive | Abusive language without hate speech — insult/profanity | **insult** |
| `HS=1, HS_Religion=1 or HS_Race=1` | hatespeech_abusive | Hate speech targeting religion or race | **hate_speech** |
| `HS=1, HS_Physical=1` | hatespeech_abusive | Hate speech with physical targeting — threat/intimidation | **threat** |
| `HS=1, HS_Gender=1` | hatespeech_abusive | Hate speech targeting gender | **harassment** |
| `HS=1, HS_Other=1` | hatespeech_abusive | Other forms of hate speech | **hate_speech** |
| `threat_incitement_to_violence=1` | indotoxic2024 | Text contains threats or incitement to violence (highest priority) | **threat** |
| `identity_attack=1` | indotoxic2024 | Text attacks based on identity (race, religion, etc.) | **hate_speech** |
| `insults=1` | indotoxic2024 | Text contains insults/personal attacks | **insult** |
| `sexually_explicit=1` | indotoxic2024 | Text contains sexually explicit content | **sexually_explicit** |
| `profanity_obscenity=1` | indotoxic2024 | Text contains profanity/obscenity without other specific category | **harassment** |
| `all annotations=0` | indotoxic2024 | No toxic annotations — normal text | **normal** |

---

## Final Label Schema

| Final Label | Description | Source Categories |
|-------------|-------------|-------------------|
| `normal` | Text does not contain cyberbullying | not_cyberbullying, HS=0+Abusive=0, all annotations=0 |
| `insult` | Insulting, demeaning, or abusive language | other_cyberbullying, HS=0+Abusive=1, insults=1 |
| `hate_speech` | Identity-based hate speech (religion, race, ethnicity) | ethnicity, religion, HS_Religion, HS_Race, HS_Other, identity_attack=1 |
| `threat` | Threats, intimidation, or incitement to violence | HS_Physical, threat_incitement=1 |
| `harassment` | General harassment (age, gender, profanity-based) | age, gender, HS_Gender, profanity_obscenity=1 |
| `sexually_explicit` | Sexually explicit content | sexually_explicit=1 |

---

## Mapping Justifications

### `age` → `harassment`

Age-based cyberbullying (ageism) is a form of general harassment. It does not fit hate_speech (which focuses on immutable identity characteristics like race/religion in Indonesian legal context) nor insult (which is non-targeted abusive language).

### `ethnicity` → `hate_speech`

Ethnicity-based cyberbullying is targeting someone based on their ethnic/racial background. This aligns with the definition of hate speech (ujaran kebencian) in Indonesian context.

### `gender` → `harassment`

Gender-based cyberbullying is a form of harassment. In the DRD, harassment is defined as 'pelecehan verbal' (verbal harassment), which includes gender-based targeting.

### `not_cyberbullying` → `normal`

Directly maps to the non-cyberbullying class.

### `other_cyberbullying` → `insult`

The 'other' category from the cyberbullying dataset captures generic cyberbullying that doesn't fall into specific categories. This most closely aligns with 'insult' (general abusive language).

### `religion` → `hate_speech`

Religion-based cyberbullying targets religious identity. This is hate speech (ujaran kebencian) in Indonesian legal and social context.

### `HS=1, HS_Physical=1` → `threat`

Physical targeting in hate speech context implies threats or intimidation against physical safety.

### `sexually_explicit=1` → `sexually_explicit`

Sexually explicit content is kept as its own category as it represents a distinct type of cyberbullying.

---

## Notes

- The `sexually_explicit` class may have very few samples. If it has fewer than 50 samples, it may need to be merged into `harassment` during model training to avoid class imbalance issues.
- All mappings are based on semantic equivalence between source labels and the target schema.
- No labels were fabricated; all derive from existing annotations.
