# Preprocessing Summary Report

**Generated**: 2026-07-15 21:12:04

---

## Overview

| Metric | Value |
|--------|-------|
| Total Rows (Input) | 41,556 |
| Processed Rows (Output) | 41,556 |
| Removed Empty Rows | 32 |
| Removed Duplicate Rows | 0 |
| Avg Text Length Before (words) | 34.47 |
| Avg Text Length After (words) | 21.47 |
| Processing Time | 3956.15 seconds |

---

## Active Preprocessing Steps

1. Case Folding
2. URL Removal
3. HTML Removal
4. Mention Removal
5. Hashtag Removal
6. Emoji Removal
7. Punctuation Removal
8. Number Removal
9. Stopword Removal
10. Stemming

---

## Pipeline Order

```
Raw Text → Case Folding → Text Cleaning → Tokenization
→ Stopword Removal → Stemming → Join Tokens
```

---

## Example Transformations

### Example 1

**Before**:
```
setiap orang adalah seorang gadis yang akan mengganggu saya di sekolah tinggi
```

**After**:
```
orang gadis ganggu sekolah
```

### Example 2

**Before**:
```
bahwa pos ab kpop stans pergi ke sekolah bersamasama dan semua orang mengatakan mereka akan menggertak tentara dalam jawaban dan aku seharusnya takut dari sebuah kpop stan mengapa
```

**After**:
```
pos ab kpop stans pergi sekolah bersamasama orang gertak tentara takut kpop stan
```

### Example 3

**Before**:
```
karena beberapa orang tidak ada yang lebih baik untuk dilakukan atau mereka adalah pengganggu di sekolah dan mereka tidak pernah outgrew bahwa
```

**After**:
```
orang ganggu sekolah outgrew
```

### Example 4

**Before**:
```
bro aku pelatih jv tahun lalu di skyline dan aku harus mengubah air menjadi anggur kami memenangkan pertandingan karena kami adalah tim super atletik dan bisa menggertak kalah dengan tim yang ahli yang lulus dan mengambil gambar yang bagus
```

**After**:
```
bro latih jv skyline ubah air anggur menang tanding tim super atletik gertak kalah tim ahli lulus ambil gambar bagus
```

### Example 5

**Before**:
```
wanitawanita ini benarbenar mengingatkan saya pada anak ayam sma dengan semua penindasan oleh satu diva dan semua orang mengadukaduk nya pada
```

**After**:
```
wanitawanita benarbenar anak ayam sma tindas diva orang mengadukaduk
```

---

## Notes

- Preprocessing was applied to the `text` column only.
- The `label` column was not modified.
- Results are saved in `text_clean` column.
- This dataset is ready for TF-IDF feature extraction.

---

*Report generated automatically by the preprocessing pipeline.*
