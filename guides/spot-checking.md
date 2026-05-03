# Spot-Check Labeling Guide — Tier 1 BERTopic Aspect Validation

> **For coding-agent delegation.** This document is the single source of truth for the spot-check labeling task. Do not improvise rules. Do not extend the schema. If a doc doesn't fit the rules below, mark it `borderline` and move on — don't invent new categories.

---

## Project context (read first)

**What this is:** Native-speaker validation of a guided BERTopic run (`nb_015_real_filtered`) on Philippine student feedback (Cebuano + Tagalog + English code-switched). The model assigned each document to a primary topic, which is mapped to one of 10 academic aspects (or `emergent`). Your job is to verify whether those aspect assignments are correct on a sample of documents.

**What this is NOT:** Refitting the model. Editing the seed lists. Tuning thresholds. Improving keywords. Those are out of scope for this task.

**Output:** A reviewed CSV with three new columns (`verdict`, `note`, `missed_secondary`) appended to the input CSV. Save to `runs/run_nb_015_real_filtered/spot_check_reviewed.csv`.

---

## Input file

**Path:** `runs/run_nb_015_real_filtered/spot_check_samples.csv`

**Columns:**
- `aspect` — the academic aspect the model assigned (one of 10 known aspects, or `emergent`)
- `secondary_aspect` — the secondary aspect if multi-topic, else blank
- `is_multi_topic` — True if doc has both primary and secondary topic
- `primary_topic` — topic ID (integer)
- `primary_confidence` — float, 0.0–1.0
- `secondary_topic` — topic ID as float (e.g., `1.0` means Topic 1; blank means no secondary)
- `text` — the student feedback text (Cebuano/Tagalog/English code-switched)

**Row count:** ~390 (up to 30 docs per aspect × 13 aspects). You don't need to label all rows — see "How many to label" below.

---

## The 10 aspects (taxonomy)

| Aspect | Definition |
|---|---|
| `teaching_clarity` | Whether students *understood* what was taught. Comprehension. |
| `teaching_methodology` | *How* the teacher taught. Methods, activities, structure, approach. |
| `instructor_attitude` | Teacher's demeanor: approachable, strict, kind, intimidating, etc. |
| `assessment_grading` | Exams, quizzes, grades, fairness of grading, exam difficulty. |
| `punctuality_attendance` | Teacher (or student) being late, absent, on time, attending class. |
| `responsiveness_support` | Teacher responding to questions, helping students, providing support. |
| `workload_pacing` | Amount of work, speed of coverage, deadlines, course load. |
| `real_world_relevance` | Practical/applied content, real-world examples, industry relevance. |
| `lms_digital_tools` | LMS use, digital tools, online platforms, tech in teaching. |
| `language_communication` | Language barriers, code-switching issues, communication style. |

Plus `emergent` — assigned by the model when no aspect fit above the similarity threshold (0.65).

---

## The schema (three columns to add)

### Column 1: `verdict` (required for every row you label)

Pick exactly one:

| Value | When to use |
|---|---|
| `correct` | The assigned `aspect` fits the `text`. A native speaker would agree. |
| `mislabel` | The assigned `aspect` is wrong, but a *different* known aspect fits better. |
| `noise` | The doc isn't substantive feedback. Greetings, "no comment", "10/10", holiday wishes, expressions like "amping permi". Doesn't belong in any aspect. |
| `borderline` | Could plausibly fit, or could fit another aspect equally well, or genuinely ambiguous. Use sparingly (target: <15% of labeled rows). |

### Column 2: `note` (optional, used selectively)

Free text but use **consistent strings** so they can be grep'd later. Use cases:

- When `verdict=mislabel`: write the aspect name it should have been. Example: `teaching_methodology`. One word, the correct aspect.
- When `verdict=noise`: brief reason. Examples: `null comment`, `gratitude only`, `rating only`, `greeting only`.
- When `verdict=borderline`: brief reason. Examples: `clarity_or_methodology`, `attitude_or_support`.
- When `verdict=correct`: leave blank, **except** if you spot the alignment pattern (see special rules below).

### Column 3: `missed_secondary` (optional, used selectively)

When the **primary** is correct but the doc *also* contains content from another aspect that the model didn't capture in `secondary_aspect`. Fill in the missed aspect's name.

Example: doc says "di kaayo mo sud nya gagmay hatag grado" (doesn't enter class, gives low grades). Model assigned `assessment_grading` (correct, the grading content is there). But the attendance content was missed. So:
- `verdict=correct`
- `note=` (blank or `mixed grading + attendance`)
- `missed_secondary=punctuality_attendance`

Leave blank when there's no missed secondary, or when the model already captured it in `secondary_aspect`.

---

## Quick reference table (memorize this)

| Situation | verdict | note | missed_secondary |
|---|---|---|---|
| Doc clearly about teaching style, mapped to `teaching_methodology` | `correct` | (blank) | (blank) |
| Doc clearly about grading, mapped to `punctuality_attendance` | `mislabel` | `assessment_grading` | (blank) |
| Doc says "thank you po", mapped anywhere | `noise` | `gratitude only` | (blank) |
| Doc says "n/a" or "no comment", mapped anywhere | `noise` | `null comment` | (blank) |
| Doc says "10/10 perfect", mapped anywhere | `noise` | `rating only` | (blank) |
| Doc says "amping permi, hi mam", mapped anywhere | `noise` | `greeting only` | (blank) |
| Doc could be teaching OR grading, genuinely ambiguous | `borderline` | `teaching_or_grading` | (blank) |
| Doc says "approachable but always late", mapped to `instructor_attitude` | `correct` | (blank — primary fits) | `punctuality_attendance` |
| Doc says "discussed X, exam was Y", mapped to `assessment_grading` | `correct` | `instruction_assessment_alignment` | (blank) |
| Doc says "discussed X, exam was Y", mapped to `teaching_methodology` | `mislabel` | `assessment_grading + alignment` | (blank) |
| Doc says "discussed X, exam was Y", mapped to `emergent` | `correct` | `alignment — gap aspect` | (blank) |
| Doc clearly about grading + attendance, mapped to grading only | `correct` | `mixed grading + attendance` | `punctuality_attendance` |

---

## Disambiguation rules for the hard pairs

### `teaching_clarity` vs `teaching_methodology`

**Test:** Could a student answer this comment with "yes/no, I did/didn't get it"?

- **Yes → `teaching_clarity`.** About comprehension/understanding.
- **No → `teaching_methodology`.** About method, structure, approach.

**Signal words for clarity:** *clear, understand, explain, makasabot, klaro, naintindihan, gets us, doesn't explain*. About whether the message was received.

**Signal words for methodology:** *discuss, lecture, group work, activity, ppt, video, demo, recitation, method, approach, way of teaching*. About what the teacher *does*.

**Default rule when stuck:** If a doc could plausibly be either, default to `teaching_methodology`. It's the broader, more permissive category. Reserve `teaching_clarity` for docs where comprehension is *explicitly* the point.

**Borderline cases:**

| Doc | Verdict |
|---|---|
| "He just reads from the PowerPoint" | `teaching_methodology` (criticizes technique) |
| "Good teaching" / "Maayo siya mu-teach" | `teaching_methodology` (generic praise) |
| "Clear and engaging discussion" | `teaching_methodology` (primary signal is "discussion") |
| "He explains everything step by step" | `teaching_clarity` if point is comprehension; `teaching_methodology` if point is structure |
| "Hinay mu-discuss / speaks softly" | `mislabel` → `emergent`/audibility (neither clarity nor methodology) |

### `instructor_attitude` vs `responsiveness_support`

- `instructor_attitude` = how the teacher *is* (approachable, strict, kind, intimidating)
- `responsiveness_support` = how the teacher *acts when students need help* (answers questions, helps with problems, provides feedback)

If the doc describes a personality trait → `instructor_attitude`.
If the doc describes the teacher *doing something* in response to a student need → `responsiveness_support`.

"Approachable" alone is `instructor_attitude` (a trait).
"Approachable and answers all our questions" leans `responsiveness_support` (an action), with `missed_secondary=instructor_attitude`.

### `assessment_grading` vs `workload_pacing`

- `assessment_grading` = exams, quizzes, grades, fairness, exam content
- `workload_pacing` = amount of work, speed of coverage, deadlines

"Too many quizzes" → `assessment_grading` (about quizzes specifically).
"Too much work" → `workload_pacing` (about volume).
"Rushed through the material" → `workload_pacing` (about pacing).
"Hard exams" → `assessment_grading`.

### Special pattern: instruction-assessment alignment

Pattern: "Prof discussed topic X, but the exam was all about topic Y."

This is a real, recurring complaint type that doesn't have a dedicated aspect. Rule:

- Primary mapping should be `assessment_grading` (the complaint is fundamentally about the exam).
- Always add `note=instruction_assessment_alignment` (use this exact string).
- If model put it in `teaching_methodology` or elsewhere → `mislabel`, note `assessment_grading + alignment`.
- If model put it in `emergent` → `correct`, note `alignment — gap aspect`.

This is a finding-tracking note: a high count of `instruction_assessment_alignment` notes suggests the taxonomy needs a new aspect in a future revision.

---

## Reading order for each row

1. Read `text` first. Form your own opinion of what the doc is about.
2. Look at `aspect`. Compare to your opinion.
3. If they agree → `verdict=correct`, leave note blank, move on.
4. If they disagree → check `primary_confidence`. Low confidence (<0.30) means the model was hedging; the mismatch is less alarming. High confidence (>0.50) means the model was confidently wrong, which is a more serious finding.
5. Decide: is this `mislabel` (different known aspect fits), `noise` (not real feedback), or `borderline` (genuinely ambiguous)?
6. If `is_multi_topic=True`, glance at `secondary_aspect` and decide if the pairing makes sense. (Doesn't change `verdict` — that's primary-only.)
7. Whether `verdict` is `correct` or `mislabel`, ask: is there a *missed* secondary aspect in the text? If yes → fill `missed_secondary`.
8. Move to the next row.

**Don't worry about `aspect_sim`** — it's a topic-level score, identical across all rows of the same primary topic. Tells you nothing new at the row level.

---

## How many to label

**Target: 10–15 rows per aspect, ~150 rows total. Not all 390.**

Patterns emerge fast. If 8 of 10 docs in `instructor_attitude` are `correct`, stop labeling that aspect. If 7 of 10 docs in `punctuality_attendance` are `mislabel` → `teaching_methodology`, the seed-bleed pattern is confirmed and 20 more labels won't change the conclusion.

**Prioritization (high-suspicion aspects first):**

1. **`punctuality_attendance`** — Topic 1 (833 docs of "good sometimes, hes good") and Topic 11 ("amping permi, hi mam") are very likely to be mislabeled. Read these topics' docs early.
2. **`responsiveness_support`** — risk of overlap with `instructor_attitude`.
3. **`workload_pacing`** — Topics 4, 10, 15 still mapped here; verify they're actually about workload/pacing.
4. **The `emergent` topics** — confirm they're genuinely off-axis from the taxonomy and not "this aspect should exist but isn't in the seed list." Pay attention to Topic 4 ("learned many, learned lot") — possible candidate for a `learning_outcomes` aspect in future revisions.
5. **High-confidence aspects last** (`teaching_methodology`, `instructor_attitude`, `teaching_clarity`, `assessment_grading`, `language_communication`) — label 8–10 each just to confirm.

---

## Stopping rules (when to stop labeling an aspect)

Stop labeling an aspect when **any** of these triggers:

- 10 consecutive rows in that aspect labeled `correct` → aspect is clean, move on
- 7 of 10 rows labeled `mislabel` to the same target aspect → seed-bleed confirmed, no more data needed
- 7 of 10 rows labeled `noise` → aspect is absorbing junk, no more data needed
- You've labeled 15 rows in that aspect regardless of distribution → diminishing returns

Don't grind through 30 rows just because they're there. The output is a *characterization* of each aspect's quality, not an exhaustive census.

---

## Forbidden actions (do not do these)

- ❌ Don't add columns beyond `verdict`, `note`, `missed_secondary`.
- ❌ Don't change values in existing columns (`aspect`, `text`, `primary_confidence`, etc.).
- ❌ Don't rewrite the docs to be "cleaner" — they're code-switched on purpose.
- ❌ Don't merge or split aspects on your own — taxonomy revision is out of scope.
- ❌ Don't refit, re-cluster, or re-tune anything. This is a *labeling* task only.
- ❌ Don't paraphrase the canonical note strings. `instruction_assessment_alignment` stays as `instruction_assessment_alignment`. `gratitude only` stays as `gratitude only`. Don't write "thank you note" or "expression of gratitude" or any variant.
- ❌ Don't label rows you can't read. If a doc is too short, too garbled, or in a script you can't parse, mark `borderline` with note `unreadable` and move on.

---

## After labeling: deliverables

### 1. Save the reviewed CSV

Path: `runs/run_nb_015_real_filtered/spot_check_reviewed.csv`

Same columns as input plus `verdict`, `note`, `missed_secondary` at the end.

### 2. Generate per-aspect summary stats

Run this in a notebook cell or standalone script:

```python
import pandas as pd
reviewed = pd.read_csv("runs/run_nb_015_real_filtered/spot_check_reviewed.csv")

# Only count labeled rows
labeled = reviewed[reviewed["verdict"].notna() & (reviewed["verdict"] != "")]

per_aspect = labeled.groupby("aspect").agg(
    n_reviewed=("verdict", "count"),
    n_correct=("verdict", lambda v: (v == "correct").sum()),
    n_mislabel=("verdict", lambda v: (v == "mislabel").sum()),
    n_noise=("verdict", lambda v: (v == "noise").sum()),
    n_borderline=("verdict", lambda v: (v == "borderline").sum()),
)
per_aspect["pct_correct"] = (per_aspect["n_correct"] / per_aspect["n_reviewed"]).round(2)
per_aspect = per_aspect.sort_values("pct_correct", ascending=False)
print(per_aspect)

# Also: top mislabel targets per aspect
mislabels = labeled[labeled["verdict"] == "mislabel"]
print("\nMislabel targets:")
print(mislabels.groupby(["aspect", "note"]).size().sort_values(ascending=False))

# And: count of alignment-pattern notes
alignment_count = labeled["note"].str.contains("alignment", na=False).sum()
print(f"\nalignment pattern: {alignment_count} docs")

# And: missed secondary tally
missed = labeled[labeled["missed_secondary"].notna() & (labeled["missed_secondary"] != "")]
print(f"\nmissed secondary: {len(missed)} docs")
print(missed.groupby(["aspect", "missed_secondary"]).size().sort_values(ascending=False))
```

Save the printed output to `runs/run_nb_015_real_filtered/spot_check_summary.txt`.

### 3. Bucket the aspects

Sort each labeled aspect into one of three buckets based on `pct_correct`:

- **Validated (≥80% correct):** Aspect is working. No action needed.
- **Soft mismatch (50–80% correct):** Aspect partially working. Document the bleed pattern (which aspects it gets confused with). Candidate for taxonomy revision.
- **Broken (<50% correct):** Real problem. Two response paths:
  - If caused by **noise topics being mapped** (high `noise` count): document as a known limitation.
  - If caused by **seed-list bleed** (high `mislabel` count to the same target aspect): seed list needs revision before refitting in `nb_016`.

### 4. Final report (short, ~3 paragraphs)

Save to `runs/run_nb_015_real_filtered/spot_check_findings.md`. Structure:

**Para 1 — Summary.** Total rows labeled. Aspects in each bucket (validated / soft mismatch / broken). Bottom-line answer: is `nb_015_real_filtered` a defensible Tier 1 baseline as-is, or does it need a `nb_016` refit?

**Para 2 — Per-aspect findings.** For each aspect: pct_correct, top mislabel target if any, dominant noise pattern if any. Be brief — one line per aspect.

**Para 3 — Taxonomy gap signals.** Count of `instruction_assessment_alignment` notes. Count and distribution of `missed_secondary`. Any aspect candidates that emerged from the `emergent` topics that should be considered for a future taxonomy revision (e.g., `learning_outcomes`, `audibility`).

---

## Glossary (single-source canonical strings)

Use these **exact strings** in the `note` column when applicable:

- `instruction_assessment_alignment`
- `assessment_grading + alignment`
- `alignment — gap aspect` (note the em-dash)
- `gratitude only`
- `null comment`
- `rating only`
- `greeting only`
- `unreadable`
- `clarity_or_methodology`
- `attitude_or_support`
- `mixed grading + attendance` (and similar `mixed X + Y`)
- For `mislabel` notes: just the target aspect name, lowercase, exact form (e.g., `teaching_methodology`)

Don't invent new strings unless the situation genuinely doesn't fit any above. If you do invent one, use snake_case and document it at the top of the report.

---

*Document version 1.0. Tied to run `nb_015_real_filtered`. If the underlying run is regenerated (different RUN_ID, different aspect mapping, different thresholds), confirm the aspect list and seed taxonomy haven't drifted before applying these rules.*