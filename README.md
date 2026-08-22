# 🪸 Coral Change Detection

A computer vision system for detecting and classifying changes in coral colonies by comparing **BEFORE** and **CURRENT** images.

The system aims to identify four main types of coral changes:

- 🟢 **Growth**
- 🟡 **Damage**
- 🔴 **Bleaching**
- 🔵 **Recovery**

The project focuses on building a coral-aware change detection pipeline that can separate actual coral changes from background differences, lighting variations, and image alignment errors.

---

# 📌 1. Problem Statement

Monitoring coral health over time requires comparing images of the same coral colony captured at different points in time.

However, directly comparing two images is challenging because the images may differ in:

- Camera position.
- Image scale.
- Rotation.
- Lighting conditions.
- Color appearance.
- Background.
- Water conditions.
- Coral position inside the image.

A simple pixel-by-pixel comparison can therefore produce many false detections.
A conventional image difference method may interpret the background difference as coral damage or growth.

Therefore, the main objective is:

> **Align the images, identify the coral, and then analyze how the coral itself has changed.**

---

# 🎯 2. Main Challenges

The project is built around several major computer vision challenges.

## 2.1 Image Alignment

The BEFORE and CURRENT images may not be perfectly aligned.

Even a small shift can create artificial changes around coral boundaries.

```text
Small Camera Shift
        ↓
Misalignment
        ↓
False Growth / Damage
```

---

## 2.2 Coral vs. Background — White Coral Detection

One of the most challenging parts of the system is distinguishing the coral from the surrounding background, especially when the coral itself is white.

The main difficulty is:

```text
White Coral ≈ White Background
```

Simply detecting bright, low-saturation pixels produces many false positives.
Therefore, white coral detection requires additional contextual and background information.

---
## 2.3 Real Coral Changes vs. Image Artifacts

Not every difference between two images represents a biological change.

A detected difference may instead be caused by:

- Lighting.
- Camera movement.
- Alignment error.
- Background changes.
- Water conditions.
- Segmentation noise.

The system therefore needs spatial and structural constraints before classifying a region as growth, damage, bleaching, or recovery.

---

## 2.4 Coral Located Inside a Larger Image

The coral may occupy only a small portion of the image.

For example:

```text
Full Image
│
├── Background
├── Water
├── Equipment
└── Coral Colony
```

The system should eventually identify the coral colony explicitly instead of assuming that the entire image represents coral.

---

## 2.5 Underwater Conditions

Real underwater images introduce additional challenges such as:

- Blue/green color casts.
- Low visibility.
- Backscatter.
- Turbidity.
- Shadows.
- Uneven illumination.
- Different camera exposure.

These conditions can significantly change the appearance of the same coral.

---

# 🔬 3. Approaches We Considered

Several approaches were considered before selecting the final pipeline.

---

## Approach 1 — Direct Pixel Difference

The simplest solution was to align the images and calculate their pixel difference.

### Advantages

- Very simple.
- Fast.
- Easy to implement.
- Easy to visualize.

### Problems

It is highly sensitive to:

- Lighting changes.
- Camera movement.
- Color changes.
- Small alignment errors.
- Background changes.

A pixel difference does not understand whether a detected change belongs to the coral or to the background.

Therefore, this approach was not sufficient as the main detection method.

---

## Approach 2 — Feature-Based Alignment

The second approach was to align the images using local visual features.

We selected **SIFT (Scale-Invariant Feature Transform)** because it provides robustness to:

- Scale changes.
- Rotation.
- Moderate illumination changes.
- Local structural differences.

The alignment pipeline became:

```text
SIFT
  ↓
Feature Matching
  ↓
Ratio Test
  ↓
One-to-One Matching
  ↓
RANSAC
  ↓
Affine Transformation
```

This allows the CURRENT image to be transformed into the coordinate system of the BEFORE image.

However, SIFT + RANSAC alone was not enough because small remaining alignment errors could later appear as artificial coral growth or damage.

---

## Approach 3 — Color-Based Coral Segmentation

After alignment, another important question is:

> **Which pixels actually belong to the coral?**

Instead of treating the entire image as coral, we considered detecting coral-related regions using color information.

The current implementation focuses mainly on:

- 🌸 Pink coral.
- ⚪ White coral.

HSV color space was selected because it separates hue and saturation from brightness better than directly working in RGB.

The pink mask is generated using HSV thresholds followed by morphological cleaning.

However, white coral introduced an additional challenge because white coral and white background can have very similar pixel values.

---

## Approach 4 — Structural and Context-Aware Detection

To make the segmentation more reliable, we considered using more than color alone.

For white coral, the system combines:

1. HSV value and saturation.
2. RGB channel similarity.
3. Background estimation.
4. LAB color distance.
5. Brightness difference from the estimated background.
6. Morphological processing.
7. Spatial relationship with previously detected coral.

The goal was to move from:

```text
"Is this pixel white?"
```

to:

```text
"Is this white region likely to belong to the coral?"
```

This provides much stronger control over white-background false detections.

---

# 🧠 4. Final Selected Approach

After testing the different approaches, we selected a **multi-stage, coral-aware computer vision pipeline**.

The final pipeline combines image registration, coral segmentation, background modeling, and context-aware change detection.

```text
BEFORE + CURRENT
       │
       ▼
Image Preprocessing
       │
       ▼
Pink Coral Detection
       │
       ▼
Coral-Guided SIFT
       │
       ▼
Feature Matching
       │
       ▼
RANSAC Alignment
       │
       ▼
ECC Refinement
       │
       ▼
Final Image Alignment
       │
       ▼
┌─────────────────────────┐
│    Coral Segmentation   │
│                         │
│  Pink + White Coral     │
│  + Background Modeling  │
│  + Morphological Clean  │
└─────────────────────────┘
       │
       ▼
BEFORE / CURRENT Coral Masks
       │
       ▼
Context-Aware Comparison
       │
       ├──────────┬──────────┬───────────┐
       ▼          ▼          ▼           ▼
    Growth      Damage    Bleaching   Recovery
```

## Why This Approach?

The main reason for selecting this pipeline is that **each stage solves a different failure mode**.

- **SIFT + RANSAC** handles geometric misalignment.
- **ECC** refines small remaining alignment errors.
- **Pink segmentation** provides useful coral-related structure.
- **White segmentation** handles pale/white coral.
- **Background modeling** reduces false white detections.
- **Morphological processing** removes noise and small artifacts.
- **Coral masks** prevent the system from treating the entire image as coral.
- **Context-aware rules** distinguish actual changes from simple color differences.

The final design follows one important principle:

> **Do not detect change first. Detect the coral first, then analyze how the coral changed.**

---

# 🪸 5. Coral Masks

The final coral representation is created by combining the pink and white masks.

```text
BEFORE
  │
  ├── Pink Coral
  └── White Coral
         ↓
    BEFORE Coral Mask

CURRENT
  │
  ├── Pink Coral
  └── White Coral
         ↓
    CURRENT Coral Mask
```
# ⚠️ 6. Problems Encountered During Implementation

Building the pipeline revealed several important problems that required iterative improvements.

## 6.1 False White Background Detection

The first white segmentation approach detected large portions of the white background as coral.

```text
White Coral ≈ White Background
```

## 6.2 False Damage in Coral Gaps

Empty spaces between coral branches were sometimes classified as damage.

```text
Old Coral
    ↓
Natural Gap
    ↓
False Damage
```

## 6.3 False Growth

New pink regions were sometimes classified as growth even when they were unrelated to the original coral.

## 6.4 Alignment Errors Becoming Fake Changes

Small registration errors created artificial growth and damage around coral boundaries.

## 6.5 Weak Pink Detection

In some images, pink coral was too weak or changed significantly because of lighting and color conditions.

This reduced the number of reliable SIFT features.

## 6.6 Overlapping Change Categories

A region could potentially satisfy multiple change conditions.

For example:

```text
Growth
  +
Recovery
```

could occur in the same area.

## 6.7 Small Noisy Components

Segmentation produced small isolated regions that were not meaningful coral changes.

## 6.8 Fixed Thresholds

The current system relies on manually selected thresholds for:

- Pink HSV segmentation.
- White detection.
- Background distance.
- Morphological operations.
- Minimum change areas.

These values work for the current testing conditions but may not generalize perfectly to every image.

This became one of the main motivations for the next development stage.

---

# 📊 7. Results

The system produces several intermediate and final outputs for visual evaluation.

### SIFT Matches

![SIFT Matches](images/sift_matches.png)

### RANSAC Inliers

![RANSAC Inliers](images/ransac_inliers.png)

### Final Alignment

![Final Alignment](images/final_alignment.png)

### Coral Masks

![Coral Masks](images/coral_masks.png)

### Change Masks

![Change Masks](images/change_masks.png)

### Final Detection

![Final Result](images/final_result.png)

The final output reports the number of detected regions for each category:

# 🚧 8. Current Limitations

Although the current pipeline works on the tested image conditions, there are still important cases that need to be addressed before considering the system robust for general coral monitoring.

---

## 18.1 Underwater Image Conditions

The system still needs to be tested more extensively on real underwater images 
The current HSV thresholds may not generalize to all underwater conditions.

---

## 18.2 Coral Located Inside a Larger Image

The current pipeline is strongly dependent on detecting coral-related visual information.

A more general version should explicitly identify the coral object/colony first, especially when the coral occupies only a small part of the image.

```text
Full Image
│
├── Background
├── Equipment / PVC
├── Water
└── Coral Colony
        ↓
   Coral ROI / Segmentation
        ↓
   Change Detection
```

This would prevent unrelated background regions from participating in the change analysis.

---

## 18.3 Different Backgrounds

Another important case is when BEFORE and CURRENT images have significantly different backgrounds.

```text
BEFORE
Coral + Background A

        ↓

CURRENT
Coral + Background B
```

In this situation, background-based assumptions can become unreliable.

The next version should therefore become more **coral-centric** rather than relying heavily on the image border or global background appearance.

---

# 🚀 9. Future Improvements

The next development stage will focus on improving robustness and reducing dependency on fixed thresholds.

- 🌊 Improve underwater color correction.
- 🪸 Explicitly detect and segment the coral colony.
- 🏞️ Make the system robust to different backgrounds.
- 📍 Handle coral located anywhere inside the image.
- 🎨 Reduce dependency on fixed HSV thresholds.
- 🌈 Support a wider range of coral colors.
- 🧠 Investigate learning-based coral segmentation.
- 🌊 Validate the pipeline on real underwater image sequences.

---

# 🧰 10. Technologies

The project currently uses:

- **Python**
- **OpenCV**
- **NumPy**
- **Matplotlib**
- **SIFT**
- **RANSAC**
- **ECC**
- **HSV Color Segmentation**
- **LAB Color Space**
- **Morphological Image Processing**

---

> **The current version establishes the core computer-vision pipeline. The next stage is focused on making it robust enough for real-world underwater coral monitoring.**
