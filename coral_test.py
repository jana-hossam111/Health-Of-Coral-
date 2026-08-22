import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

BEFORE_PATH = r"C:\Users\Hadeel\Downloads\coral_project\before.png"
CURRENT_PATH = r"C:\Users\Hadeel\Downloads\coral1.jpg"

TARGET_WIDTH = 1200


# ============================================================
# SIFT / ALIGNMENT SETTINGS
# ============================================================

SIFT_FEATURES = 3000
SIFT_RATIO = 0.78

RANSAC_REPROJ_THRESHOLD = 6.0
MIN_RANSAC_INLIERS = 3
MIN_INLIER_RATIO = 0.20

ECC_ITERATIONS = 150
ECC_EPSILON = 1e-6


# ============================================================
# PINK DETECTION RELIABILITY
# ============================================================

MIN_PINK_PIXELS_FOR_RELIABLE_SIFT = 3000

# Maximum image fraction used when pink detection is unreliable.
FALLBACK_SEARCH_AREA_FRACTION = 0.55


# ============================================================
# PINK SEGMENTATION
# ============================================================

LOWER_PINK = np.array([140, 35, 25])
UPPER_PINK = np.array([179, 255, 255])

PINK_CLEAN_SIZE = 5


# ============================================================
# WHITE SEGMENTATION
# ============================================================

WHITE_MIN_VALUE = 120
WHITE_MAX_SATURATION = 35
WHITE_MAX_CHANNEL_SPREAD = 45

# Controlled gap closing.
WHITE_FILL_CLOSE_SIZE = 15

# Minimum contour/component sizes.
WHITE_FILL_MIN_CONTOUR = 250
WHITE_MIN_COMPONENT = 500

# Background estimation.
BACKGROUND_BORDER_FRACTION = 0.06
WHITE_BG_DISTANCE = 22.0
WHITE_BG_LIFT = 15.0

# Maximum expansion used to recover darker white interiors.
WHITE_RECOVER_DILATE = 11


# ============================================================
# ALIGNMENT STRUCTURE
# ============================================================

ALIGN_MASK_DILATE_SIZE = 121


# ============================================================
# CHANGE DETECTION
# ============================================================

# -------------------------
# Growth
# -------------------------

GROWTH_NEIGHBOR_SIZE = 101

# Small tolerance for alignment errors.
GROWTH_ALIGNMENT_TOLERANCE = 12

# Minimum distance from old coral for real growth.
GROWTH_MIN_DISTANCE = 12


# -------------------------
# Damage
# -------------------------

DAMAGE_CORE_SIZE = 31


# -------------------------
# General
# -------------------------

CHANGE_OPEN_SIZE = 7
MIN_CHANGE_AREA = 1500


# ============================================================
# DEBUG / VISUALIZATION
# ============================================================

SHOW_SIFT_MATCHES = True
SHOW_ALIGNMENT = True
SHOW_COLOR_MASKS = True
SHOW_CHANGE_MASKS = True


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def resize_keep_aspect(image, target_width):
    """
    Resize an image to the target width while preserving
    its original aspect ratio.
    """
    height, width = image.shape[:2]

    scale = target_width / float(width)

    new_width = target_width
    new_height = int(round(height * scale))

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_CUBIC
    )


def add_central_region(mask, image_shape, area_fraction=0.55):
    """
    Add a central search region covering approximately
    `area_fraction` of the image.

    The square-root calculation converts the desired
    area fraction into an appropriate margin.
    """
    height, width = image_shape[:2]

    margin_fraction = (
        1 - np.sqrt(area_fraction)
    ) / 2

    margin_x = int(width * margin_fraction)
    margin_y = int(height * margin_fraction)

    central = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    central[
        margin_y:height - margin_y,
        margin_x:width - margin_x
    ] = 255

    return cv2.bitwise_or(mask, central)


def remove_small_components(mask, min_area):
    """
    Remove connected components smaller than min_area.
    """
    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    cleaned = np.zeros_like(mask)

    for label in range(1, num_labels):
        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        if area >= min_area:
            cleaned[labels == label] = 255

    return cleaned


def get_change_areas(mask, min_area):
    """
    Extract contours representing valid change areas.
    """
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    areas = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        x, y, width, height = (
            cv2.boundingRect(contour)
        )

        areas.append({
            "contour": contour,
            "area": area,
            "bbox": (x, y, width, height)
        })

    return areas


# ============================================================
# PINK MASK
# ============================================================

def build_pink_mask(image):
    """
    Detect pink regions using HSV thresholding
    followed by morphological opening.
    """
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    pink_raw = cv2.inRange(
        hsv,
        LOWER_PINK,
        UPPER_PINK
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (PINK_CLEAN_SIZE, PINK_CLEAN_SIZE)
    )

    pink_clean = cv2.morphologyEx(
        pink_raw,
        cv2.MORPH_OPEN,
        kernel
    )

    return pink_clean


# ============================================================
# SIFT REGION
# ============================================================

def build_sift_region(pink_mask):
    """
    Build the primary SIFT search region around detected pink coral.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (121, 121)
    )

    return cv2.dilate(
        pink_mask,
        kernel,
        iterations=1
    )


def apply_sift_fallback(
    sift_region,
    pink_pixel_count,
    image_shape,
    image_name
):
    """
    If pink detection is unreliable, add a restricted
    central fallback search region.
    """

    if pink_pixel_count >= MIN_PINK_PIXELS_FOR_RELIABLE_SIFT:
        return sift_region

    print()
    print(
        f"!! Warning: Pink detection is very weak "
        f"in {image_name}."
    )

    print(
        "!! The color/lighting may be different "
        "from the expected conditions."
    )

    print(
        "!! Using a restricted fallback SIFT region."
    )

    print(
        "!! Alignment may be unreliable. "
        "Review the SIFT MATCHES carefully."
    )

    return add_central_region(
        sift_region,
        image_shape,
        area_fraction=FALLBACK_SEARCH_AREA_FRACTION
    )


# ============================================================
# BUILD ALIGNMENT STRUCTURE
# ============================================================

def build_alignment_structure(
    pink_mask,
    image
):
    """
    Build the structural representation used by ECC.
    """

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            ALIGN_MASK_DILATE_SIZE,
            ALIGN_MASK_DILATE_SIZE
        )
    )

    support = cv2.dilate(
        pink_mask,
        kernel,
        iterations=1
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    structural = cv2.bitwise_and(
        gray,
        support
    )

    structural = cv2.addWeighted(
        structural,
        0.35,
        cv2.GaussianBlur(
            pink_mask,
            (9, 9),
            0
        ),
        0.65,
        0
    )

    return structural.astype(np.float32)


# ============================================================
# ECC REFINEMENT
# ============================================================

def refine_with_ecc(
    template_image,
    input_image,
    template_pink,
    input_pink,
    initial_transform
):
    """
    Refine the initial SIFT/RANSAC transformation using ECC.
    """

    template_align = build_alignment_structure(
        template_pink,
        template_image
    )

    input_align = build_alignment_structure(
        input_pink,
        input_image
    )

    ecc_matrix = cv2.invertAffineTransform(
        initial_transform
    ).astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS |
        cv2.TERM_CRITERIA_COUNT,
        ECC_ITERATIONS,
        ECC_EPSILON
    )

    try:

        correlation, ecc_matrix = (
            cv2.findTransformECC(
                template_align,
                input_align,
                ecc_matrix,
                cv2.MOTION_AFFINE,
                criteria,
                None,
                5
            )
        )

        refined_transform = (
            cv2.invertAffineTransform(
                ecc_matrix
            ).astype(np.float32)
        )

        return (
            refined_transform,
            correlation,
            True
        )

    except cv2.error as error:

        print()
        print("WARNING: ECC refinement failed.")
        print(str(error))

        return (
            initial_transform,
            None,
            False
        )


# ============================================================
# WHITE CANDIDATE
# ============================================================

def build_white_candidate(image):
    """
    Detect potential white regions using HSV saturation/value
    and RGB channel spread.
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    _, saturation, value = cv2.split(hsv)

    blue, green, red = cv2.split(image)

    blue16 = blue.astype(np.int16)
    green16 = green.astype(np.int16)
    red16 = red.astype(np.int16)

    max_channel = np.maximum(
        np.maximum(red16, green16),
        blue16
    )

    min_channel = np.minimum(
        np.minimum(red16, green16),
        blue16
    )

    channel_spread = (
        max_channel - min_channel
    )

    white_condition = (
        (value >= WHITE_MIN_VALUE)
        &
        (saturation <= WHITE_MAX_SATURATION)
        &
        (channel_spread <= WHITE_MAX_CHANNEL_SPREAD)
    )

    return (
        white_condition.astype(np.uint8) * 255
    )


# ============================================================
# BACKGROUND MODEL
# ============================================================

def build_background_mask(image):
    """
    Estimate the background from image borders and identify
    pixels sufficiently different and brighter than the background.
    """

    height, width = image.shape[:2]

    border = max(
        10,
        int(
            round(
                min(height, width)
                * BACKGROUND_BORDER_FRACTION
            )
        )
    )

    samples = np.concatenate(
        [
            image[:border].reshape(-1, 3),
            image[-border:].reshape(-1, 3),
            image[:, :border].reshape(-1, 3),
            image[:, -border:].reshape(-1, 3)
        ],
        axis=0
    )

    samples_lab = cv2.cvtColor(
        samples.reshape(-1, 1, 3),
        cv2.COLOR_BGR2LAB
    ).reshape(-1, 3).astype(np.float32)

    background_lab = np.median(
        samples_lab,
        axis=0
    )

    image_lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    ).astype(np.float32)

    difference = (
        image_lab
        - background_lab.reshape(1, 1, 3)
    )

    distance = np.sqrt(
        np.sum(
            difference * difference,
            axis=2
        )
    )

    brightness_lift = (
        image_lab[:, :, 0]
        - background_lab[0]
    )

    foreground = (
        (distance >= WHITE_BG_DISTANCE)
        &
        (brightness_lift >= WHITE_BG_LIFT)
    )

    return (
        foreground.astype(np.uint8) * 255
    )


# ============================================================
# RECOVER WHITE INTERIORS
# ============================================================

def recover_white_interiors(
    white_mask,
    image
):
    """
    Recover darker interiors of detected white regions.
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    _, saturation, value = cv2.split(hsv)

    interior_candidate = (
        (saturation <= 48)
        &
        (value >= 75)
    ).astype(np.uint8) * 255

    recover_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            WHITE_RECOVER_DILATE,
            WHITE_RECOVER_DILATE
        )
    )

    expanded_white = cv2.dilate(
        white_mask,
        recover_kernel,
        iterations=1
    )

    recovered = cv2.bitwise_and(
        expanded_white,
        interior_candidate
    )

    pink_mask = cv2.inRange(
        hsv,
        LOWER_PINK,
        UPPER_PINK
    )

    recovered = cv2.bitwise_and(
        recovered,
        cv2.bitwise_not(pink_mask)
    )

    return recovered


# ============================================================
# STEP 1 — LOAD IMAGES
# ============================================================

before = cv2.imread(BEFORE_PATH)
current = cv2.imread(CURRENT_PATH)

if before is None:
    raise FileNotFoundError(
        f"BEFORE image not found:\n{BEFORE_PATH}"
    )

if current is None:
    raise FileNotFoundError(
        f"CURRENT image not found:\n{CURRENT_PATH}"
    )

print("Before original :", before.shape)
print("Current original:", current.shape)


# ============================================================
# STEP 2 — RESIZE WITHOUT DISTORTING
# ============================================================

before_r = resize_keep_aspect(
    before,
    TARGET_WIDTH
)

current_r = resize_keep_aspect(
    current,
    TARGET_WIDTH
)

print()
print("Before resized :", before_r.shape)
print("Current resized:", current_r.shape)


# ============================================================
# STEP 3 — INITIAL PINK DETECTION
# ============================================================

pink_before = build_pink_mask(before_r)
pink_current = build_pink_mask(current_r)


# ============================================================
# STEP 4 — BUILD SIFT REGIONS
# ============================================================

sift_region_before = build_sift_region(
    pink_before
)

sift_region_current = build_sift_region(
    pink_current
)

pink_pixel_count_before = int(
    np.sum(pink_before > 0)
)

pink_pixel_count_current = int(
    np.sum(pink_current > 0)
)

print()
print("========== PINK DETECTION RELIABILITY ==========")

print(
    "Pink pixels (BEFORE) :",
    pink_pixel_count_before
)

print(
    "Pink pixels (CURRENT):",
    pink_pixel_count_current
)

sift_region_before = apply_sift_fallback(
    sift_region_before,
    pink_pixel_count_before,
    before_r.shape,
    "BEFORE"
)

sift_region_current = apply_sift_fallback(
    sift_region_current,
    pink_pixel_count_current,
    current_r.shape,
    "CURRENT"
)


# ============================================================
# STEP 5 — GRAYSCALE + CLAHE
# ============================================================

gray_before = cv2.cvtColor(
    before_r,
    cv2.COLOR_BGR2GRAY
)

gray_current = cv2.cvtColor(
    current_r,
    cv2.COLOR_BGR2GRAY
)

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

gray_before_sift = clahe.apply(
    gray_before
)

gray_current_sift = clahe.apply(
    gray_current
)


# ============================================================
# STEP 6 — SIFT
# ============================================================

sift = cv2.SIFT_create(
    nfeatures=SIFT_FEATURES,
    contrastThreshold=0.025,
    edgeThreshold=10,
    sigma=1.6
)

keypoints_before, descriptors_before = (
    sift.detectAndCompute(
        gray_before_sift,
        sift_region_before
    )
)

keypoints_current, descriptors_current = (
    sift.detectAndCompute(
        gray_current_sift,
        sift_region_current
    )
)

if descriptors_before is None:
    raise ValueError(
        "No SIFT descriptors found in BEFORE image."
    )

if descriptors_current is None:
    raise ValueError(
        "No SIFT descriptors found in CURRENT image."
    )

print()
print("========== SIFT ==========")

print(
    "Before keypoints :",
    len(keypoints_before)
)

print(
    "Current keypoints:",
    len(keypoints_current)
)


# ============================================================
# STEP 7 — MATCHING
# ============================================================

bf = cv2.BFMatcher(
    cv2.NORM_L2,
    crossCheck=False
)

matches_current_to_before = bf.knnMatch(
    descriptors_current,
    descriptors_before,
    k=2
)

ratio_matches = []

for pair in matches_current_to_before:

    if len(pair) < 2:
        continue

    match, second_match = pair

    if match.distance < (
        SIFT_RATIO * second_match.distance
    ):
        ratio_matches.append(match)

ratio_matches = sorted(
    ratio_matches,
    key=lambda match: match.distance
)


# ============================================================
# STEP 8 — ONE-TO-ONE MATCHING
# ============================================================

unique_matches = []

used_before_indices = set()

for match in ratio_matches:

    if match.trainIdx in used_before_indices:
        continue

    used_before_indices.add(
        match.trainIdx
    )

    unique_matches.append(match)

print()
print("========== MATCHING ==========")

print(
    "Ratio-test matches:",
    len(ratio_matches)
)

print(
    "One-to-one matches:",
    len(unique_matches)
)

if len(unique_matches) < 2:
    raise ValueError(
        "Not enough SIFT matches to estimate alignment."
    )


# ============================================================
# STEP 9 — DISPLAY SIFT MATCHES
# ============================================================

if SHOW_SIFT_MATCHES:

    match_display = cv2.drawMatches(
        current_r,
        keypoints_current,
        before_r,
        keypoints_before,
        unique_matches[:150],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    plt.figure(figsize=(20, 9))

    plt.imshow(
        cv2.cvtColor(
            match_display,
            cv2.COLOR_BGR2RGB
        )
    )

    plt.title(
        f"SIFT MATCHES — {len(unique_matches)}"
    )

    plt.axis("off")
    plt.tight_layout()
    plt.show()


# ============================================================
# STEP 10 — RANSAC
# ============================================================

src_points = np.float32([
    keypoints_current[match.queryIdx].pt
    for match in unique_matches
])

dst_points = np.float32([
    keypoints_before[match.trainIdx].pt
    for match in unique_matches
])

M, inlier_mask = cv2.estimateAffinePartial2D(
    src_points,
    dst_points,
    method=cv2.RANSAC,
    ransacReprojThreshold=RANSAC_REPROJ_THRESHOLD,
    maxIters=20000,
    confidence=0.995,
    refineIters=100
)

if M is None:
    raise ValueError(
        "Could not calculate alignment transform."
    )

inlier_flags = (
    inlier_mask.ravel().astype(bool)
)

inlier_count = int(
    np.sum(inlier_flags)
)

total_matches = len(unique_matches)

inlier_ratio = (
    inlier_count
    / max(total_matches, 1)
)

print()
print("========== SIMILARITY ALIGNMENT ==========")

print("Transform matrix:")
print(M)

print(
    "Inliers:",
    inlier_count,
    "/",
    total_matches
)

print(
    "Inlier ratio:",
    round(inlier_ratio, 3)
)


# ============================================================
# STEP 11 — RANSAC INLIERS
# ============================================================

inlier_matches = [
    match
    for index, match in enumerate(unique_matches)
    if inlier_flags[index]
]

if SHOW_SIFT_MATCHES:

    inlier_display = cv2.drawMatches(
        current_r,
        keypoints_current,
        before_r,
        keypoints_before,
        inlier_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    plt.figure(figsize=(20, 9))

    plt.imshow(
        cv2.cvtColor(
            inlier_display,
            cv2.COLOR_BGR2RGB
        )
    )

    plt.title(
        f"RANSAC INLIERS — "
        f"{inlier_count}/{total_matches}"
    )

    plt.axis("off")
    plt.tight_layout()
    plt.show()


# ============================================================
# STEP 12 — ECC REFINEMENT
# ============================================================

M_refined, ecc_correlation, ecc_success = (
    refine_with_ecc(
        before_r,
        current_r,
        pink_before,
        pink_current,
        M
    )
)

if ecc_success:

    print()
    print("========== ECC REFINEMENT ==========")

    print(
        "ECC correlation:",
        round(
            float(ecc_correlation),
            6
        )
    )

    print("Refined transform:")
    print(M_refined)

else:

    print(
        "Using SIFT/RANSAC transform."
    )


M_final = M_refined


# ============================================================
# STEP 13 — FINAL ALIGNMENT
# ============================================================

before_height, before_width = (
    before_r.shape[:2]
)

current_aligned = cv2.warpAffine(
    current_r,
    M_final,
    (before_width, before_height),
    flags=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=(0, 0, 0)
)


# ============================================================
# STEP 14 — ALIGNMENT CHECK
# ============================================================

if SHOW_ALIGNMENT:

    plt.figure(figsize=(18, 7))

    plt.subplot(1, 2, 1)

    plt.imshow(
        cv2.cvtColor(
            before_r,
            cv2.COLOR_BGR2RGB
        )
    )

    plt.title("BEFORE")
    plt.axis("off")

    plt.subplot(1, 2, 2)

    plt.imshow(
        cv2.cvtColor(
            current_aligned,
            cv2.COLOR_BGR2RGB
        )
    )

    plt.title(
        "CURRENT — FINAL ALIGNED"
    )

    plt.axis("off")

    plt.tight_layout()
    plt.show()

    overlay = cv2.addWeighted(
        before_r,
        0.5,
        current_aligned,
        0.5,
        0
    )

    plt.figure(figsize=(15, 9))

    plt.imshow(
        cv2.cvtColor(
            overlay,
            cv2.COLOR_BGR2RGB
        )
    )

    plt.title(
        "FINAL ALIGNMENT OVERLAY"
    )

    plt.axis("off")
    plt.tight_layout()
    plt.show()


# ============================================================
# STEP 15 — HSV AFTER ALIGNMENT
# ============================================================

hsv_before = cv2.cvtColor(
    before_r,
    cv2.COLOR_BGR2HSV
)

hsv_current = cv2.cvtColor(
    current_aligned,
    cv2.COLOR_BGR2HSV
)


# ============================================================
# STEP 16 — FINAL PINK MASKS
# ============================================================

pink_before = cv2.inRange(
    hsv_before,
    LOWER_PINK,
    UPPER_PINK
)

pink_current = cv2.inRange(
    hsv_current,
    LOWER_PINK,
    UPPER_PINK
)

pink_before = cv2.morphologyEx(
    pink_before,
    cv2.MORPH_OPEN,
    cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (PINK_CLEAN_SIZE, PINK_CLEAN_SIZE)
    )
)

pink_current = cv2.morphologyEx(
    pink_current,
    cv2.MORPH_OPEN,
    cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (PINK_CLEAN_SIZE, PINK_CLEAN_SIZE)
    )
)


# ============================================================
# STEP 17 — WHITE CANDIDATES
# ============================================================

white_candidate_before = (
    build_white_candidate(before_r)
)

white_candidate_current = (
    build_white_candidate(current_aligned)
)


# ============================================================
# STEP 18 — BACKGROUND MODEL
# ============================================================

background_diff_before = (
    build_background_mask(before_r)
)

background_diff_current = (
    build_background_mask(current_aligned)
)


# ============================================================
# STEP 19 — INITIAL WHITE MASKS
# ============================================================

white_before = cv2.bitwise_and(
    white_candidate_before,
    background_diff_before
)

white_current = cv2.bitwise_and(
    white_candidate_current,
    background_diff_current
)


# ============================================================
# STEP 20 — NEVER WHITE = PINK
# ============================================================

white_before = cv2.bitwise_and(
    white_before,
    cv2.bitwise_not(pink_before)
)

white_current = cv2.bitwise_and(
    white_current,
    cv2.bitwise_not(pink_current)
)


# ============================================================
# STEP 21 — RECOVER DARK WHITE INTERIORS
# ============================================================

white_before = recover_white_interiors(
    white_before,
    before_r
)

white_current = recover_white_interiors(
    white_current,
    current_aligned
)


# ============================================================
# STEP 22 — CONTROLLED CLOSE
# ============================================================

white_fill_kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (
        WHITE_FILL_CLOSE_SIZE,
        WHITE_FILL_CLOSE_SIZE
    )
)

white_before = cv2.morphologyEx(
    white_before,
    cv2.MORPH_CLOSE,
    white_fill_kernel
)

white_current = cv2.morphologyEx(
    white_current,
    cv2.MORPH_CLOSE,
    white_fill_kernel
)


# ============================================================
# STEP 23 — SMALL CLEANUP
# ============================================================

white_open_kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (3, 3)
)

white_before = cv2.morphologyEx(
    white_before,
    cv2.MORPH_OPEN,
    white_open_kernel
)

white_current = cv2.morphologyEx(
    white_current,
    cv2.MORPH_OPEN,
    white_open_kernel
)


# ============================================================
# STEP 24 — REMOVE SMALL WHITE COMPONENTS
# ============================================================

white_before = remove_small_components(
    white_before,
    WHITE_MIN_COMPONENT
)

white_current = remove_small_components(
    white_current,
    WHITE_MIN_COMPONENT
)


# ============================================================
# STEP 25 — FINAL PINK EXCLUSION
# ============================================================

white_before = cv2.bitwise_and(
    white_before,
    cv2.bitwise_not(pink_before)
)

white_current = cv2.bitwise_and(
    white_current,
    cv2.bitwise_not(pink_current)
)


# ============================================================
# STEP 26 — SHOW COLOR MASKS
# ============================================================

if SHOW_COLOR_MASKS:

    plt.figure(figsize=(15, 9))

    plt.subplot(2, 2, 1)
    plt.imshow(
        pink_before,
        cmap="gray"
    )
    plt.title("BEFORE — PINK")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(
        white_before,
        cmap="gray"
    )
    plt.title("BEFORE — WHITE CORAL")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(
        pink_current,
        cmap="gray"
    )
    plt.title("CURRENT — PINK")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(
        white_current,
        cmap="gray"
    )
    plt.title("CURRENT — WHITE CORAL")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# ============================================================
# STEP 27 — CORAL MASKS
# ============================================================

coral_before = cv2.bitwise_or(
    pink_before,
    white_before
)

coral_current = cv2.bitwise_or(
    pink_current,
    white_current
)


# ============================================================
# STEP 28 — RECOVERY
# ============================================================

recovery_mask = cv2.bitwise_and(
    white_before,
    pink_current
)


# ============================================================
# STEP 29 — BLEACHING
# ============================================================

bleaching_mask = cv2.bitwise_and(
    pink_before,
    white_current
)


# ============================================================
# STEP 30 — REMOVE ALIGNMENT EDGE TRANSITIONS
# ============================================================

transition_kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (7, 7)
)

recovery_core = cv2.erode(
    recovery_mask,
    transition_kernel,
    iterations=1
)

bleaching_core = cv2.erode(
    bleaching_mask,
    transition_kernel,
    iterations=1
)


# ============================================================
# STEP 31 — DAMAGE
# ============================================================

damage_kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (
        DAMAGE_CORE_SIZE,
        DAMAGE_CORE_SIZE
    )
)

old_coral_core = cv2.erode(
    coral_before,
    damage_kernel,
    iterations=1
)

damage_mask = cv2.bitwise_and(
    old_coral_core,
    cv2.bitwise_not(coral_current)
)


# ============================================================
# STEP 32 — GROWTH
# ============================================================

new_pink = cv2.bitwise_and(
    pink_current,
    cv2.bitwise_not(coral_before)
)

distance_from_old = cv2.distanceTransform(
    cv2.bitwise_not(coral_before),
    cv2.DIST_L2,
    5
)

growth_neighborhood_kernel = (
    cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            GROWTH_NEIGHBOR_SIZE,
            GROWTH_NEIGHBOR_SIZE
        )
    )
)

old_coral_neighborhood = cv2.dilate(
    coral_before,
    growth_neighborhood_kernel,
    iterations=1
)

near_old_coral = cv2.bitwise_and(
    new_pink,
    old_coral_neighborhood
)

growth_distance_mask = (
    distance_from_old >= GROWTH_MIN_DISTANCE
).astype(np.uint8) * 255

growth_mask = cv2.bitwise_and(
    near_old_coral,
    growth_distance_mask
)


# ============================================================
# STEP 33 — RECOVERY HAS ABSOLUTE PRIORITY
# ============================================================

growth_mask = cv2.bitwise_and(
    growth_mask,
    cv2.bitwise_not(recovery_mask)
)

growth_mask = cv2.bitwise_and(
    growth_mask,
    cv2.bitwise_not(recovery_core)
)

growth_mask = cv2.bitwise_and(
    growth_mask,
    cv2.bitwise_not(bleaching_mask)
)


# ============================================================
# STEP 34 — DAMAGE PRIORITY
# ============================================================

damage_mask = cv2.bitwise_and(
    damage_mask,
    cv2.bitwise_not(recovery_mask)
)

damage_mask = cv2.bitwise_and(
    damage_mask,
    cv2.bitwise_not(bleaching_mask)
)


# ============================================================
# STEP 35 — CLEAN CHANGE MASKS
# ============================================================

change_kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (
        CHANGE_OPEN_SIZE,
        CHANGE_OPEN_SIZE
    )
)

growth_mask = cv2.morphologyEx(
    growth_mask,
    cv2.MORPH_OPEN,
    change_kernel
)

damage_mask = cv2.morphologyEx(
    damage_mask,
    cv2.MORPH_OPEN,
    change_kernel
)

bleaching_mask = cv2.morphologyEx(
    bleaching_mask,
    cv2.MORPH_OPEN,
    change_kernel
)

recovery_mask = cv2.morphologyEx(
    recovery_mask,
    cv2.MORPH_OPEN,
    change_kernel
)


# ============================================================
# STEP 36 — REMOVE SMALL CHANGE COMPONENTS
# ============================================================

growth_mask = remove_small_components(
    growth_mask,
    MIN_CHANGE_AREA
)

damage_mask = remove_small_components(
    damage_mask,
    MIN_CHANGE_AREA
)

bleaching_mask = remove_small_components(
    bleaching_mask,
    MIN_CHANGE_AREA
)

recovery_mask = remove_small_components(
    recovery_mask,
    MIN_CHANGE_AREA
)


# ============================================================
# STEP 37 — FINAL MUTUAL EXCLUSION
# ============================================================

growth_mask = cv2.bitwise_and(
    growth_mask,
    cv2.bitwise_not(recovery_mask)
)

growth_mask = cv2.bitwise_and(
    growth_mask,
    cv2.bitwise_not(bleaching_mask)
)

damage_mask = cv2.bitwise_and(
    damage_mask,
    cv2.bitwise_not(recovery_mask)
)

damage_mask = cv2.bitwise_and(
    damage_mask,
    cv2.bitwise_not(bleaching_mask)
)


# ============================================================
# STEP 38 — DISPLAY FINAL CHANGE MASKS
# ============================================================

if SHOW_CHANGE_MASKS:

    plt.figure(figsize=(15, 9))

    plt.subplot(2, 2, 1)
    plt.imshow(
        growth_mask,
        cmap="gray"
    )
    plt.title("GROWTH")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(
        damage_mask,
        cmap="gray"
    )
    plt.title("DAMAGE")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(
        bleaching_mask,
        cmap="gray"
    )
    plt.title("BLEACHING")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(
        recovery_mask,
        cmap="gray"
    )
    plt.title("RECOVERY")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# ============================================================
# STEP 39 — GET DETECTION AREAS
# ============================================================

growth_areas = get_change_areas(
    growth_mask,
    MIN_CHANGE_AREA
)

damage_areas = get_change_areas(
    damage_mask,
    MIN_CHANGE_AREA
)

bleaching_areas = get_change_areas(
    bleaching_mask,
    MIN_CHANGE_AREA
)

recovery_areas = get_change_areas(
    recovery_mask,
    MIN_CHANGE_AREA
)


# ============================================================
# STEP 40 — RESULTS
# ============================================================

print()
print("================================================")
print("              FINAL DETECTION")
print("================================================")

print(
    "Growth areas    :",
    len(growth_areas)
)

print(
    "Damage areas    :",
    len(damage_areas)
)

print(
    "Bleaching areas :",
    len(bleaching_areas)
)

print(
    "Recovery areas  :",
    len(recovery_areas)
)

total = (
    len(growth_areas)
    + len(damage_areas)
    + len(bleaching_areas)
    + len(recovery_areas)
)

print(
    "Total areas     :",
    total
)


# ============================================================
# STEP 41 — DRAW RESULTS
# ============================================================

result = current_aligned.copy()


# -------------------------
# Growth
# -------------------------

for item in growth_areas:

    x, y, width, height = item["bbox"]

    cv2.rectangle(
        result,
        (x, y),
        (x + width, y + height),
        (0, 255, 0),
        4
    )

    cv2.putText(
        result,
        "GROWTH",
        (x, max(y - 10, 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


# -------------------------
# Damage
# -------------------------

for item in damage_areas:

    x, y, width, height = item["bbox"]

    cv2.rectangle(
        result,
        (x, y),
        (x + width, y + height),
        (0, 255, 255),
        4
    )

    cv2.putText(
        result,
        "DAMAGE",
        (x, max(y - 10, 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )


# -------------------------
# Bleaching
# -------------------------

for item in bleaching_areas:

    x, y, width, height = item["bbox"]

    cv2.rectangle(
        result,
        (x, y),
        (x + width, y + height),
        (0, 0, 255),
        4
    )

    cv2.putText(
        result,
        "BLEACHING",
        (x, max(y - 10, 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )


# -------------------------
# Recovery
# -------------------------

for item in recovery_areas:

    x, y, width, height = item["bbox"]

    cv2.rectangle(
        result,
        (x, y),
        (x + width, y + height),
        (255, 0, 0),
        4
    )

    cv2.putText(
        result,
        "RECOVERY",
        (x, max(y - 10, 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )


# ============================================================
# STEP 42 — FINAL RESULT
# ============================================================

plt.figure(figsize=(15, 10))

plt.imshow(
    cv2.cvtColor(
        result,
        cv2.COLOR_BGR2RGB
    )
)

plt.title(
    "CORAL CHANGE DETECTION"
)

plt.axis("off")
plt.tight_layout()
plt.show()