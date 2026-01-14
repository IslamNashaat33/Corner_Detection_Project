import numpy as np
import os
import csv
import math
import random

# =========================
# CONFIG
# =========================
# Number of samples per category
NUM_SINGLE_CORNER = 300
NUM_PLUS_INTERSECTIONS = 150      # + shape
NUM_X_INTERSECTIONS = 150         # X shape
NUM_T_JUNCTIONS = 150             # T shape
NUM_Y_SHAPES = 150                # Y shape
NUM_ZIGZAGS = 200                 # multiple corners
NUM_TRIANGLES = 100               # 3 corners
NUM_RECTANGLES = 100              # 4 corners
NUM_POLYGONS = 100                # random polygons
NUM_CURVED_CORNERS = 0            # disabled - no curved corners
NUM_STAR_SHAPES = 50              # star patterns
NUM_ARROW_SHAPES = 100            # arrow shapes

POINTS_PER_SEGMENT = 50
NOISE_STD_RANGE = (0.0, 0.0)  # No noise for clean/straight strokes
ANGLE_RANGE_DEG = (25, 155)

OUTPUT_DIR = "dataset_intersections_v2"
STROKE_DIR = os.path.join(OUTPUT_DIR, "strokes")
LABEL_DIR = os.path.join(OUTPUT_DIR, "labels")

os.makedirs(STROKE_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

# =========================
# HELPERS
# =========================
def unit_vector(angle):
    """Returns unit vector for given angle in radians."""
    return np.array([math.cos(angle), math.sin(angle)])

def generate_line(start, direction, length, num_points):
    """Generate a line from start point in given direction."""
    t = np.linspace(0, length, num_points)
    return start + np.outer(t, direction)

def generate_line_centered(center, direction, half_length, num_points):
    """Generate a line centered at a point."""
    t = np.linspace(-half_length, half_length, num_points)
    return center + np.outer(t, direction)

def add_noise(stroke, noise_std):
    """Add Gaussian noise to stroke."""
    if noise_std > 0:
        return stroke + np.random.normal(0, noise_std, stroke.shape)
    return stroke

def random_rotation(points, angle_rad):
    """Rotate points around origin."""
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    return points @ rotation_matrix.T

def random_scale(points, scale_range=(0.5, 2.0)):
    """Random uniform scaling."""
    scale = random.uniform(*scale_range)
    return points * scale

def random_translate(points, translate_range=(-1.0, 1.0)):
    """Random translation."""
    tx = random.uniform(*translate_range)
    ty = random.uniform(*translate_range)
    return points + np.array([tx, ty])

def apply_random_transform(points, apply_noise=True):
    """Apply random rotation, scale, translation, and noise."""
    # Random rotation
    angle = random.uniform(0, 2 * math.pi)
    points = random_rotation(points, angle)
    
    # Random scale
    points = random_scale(points, (0.6, 1.5))
    
    # Random translation
    points = random_translate(points, (-0.5, 0.5))
    
    # Add noise
    if apply_noise:
        noise_std = random.uniform(*NOISE_STD_RANGE)
        points = add_noise(points, noise_std)
    
    return points

def save_stroke(stroke, labels, stroke_id, metadata, stroke_type, info):
    """Save stroke and labels to files."""
    filename = f"stroke_{stroke_id:04d}.npy"
    np.save(os.path.join(STROKE_DIR, filename), stroke.astype(np.float32))
    np.save(os.path.join(LABEL_DIR, filename), labels.astype(np.int64))
    metadata.append([filename, stroke_type, info])
    return stroke_id + 1

# =========================
# SHAPE GENERATORS
# =========================

def generate_single_corner():
    """Generate a single corner (two line segments meeting)."""
    angle_deg = random.uniform(*ANGLE_RANGE_DEG)
    angle_rad = math.radians(angle_deg)
    
    p0 = np.array([0.0, 0.0])
    d1 = unit_vector(0.0)
    d2 = unit_vector(angle_rad)
    
    seg_len = random.uniform(0.8, 1.5)
    pts1 = random.randint(POINTS_PER_SEGMENT // 2, POINTS_PER_SEGMENT)
    pts2 = random.randint(POINTS_PER_SEGMENT // 2, POINTS_PER_SEGMENT)
    
    seg1 = generate_line(p0, d1, seg_len, pts1)
    seg2 = generate_line(seg1[-1], d2, seg_len, pts2)
    
    stroke = np.vstack([seg1, seg2])
    labels = np.zeros(len(stroke), dtype=np.int64)
    labels[len(seg1) - 1] = 1  # corner point
    
    stroke = apply_random_transform(stroke)
    return stroke, labels, f"angle_{angle_deg:.1f}"

def generate_plus_intersection():
    """Generate + shaped intersection as a single continuous stroke with corners at center.
    Path: left -> center -> up -> center -> right -> center -> down
    Corners only at the center where direction changes."""
    half_len = random.uniform(0.8, 1.5)
    pts_per_arm = random.randint(POINTS_PER_SEGMENT // 4, POINTS_PER_SEGMENT // 2)
    
    center = np.array([0.0, 0.0])
    left = center + np.array([-half_len, 0])
    right = center + np.array([half_len, 0])
    top = center + np.array([0, half_len])
    bottom = center + np.array([0, -half_len])
    
    # Path: left -> center -> top -> center -> right -> center -> bottom
    # Segment 1: left to center
    seg1 = generate_line(left, unit_vector(0), half_len, pts_per_arm)
    # Segment 2: center to top
    seg2 = generate_line(center, unit_vector(math.pi / 2), half_len, pts_per_arm)
    # Segment 3: top back to center
    seg3 = generate_line(top, unit_vector(-math.pi / 2), half_len, pts_per_arm)
    # Segment 4: center to right
    seg4 = generate_line(center, unit_vector(0), half_len, pts_per_arm)
    # Segment 5: right back to center
    seg5 = generate_line(right, unit_vector(math.pi), half_len, pts_per_arm)
    # Segment 6: center to bottom
    seg6 = generate_line(center, unit_vector(-math.pi / 2), half_len, pts_per_arm)
    
    stroke = np.vstack([seg1, seg2, seg3, seg4, seg5, seg6])
    
    labels = np.zeros(len(stroke), dtype=np.int64)
    # Corners only at center positions (not at tips)
    labels[len(seg1) - 1] = 1  # center: left->up turn
    labels[len(seg1) + len(seg2) + len(seg3) - 1] = 1  # center: down->right turn
    labels[len(seg1) + len(seg2) + len(seg3) + len(seg4) + len(seg5) - 1] = 1  # center: left->down turn
    
    stroke = apply_random_transform(stroke)
    return stroke, labels, "plus_shape"

def generate_x_intersection():
    """Generate X shaped intersection as TWO separate straight diagonal lines (no corners).
    Returns a list of (stroke, labels, info) tuples - one for each line."""
    half_len = random.uniform(0.8, 1.5)
    pts = random.randint(POINTS_PER_SEGMENT, POINTS_PER_SEGMENT * 2)
    
    center = np.array([0.0, 0.0])
    base_angle = random.uniform(0, math.pi / 4)
    
    # Two diagonal lines crossing at center
    angle1 = base_angle + math.pi / 4
    angle2 = base_angle + 3 * math.pi / 4
    
    d1 = unit_vector(angle1)
    d2 = unit_vector(angle2)
    
    # Apply same random transform to both
    rot_angle = random.uniform(0, 2 * math.pi)
    scale = random.uniform(0.6, 1.5)
    tx, ty = random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)
    
    # First diagonal
    stroke1 = generate_line_centered(center, d1, half_len, pts)
    stroke1 = random_rotation(stroke1, rot_angle)
    stroke1 = stroke1 * scale + np.array([tx, ty])
    labels1 = np.zeros(len(stroke1), dtype=np.int64)  # no corners
    
    # Second diagonal
    stroke2 = generate_line_centered(center, d2, half_len, pts)
    stroke2 = random_rotation(stroke2, rot_angle)
    stroke2 = stroke2 * scale + np.array([tx, ty])
    labels2 = np.zeros(len(stroke2), dtype=np.int64)  # no corners
    
    return [(stroke1, labels1, "x_diag_1"), (stroke2, labels2, "x_diag_2")]

def generate_t_junction():
    """Generate T-junction (one line with another perpendicular at its end)."""
    # Vertical line
    p0 = np.array([0.0, 0.0])
    d_vert = unit_vector(math.pi / 2)
    
    vert_len = random.uniform(0.8, 1.5)
    horiz_len = random.uniform(0.6, 1.2)
    
    pts_vert = random.randint(POINTS_PER_SEGMENT // 2, POINTS_PER_SEGMENT)
    pts_horiz = random.randint(POINTS_PER_SEGMENT // 2, POINTS_PER_SEGMENT)
    
    # Vertical segment
    seg_vert = generate_line(p0, d_vert, vert_len, pts_vert)
    
    # Horizontal segment from top of vertical
    top_point = seg_vert[-1]
    d_horiz = unit_vector(0)
    seg_horiz = generate_line(top_point - d_horiz * horiz_len / 2, d_horiz, horiz_len, pts_horiz)
    
    stroke = np.vstack([seg_vert, seg_horiz])
    labels = np.zeros(len(stroke), dtype=np.int64)
    labels[len(seg_vert) - 1] = 1  # T-junction point
    
    stroke = apply_random_transform(stroke)
    return stroke, labels, "t_junction"

def generate_y_shape():
    """Generate Y-shaped stroke with corner only at the junction."""
    center = np.array([0.0, 0.0])
    
    # Stem going down
    d_stem = unit_vector(-math.pi / 2)
    stem_len = random.uniform(0.8, 1.5)
    pts_stem = random.randint(POINTS_PER_SEGMENT // 2, POINTS_PER_SEGMENT)
    
    stem = generate_line(center, d_stem, stem_len, pts_stem)
    
    # Two branches going up
    branch_angle = random.uniform(math.pi / 6, math.pi / 3)  # 30-60 degrees spread
    branch_len = random.uniform(0.6, 1.2)
    pts_branch = random.randint(POINTS_PER_SEGMENT // 3, POINTS_PER_SEGMENT // 2)
    
    d_branch1 = unit_vector(math.pi / 2 - branch_angle)
    d_branch2 = unit_vector(math.pi / 2 + branch_angle)
    
    branch1 = generate_line(center, d_branch1, branch_len, pts_branch)
    branch2 = generate_line(center, d_branch2, branch_len, pts_branch)
    
    # Combine: stem (reversed so we go from bottom to center) -> branch1 -> back to center -> branch2
    stroke = np.vstack([stem[::-1], branch1, branch1[::-1], branch2])
    
    labels = np.zeros(len(stroke), dtype=np.int64)
    # Only mark corners at the CENTER JUNCTION where direction changes (not at tips)
    labels[pts_stem - 1] = 1  # center: end of stem, start of branch1
    labels[pts_stem + 2 * pts_branch - 1] = 1  # center: coming back from branch1, start of branch2
    
    stroke = apply_random_transform(stroke)
    return stroke, labels, f"y_angle_{math.degrees(branch_angle):.1f}"

def generate_zigzag():
    """Generate zigzag pattern with multiple corners."""
    num_segments = random.randint(3, 6)
    seg_len = random.uniform(0.5, 1.0)
    pts_per_seg = POINTS_PER_SEGMENT // num_segments
    pts_per_seg = max(pts_per_seg, 10)
    
    stroke_parts = []
    corner_indices = []
    current_pos = np.array([0.0, 0.0])
    current_angle = random.uniform(0, math.pi / 4)
    
    total_points = 0
    for i in range(num_segments):
        d = unit_vector(current_angle)
        segment = generate_line(current_pos, d, seg_len, pts_per_seg)
        stroke_parts.append(segment)
        
        if i > 0:
            corner_indices.append(total_points)
        
        total_points += len(segment)
        current_pos = segment[-1]
        
        # Alternate direction with random angle variation
        angle_change = random.uniform(math.pi / 3, 2 * math.pi / 3)
        if i % 2 == 0:
            current_angle += angle_change
        else:
            current_angle -= angle_change
    
    stroke = np.vstack(stroke_parts)
    labels = np.zeros(len(stroke), dtype=np.int64)
    for idx in corner_indices:
        if 0 <= idx < len(labels):
            labels[idx] = 1
    
    stroke = apply_random_transform(stroke)
    return stroke, labels, f"zigzag_{num_segments}_segments"

def generate_triangle():
    """Generate triangle with 3 corners."""
    # Random triangle vertices
    size = random.uniform(0.8, 1.5)
    angles = [random.uniform(0, 2*math.pi) for _ in range(3)]
    angles.sort()
    
    vertices = [np.array([size * math.cos(a), size * math.sin(a)]) for a in angles]
    vertices.append(vertices[0])  # close the triangle
    
    pts_per_side = POINTS_PER_SEGMENT // 3
    pts_per_side = max(pts_per_side, 15)
    
    stroke_parts = []
    corner_indices = []
    total_points = 0
    
    for i in range(3):
        start = vertices[i]
        end = vertices[i + 1]
        direction = end - start
        length = np.linalg.norm(direction)
        direction = direction / length
        
        segment = generate_line(start, direction, length, pts_per_side)
        stroke_parts.append(segment)
        
        corner_indices.append(total_points)
        total_points += len(segment)
    
    stroke = np.vstack(stroke_parts)
    labels = np.zeros(len(stroke), dtype=np.int64)
    for idx in corner_indices:
        labels[idx] = 1
    
    stroke = apply_random_transform(stroke)
    return stroke, labels, "triangle"

def generate_rectangle():
    """Generate rectangle with 4 corners."""
    width = random.uniform(0.8, 1.5)
    height = random.uniform(0.6, 1.2)
    
    vertices = [
        np.array([0, 0]),
        np.array([width, 0]),
        np.array([width, height]),
        np.array([0, height]),
        np.array([0, 0])  # close
    ]
    
    pts_per_side = POINTS_PER_SEGMENT // 4
    pts_per_side = max(pts_per_side, 12)
    
    stroke_parts = []
    corner_indices = []
    total_points = 0
    
    for i in range(4):
        start = vertices[i]
        end = vertices[i + 1]
        direction = end - start
        length = np.linalg.norm(direction)
        if length > 0:
            direction = direction / length
        
        segment = generate_line(start, direction, length, pts_per_side)
        stroke_parts.append(segment)
        
        corner_indices.append(total_points)
        total_points += len(segment)
    
    stroke = np.vstack(stroke_parts)
    labels = np.zeros(len(stroke), dtype=np.int64)
    for idx in corner_indices:
        labels[idx] = 1
    
    stroke = apply_random_transform(stroke)
    return stroke, labels, "rectangle"

def generate_polygon():
    """Generate random polygon with n corners."""
    n_sides = random.randint(5, 8)
    size = random.uniform(0.8, 1.5)
    
    # Generate irregular polygon
    angles = sorted([random.uniform(0, 2*math.pi) for _ in range(n_sides)])
    radii = [size * random.uniform(0.7, 1.3) for _ in range(n_sides)]
    
    vertices = [np.array([r * math.cos(a), r * math.sin(a)]) 
                for r, a in zip(radii, angles)]
    vertices.append(vertices[0])  # close
    
    pts_per_side = max(POINTS_PER_SEGMENT // n_sides, 8)
    
    stroke_parts = []
    corner_indices = []
    total_points = 0
    
    for i in range(n_sides):
        start = vertices[i]
        end = vertices[i + 1]
        direction = end - start
        length = np.linalg.norm(direction)
        if length > 0:
            direction = direction / length
        
        segment = generate_line(start, direction, length, pts_per_side)
        stroke_parts.append(segment)
        
        corner_indices.append(total_points)
        total_points += len(segment)
    
    stroke = np.vstack(stroke_parts)
    labels = np.zeros(len(stroke), dtype=np.int64)
    for idx in corner_indices:
        labels[idx] = 1
    
    stroke = apply_random_transform(stroke)
    return stroke, labels, f"polygon_{n_sides}_sides"

def generate_curved_corner():
    """Generate corner with slight curve (smoother transition)."""
    angle_deg = random.uniform(*ANGLE_RANGE_DEG)
    angle_rad = math.radians(angle_deg)
    
    p0 = np.array([0.0, 0.0])
    d1 = unit_vector(0.0)
    d2 = unit_vector(angle_rad)
    
    seg_len = random.uniform(0.8, 1.2)
    pts1 = POINTS_PER_SEGMENT // 2
    pts2 = POINTS_PER_SEGMENT // 2
    curve_pts = random.randint(5, 15)
    
    seg1 = generate_line(p0, d1, seg_len, pts1)
    corner_point = seg1[-1]
    
    # Add curved transition
    curve_radius = random.uniform(0.05, 0.15)
    curve_angles = np.linspace(0, angle_rad, curve_pts)
    curve = corner_point + curve_radius * np.column_stack([np.cos(curve_angles), np.sin(curve_angles)])
    
    seg2 = generate_line(curve[-1], d2, seg_len, pts2)
    
    stroke = np.vstack([seg1, curve, seg2])
    labels = np.zeros(len(stroke), dtype=np.int64)
    # Mark the middle of the curve as corner
    labels[len(seg1) + curve_pts // 2] = 1
    
    stroke = apply_random_transform(stroke)
    return stroke, labels, f"curved_corner_{angle_deg:.1f}"

def generate_star_shape():
    """Generate star shape with multiple corners."""
    n_points = random.randint(4, 6)
    outer_radius = random.uniform(1.0, 1.5)
    inner_radius = outer_radius * random.uniform(0.3, 0.5)
    
    vertices = []
    for i in range(n_points * 2):
        angle = math.pi / 2 + i * math.pi / n_points
        radius = outer_radius if i % 2 == 0 else inner_radius
        vertices.append(np.array([radius * math.cos(angle), radius * math.sin(angle)]))
    vertices.append(vertices[0])  # close
    
    pts_per_side = max(POINTS_PER_SEGMENT // (n_points * 2), 5)
    
    stroke_parts = []
    corner_indices = []
    total_points = 0
    
    for i in range(n_points * 2):
        start = vertices[i]
        end = vertices[i + 1]
        direction = end - start
        length = np.linalg.norm(direction)
        if length > 0:
            direction = direction / length
        
        segment = generate_line(start, direction, length, pts_per_side)
        stroke_parts.append(segment)
        
        corner_indices.append(total_points)
        total_points += len(segment)
    
    stroke = np.vstack(stroke_parts)
    labels = np.zeros(len(stroke), dtype=np.int64)
    for idx in corner_indices:
        labels[idx] = 1
    
    stroke = apply_random_transform(stroke)
    return stroke, labels, f"star_{n_points}_points"

def generate_arrow_shape():
    """Generate arrow shape with corners."""
    # Arrow: shaft + head (triangle)
    shaft_len = random.uniform(0.8, 1.5)
    head_size = random.uniform(0.3, 0.5)
    shaft_pts = POINTS_PER_SEGMENT // 2
    head_pts = POINTS_PER_SEGMENT // 4
    
    # Shaft
    p0 = np.array([0.0, 0.0])
    d_shaft = unit_vector(0)
    shaft = generate_line(p0, d_shaft, shaft_len, shaft_pts)
    
    # Arrow head
    tip = shaft[-1] + np.array([head_size, 0])
    head_top = shaft[-1] + np.array([0, head_size / 2])
    head_bottom = shaft[-1] + np.array([0, -head_size / 2])
    
    # Path: shaft end -> head_top -> tip -> head_bottom
    seg_to_top = generate_line(shaft[-1], (head_top - shaft[-1]) / np.linalg.norm(head_top - shaft[-1]), 
                               np.linalg.norm(head_top - shaft[-1]), head_pts)
    seg_to_tip = generate_line(head_top, (tip - head_top) / np.linalg.norm(tip - head_top),
                               np.linalg.norm(tip - head_top), head_pts)
    seg_to_bottom = generate_line(tip, (head_bottom - tip) / np.linalg.norm(head_bottom - tip),
                                  np.linalg.norm(head_bottom - tip), head_pts)
    
    stroke = np.vstack([shaft, seg_to_top, seg_to_tip, seg_to_bottom])
    labels = np.zeros(len(stroke), dtype=np.int64)
    
    # Mark corners
    labels[shaft_pts - 1] = 1  # shaft end
    labels[shaft_pts + head_pts - 1] = 1  # head_top
    labels[shaft_pts + 2 * head_pts - 1] = 1  # tip
    
    stroke = apply_random_transform(stroke)
    return stroke, labels, "arrow"

def generate_cross_with_corners():
    """Generate a cross shape drawn as single stroke with corners."""
    size = random.uniform(0.8, 1.2)
    arm_width = size * random.uniform(0.2, 0.4)
    pts_per_seg = POINTS_PER_SEGMENT // 12
    pts_per_seg = max(pts_per_seg, 5)
    
    # Define cross outline points (12 corners for a cross outline)
    half_size = size / 2
    half_width = arm_width / 2
    
    corners = [
        np.array([-half_width, half_size]),      # top left of top arm
        np.array([half_width, half_size]),       # top right of top arm
        np.array([half_width, half_width]),      # inner top right
        np.array([half_size, half_width]),       # right arm top
        np.array([half_size, -half_width]),      # right arm bottom
        np.array([half_width, -half_width]),     # inner bottom right
        np.array([half_width, -half_size]),      # bottom right of bottom arm
        np.array([-half_width, -half_size]),     # bottom left of bottom arm
        np.array([-half_width, -half_width]),    # inner bottom left
        np.array([-half_size, -half_width]),     # left arm bottom
        np.array([-half_size, half_width]),      # left arm top
        np.array([-half_width, half_width]),     # inner top left
    ]
    corners.append(corners[0])  # close
    
    stroke_parts = []
    corner_indices = []
    total_points = 0
    
    for i in range(len(corners) - 1):
        start = corners[i]
        end = corners[i + 1]
        direction = end - start
        length = np.linalg.norm(direction)
        if length > 0:
            direction = direction / length
        
        segment = generate_line(start, direction, length, pts_per_seg)
        stroke_parts.append(segment)
        
        corner_indices.append(total_points)
        total_points += len(segment)
    
    stroke = np.vstack(stroke_parts)
    labels = np.zeros(len(stroke), dtype=np.int64)
    for idx in corner_indices:
        labels[idx] = 1
    
    stroke = apply_random_transform(stroke)
    return stroke, labels, "cross_outline"

# =========================
# MAIN DATA GENERATION
# =========================
metadata = []
stroke_id = 0

print("Generating dataset with high entropy...")

# 1) Single corners
start_id = stroke_id
print(f"Generating {NUM_SINGLE_CORNER} single corners... (offset: {start_id})")
for _ in range(NUM_SINGLE_CORNER):
    stroke, labels, info = generate_single_corner()
    stroke_id = save_stroke(stroke, labels, stroke_id, metadata, "single_corner", info)

# 2) Plus intersections
start_id = stroke_id
print(f"Generating {NUM_PLUS_INTERSECTIONS} plus (+) intersections... (offset: {start_id})")
for _ in range(NUM_PLUS_INTERSECTIONS):
    stroke, labels, info = generate_plus_intersection()
    stroke_id = save_stroke(stroke, labels, stroke_id, metadata, "plus_intersection", info)

# 3) X intersections (two strokes per X)
start_id = stroke_id
print(f"Generating {NUM_X_INTERSECTIONS} X intersections (2 strokes each)... (offset: {start_id})")
for _ in range(NUM_X_INTERSECTIONS):
    stroke_list = generate_x_intersection()
    for stroke, labels, info in stroke_list:
        stroke_id = save_stroke(stroke, labels, stroke_id, metadata, "x_intersection", info)

# 4) T junctions
start_id = stroke_id
print(f"Generating {NUM_T_JUNCTIONS} T junctions... (offset: {start_id})")
for _ in range(NUM_T_JUNCTIONS):
    stroke, labels, info = generate_t_junction()
    stroke_id = save_stroke(stroke, labels, stroke_id, metadata, "t_junction", info)

# 5) Y shapes
start_id = stroke_id
print(f"Generating {NUM_Y_SHAPES} Y shapes... (offset: {start_id})")
for _ in range(NUM_Y_SHAPES):
    stroke, labels, info = generate_y_shape()
    stroke_id = save_stroke(stroke, labels, stroke_id, metadata, "y_shape", info)

# 6) Zigzags
start_id = stroke_id
print(f"Generating {NUM_ZIGZAGS} zigzags... (offset: {start_id})")
for _ in range(NUM_ZIGZAGS):
    stroke, labels, info = generate_zigzag()
    stroke_id = save_stroke(stroke, labels, stroke_id, metadata, "zigzag", info)

# 7) Triangles
start_id = stroke_id
print(f"Generating {NUM_TRIANGLES} triangles... (offset: {start_id})")
for _ in range(NUM_TRIANGLES):
    stroke, labels, info = generate_triangle()
    stroke_id = save_stroke(stroke, labels, stroke_id, metadata, "triangle", info)

# 8) Rectangles
start_id = stroke_id
print(f"Generating {NUM_RECTANGLES} rectangles... (offset: {start_id})")
for _ in range(NUM_RECTANGLES):
    stroke, labels, info = generate_rectangle()
    stroke_id = save_stroke(stroke, labels, stroke_id, metadata, "rectangle", info)

# 9) Polygons
start_id = stroke_id
print(f"Generating {NUM_POLYGONS} random polygons... (offset: {start_id})")
for _ in range(NUM_POLYGONS):
    stroke, labels, info = generate_polygon()
    stroke_id = save_stroke(stroke, labels, stroke_id, metadata, "polygon", info)

# 10) Curved corners
start_id = stroke_id
print(f"Generating {NUM_CURVED_CORNERS} curved corners... (offset: {start_id})")
for _ in range(NUM_CURVED_CORNERS):
    stroke, labels, info = generate_curved_corner()
    stroke_id = save_stroke(stroke, labels, stroke_id, metadata, "curved_corner", info)

# 11) Star shapes
start_id = stroke_id
print(f"Generating {NUM_STAR_SHAPES} star shapes... (offset: {start_id})")
for _ in range(NUM_STAR_SHAPES):
    stroke, labels, info = generate_star_shape()
    stroke_id = save_stroke(stroke, labels, stroke_id, metadata, "star", info)

# 12) Arrow shapes
start_id = stroke_id
print(f"Generating {NUM_ARROW_SHAPES} arrow shapes... (offset: {start_id})")
for _ in range(NUM_ARROW_SHAPES):
    stroke, labels, info = generate_arrow_shape()
    stroke_id = save_stroke(stroke, labels, stroke_id, metadata, "arrow", info)

# 13) Cross outlines (bonus)
start_id = stroke_id
print(f"Generating 50 cross outlines... (offset: {start_id})")
for _ in range(50):
    stroke, labels, info = generate_cross_with_corners()
    stroke_id = save_stroke(stroke, labels, stroke_id, metadata, "cross", info)

# =========================
# SAVE METADATA
# =========================
with open(os.path.join(OUTPUT_DIR, "metadata.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["stroke_file", "type", "info"])
    writer.writerows(metadata)

print(f"\n{'='*50}")
print(f"Dataset generation complete!")
print(f"Total samples: {stroke_id}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"{'='*50}")

# Print summary by type
from collections import Counter
type_counts = Counter([m[1] for m in metadata])
print("\nSamples by type:")
for t, count in sorted(type_counts.items()):
    print(f"  {t}: {count}")
