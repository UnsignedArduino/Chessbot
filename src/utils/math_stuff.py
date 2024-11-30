import numpy as np
from shapely.geometry.polygon import Polygon


def angle_between_points(a, b, c):
    # Convert points to numpy arrays
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # Calculate vectors BA and BC
    ba = a - b
    bc = c - b

    # Calculate the dot product and magnitudes of the vectors
    dot_product = np.dot(ba, bc)
    magnitude_ba = np.linalg.norm(ba)
    magnitude_bc = np.linalg.norm(bc)

    # Calculate the cosine of the angle
    cos_angle = dot_product / (magnitude_ba * magnitude_bc)

    # Calculate the angle in radians
    angle_radians = np.arccos(cos_angle)

    # Convert the angle to degrees
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees


def find_closest_to_right_angles(polygon: Polygon) -> Polygon:
    points = list(polygon.exterior.coords)[:-1]  # Exclude the repeated last point
    angles = []

    for i in range(len(points)):
        a = points[i - 1]
        b = points[i]
        c = points[(i + 1) % len(points)]
        angle = angle_between_points(a, b, c)
        angles.append((b, angle))

    # Remove the point furthest from 90 degrees until only 4 points remain
    # This keeps the points in order
    while len(angles) > 4:
        worst_angle_index = 0
        for i, (_, angle) in enumerate(angles):
            if abs(angle - 90) > abs(angles[worst_angle_index][1] - 90):
                worst_angle_index = i
        angles.pop(worst_angle_index)

    # Keep only the four points closest to 90 degrees
    closest_points = [point for point, angle in angles[:4]]
    return Polygon(closest_points)
