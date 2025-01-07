import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.path import Path

import statsmodels.api as sm

from sklearn.model_selection import KFold

from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError  # Import QhullError
from scipy.interpolate import interp1d
from scipy.interpolate import griddata

class MyGeometry:
    def __init__(self, points, roundness, list_circle, inscribed_circle):
        self.points = points
        self.roundness = roundness
        self.list_circle = list_circle
        self.inscribed_circle = inscribed_circle

class MyCircle:
    def __init__(self, radius, center):
        self.radius = radius
        self.center = center
    
    def __repr__(self):
        return f"Circle(radius={self.radius}, center={self.center})"

def calc_geometry(y_original, critical_ratio = 0.03): # critical ration = 0.03 %
    
    # Generate profile
    X_MIN = 0
    X_MAX = 2 * math.pi
    theta = np.linspace(X_MIN, X_MAX, len(y_original))

    profile = np.column_stack((theta, y_original))

    # Generate random points
    
    pointsX,pointsY = calc_coordinate(y_original)  # Example irregularities with 64 points

    points = np.column_stack((pointsX, pointsY))

    # Calculate the centroid point

    xcen = np.mean(points[:,0])
    ycen = np.mean(points[:,1])

    O = [xcen, ycen] 

    # Create boundary points
    hull = ConvexHull(points)
    boundary_points = points[hull.vertices]

    # Find furthest points among the boundary points
    point1, point2 = find_furthest_points(boundary_points)

    # Create initial circle
    init_circle = make_circle(point1, point2)

    # Find the outer circle
    outer_circle = minimum_circle(points, init_circle)

    # Calculate the threshold_distance due to the outer circle
    #threshold_distance = critical_ratio #float(2.0*critical_ratio/100.0*outer_circle.radius) #critical_ratio/100.0*outer_circle.radius
    outer_circle_diamter = 2*outer_circle.radius
    threshold_distance = critical_ratio; #critical_ratio/100.0*outer_circle_diamter #(2*0.8607104162155964)
    print(f"threshold_distance = {threshold_distance}")

    # Identify corner points
    key_points_polar = find_key_points(profile, threshold_distance)

    key_points_Oxy = polar2oxy(key_points_polar)
    #print(np.shape(smoothed_points))

    corner_points_idx ,non_corner_points_idx = identify_corner_points(key_points_Oxy, O)


    corner_X = key_points_Oxy[corner_points_idx,0]
    corner_Y = key_points_Oxy[corner_points_idx,1]

    corner_points = np.column_stack((corner_X, corner_Y))

    non_corner_X = key_points_Oxy[non_corner_points_idx,0]
    non_corner_Y = key_points_Oxy[non_corner_points_idx,1]

    non_corner_points = np.column_stack((non_corner_X, non_corner_Y))

    # Find the best fit circle
    center_circle, radius_circle = fit_circle(corner_points)
    print(f'threshold distance:{threshold_distance}')
    print(f'original points: {len(points)-1}; keypoints:{len(key_points_Oxy)}')
    print(f'corner points: {len(corner_points)}; non-corner points:{len(non_corner_points)}')

    # Create an empty list to store circle objects
    circle_list = []

    # Initial temp_points
    temp_points = corner_points
    index_corner = np.linspace(0,len(corner_points)-1,len(corner_points), dtype=int)

    index_points = index_corner

    # start index
    p = 0
    IsAccept = 0
    cutoffid = 0
    startid = -1
    prev_index_points = -2

    required_min_circle = 3

    for p in range(len(corner_points)):
    #while len(temp_points) > 3:  
        if IsAccept == 1:
            startid = cutoffid+1
        elif IsAccept == 0:
            startid += 1
        if startid > len(corner_points) -1:
                break
        temp_points = corner_points[startid:]
        index_points = index_corner[startid:]
        #print(f"start id{startid}, p {p}")
        for i in range(len(temp_points)):
            #print(f"points_shape {np.shape(temp_points)}")
            if len(temp_points) < required_min_circle:
                IsAccept = 0
                break
            #print(f"i{p}, shape {np.shape(temp_points)}")
            if len(temp_points) == required_min_circle:
                circle_info = fit_circle_with_three_points(temp_points)
            else:
                circle_info = fit_circle(temp_points)
            if circle_info is not None:
                center_circle, radius_circle = circle_info
                if point_inside_polygon(center_circle, points):
                    #if i % 10 ==0:
                    #plot_fit_circle(points, corner_points, non_corner_points, O, center_circle, radius_circle)
                    # Remove outer circle
                    min_disT = 0
                    for j in range(len(points)):
                        dis_T = distance(center_circle, points[j])
                        if j == 0:
                            min_disT = dis_T
                        if min_disT > dis_T:
                            min_disT = dis_T
                        #print(f"T={dis_T:0.4f}, R={radius_circle:0.4f}")
                    # Check min Distance T
                    if min_disT < 0.98*radius_circle:
                        temp_points = temp_points[:-1] # Remove the last element
                        index_points = index_points[:-1] # Remove the last index
                        IsAccept = 0
                        #print(f"T={min_disT:0.4f}, R={radius_circle:0.4f}")
                    # Condition for acceptable circle
                    elif min_disT >= 0.98*radius_circle and min_disT <= radius_circle:
                        #plot_fit_circle(points,corner_points, non_corner_points, O, center_circle, radius_circle)
                        tmp_circle = MyCircle(radius_circle, center_circle)
                        #if len(temp_points) >= 3:
                        # if len(circle_list) == 0:
                        #     prev_radius = radius_circle
                        # if len(circle_list) > 1:
                        #     prev_radius = circle_list[-1].radius_circle
                        if prev_index_points < index_points[0]: # and abs(radius_circle - prev_radius) > 1e-2:
                            #print(f"prev_index={prev_index_points}, index={index_points[0]}")
                            circle_list.append(tmp_circle)
                            cutoffid = int(index_points[-1])
                            IsAccept = 1
                            #print(f"cutoff id {index_points[-1]}")
                            #print(f"cutoff id {index_points[0]}")
                            #print(f"Acceptable circle {index_points}")
                            prev_index_points = int(index_points[-1])
                            break
                        else:
                            IsAccept = 0
                            temp_points = temp_points[:-1] # Remove the last element
                            index_points = index_points[:-1] # Remove the last index
                else:
                    IsAccept = 0
                    temp_points = temp_points[:-1] # Remove the last element
                    index_points = index_points[:-1] # Remove the last index
                #prev_radius = radius_circle
            else:
                IsAccept = 0
                temp_points = temp_points[:-1] # Remove the last element
                index_points = index_points[:-1] # Remove the last index  
            
    circle_list = remove_similar_circles(circle_list)
    # Example usage
    polygon_points = points  # Example polygon points

    grid_points_inside_polygon = points_inside_polygon(polygon_points, -1, 1, -1, 1, 0.01)

    check_point = grid_points_inside_polygon[0]

    min_dist = min_distance_to_point(check_point, polygon_points)

    #print(f'min. distance:{min_dist}')
    #print(f'shape:{np.shape(grid_points_inside_polygon)}')
    #print(grid_points_inside_polygon)

    zval = calc_level(grid_points_inside_polygon, polygon_points)

    # Example data
    X = grid_points_inside_polygon[:,0]  # Example x-coordinates
    Y = grid_points_inside_polygon[:,1] # Example y-coordinates
    Z = zval  # Example 1D Z values

    # Define grid for interpolation
    X_grid, Y_grid = np.meshgrid(np.linspace(-1, 1, 50), np.linspace(-1, 1, 50))

    #xy_points, Z = pointsXYZ_inside_polygon(polygon_points,X,Y,Z_init)

    #X_grid = xy_points[:,0]
    #Y_grid = xy_points[:,1]

    # Interpolate to generate Z_grid
    Z_grid = griddata((X, Y), Z, (X_grid, Y_grid), method='cubic')



    # Define boundary points
    boundary_points = points

    # Create a Path object from boundary points
    boundary_path = Path(boundary_points)

    # Mask grid points outside the boundary polygon
    points_inside_boundary = boundary_path.contains_points(np.column_stack((X_grid.flatten(), Y_grid.flatten())))
    Z_grid[~points_inside_boundary.reshape(X_grid.shape)] = np.nan

    # Find the maximum value in Z
    #max_Z = np.max(Z)

    # Round the maximum value to the nearest even integer
    #rounded_max_Z = np.ceil(max_Z*10)/10
    # Redifine the inscribed circle

    # Find indices of maximum Z value
    max_index = np.unravel_index(np.nanargmax(Z_grid), Z_grid.shape)
    max_z_value = Z_grid[max_index]

    # Get corresponding X and Y coordinates
    max_x = X_grid[max_index]
    max_y = Y_grid[max_index]


    radius_insc = max_z_value
    center_insc =[max_x, max_y]

    inscribed_circle = MyCircle(radius_insc, center_insc )

    roundness = calc_roundness(circle_list, inscribed_circle)

    print(f"Roundness coefficient: {roundness:0.4f}")

    return MyGeometry(points, roundness, circle_list, inscribed_circle),len(key_points_polar),len(corner_points)


def is_inside_circle(circle, p):
    return distance(circle.center, p) <= circle.radius

def make_circle(p1, p2):
    center = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]
    radius = distance(p1, p2) / 2
    return MyCircle(radius, center)

def make_circle_three_points(p1, p2, p3):
    # Midpoints of the sides of the triangle
    mid12 = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]
    mid23 = [(p2[0] + p3[0]) / 2, (p2[1] + p3[1]) / 2]

    # Slopes of the perpendicular bisectors
    slope12 = -1 / ((p2[1] - p1[1]) / (p2[0] - p1[0]))
    slope23 = -1 / ((p3[1] - p2[1]) / (p3[0] - p2[0]))

    # Calculate the intersection point of the perpendicular bisectors
    
    center_x = (slope12 * mid12[0] - slope23 * mid23[0] + mid23[1] - mid12[1]) / (slope12 - slope23)
    center_y = slope12 * ( center_x - mid12[0]) + mid12[1]
    
    center = [center_x, center_y]

    # Calculate the radius
    radius = distance(center, p1)

    return MyCircle(radius, center)

def find_furthest_points(points):
    max_distance = 0
    furthest_points = None
    
    for i, point1 in enumerate(points):
        for j, point2 in enumerate(points):
            if i != j:  # Exclude comparing the same point
                distance = np.linalg.norm(point1 - point2)  # Euclidean distance
                if distance > max_distance:
                    max_distance = distance
                    furthest_points = (point1, point2)
    
    return furthest_points

def minimum_circle(points, init_circle):
    
    tmp_points = points
    # Shuffle the points randomly
    #np.random.shuffle(tmp_points) 
    
    # Initialize the circle to None
    min_circle = init_circle

    for i, p in enumerate(tmp_points):
        if not min_circle is None:
            if not is_inside_circle(min_circle, p):
                min_circle = MyCircle(0, p)
                for j in range(i):
                    if not is_inside_circle(min_circle, points[j]):
                        min_circle = make_circle(p, points[j])
                        for k in range(j):
                            if not is_inside_circle(min_circle, points[k]):
                                min_circle = make_circle_three_points(p, points[j], points[k])
        assert min_circle is not None
    return min_circle

def calc_coordinate(polygons01):
   
    r01 = polygons01 # The distance from the centroid

    X_MIN = 0
    X_MAX = 2 * math.pi
    theta = np.linspace(X_MIN, X_MAX, np.shape(polygons01)[0])

    # Insert the first element as the new last element
    r01 = np.append(r01, r01[0])
   
    theta = np.append(theta, X_MAX)

    x = r01 * np.cos(theta)  # The x-coordinate
    y = r01 * np.sin(theta)  # The y-coordinate

    return x, y

def polar2oxy(polygons01): 
    r01 = polygons01[:,1] # The distance from the centroid
    theta = polygons01[:,0]
    x = r01 * np.cos(theta)  # The x-coordinate
    y = r01 * np.sin(theta)  # The y-coordinate

    return np.column_stack((x, y))

def distance(point01, point02):
    return np.sqrt((point01[0] - point02[0])**2 + (point01[1] - point02[1])**2)

def distance_to_segment(point, A, B):
    """
    Calculate the distance between a point and a line segment defined by two endpoints A and B.

    Parameters:
        point (numpy.ndarray): Coordinates of the point.
        A (numpy.ndarray): Coordinates of the first endpoint of the line segment.
        B (numpy.ndarray): Coordinates of the second endpoint of the line segment.

    Returns:
        float: The distance between the point and the line segment.
    """
    # Calculate vector representing the line segment AB
    AB = B - A

    # Calculate vector representing the line segment from the point to point A
    AP = point - A

    # Project AP onto AB
    projection = np.dot(AP, AB) / np.dot(AB, AB)

    if projection <= 0:
        # The closest point is A
        distance = np.linalg.norm(point - A)
    elif projection >= 1:
        # The closest point is B
        distance = np.linalg.norm(point - B)
    else:
        # The closest point is between A and B
        closest_point = A + projection * AB
        distance = np.linalg.norm(point - closest_point)

    return distance
def angle_between_vectors(C, A, B):
    """
    Calculate the angle between vectors CA and CB.

    Parameters:
        C (numpy.ndarray): Coordinates of point C.
        A (numpy.ndarray): Coordinates of point A.
        B (numpy.ndarray): Coordinates of point B.

    Returns:
        float: Angle between vectors CA and CB in degrees.
    """
    # Calculate vectors CA and CB
    CA = A - C
    CB = B - C

    # Calculate the dot product of CA and CB
    dot_product = np.dot(CA, CB)

    # Calculate the magnitudes of CA and CB
    magnitude_CA = np.linalg.norm(CA)
    magnitude_CB = np.linalg.norm(CB)

    # Calculate the cosine of the angle between CA and CB using the dot product formula
    cos_angle = dot_product / (magnitude_CA * magnitude_CB)

    # Calculate the angle in radians using the arccosine function
    angle_rad = np.arccos(cos_angle)

    # Convert the angle from radians to degrees
    angle_deg = np.degrees(angle_rad)

    return angle_deg

def residuals(params, points):
    center_x, center_y, radius = params
    return sum((distance(point, (center_x, center_y)) - radius)**2 for point in points)

def fit_circle_back(points):
    # Initial guess for center and radius (you may adjust this depending on your data)
    initial_guess = [np.mean(points[:, 0]), np.mean(points[:, 1]), np.std(points)]
    
    # Use optimization to find the best fit circle parameters
    result = minimize(residuals, initial_guess, args=(points,), method='Nelder-Mead')
    
    # Extract center and radius from the optimization result
    center_x, center_y, radius = result.x
    
    return (center_x, center_y), radius

def fit_circle_with_three_points(points):
    # Calculate midpoints
    mid1 = [(points[0][0] + points[1][0]) / 2, (points[0][1] + points[1][1]) / 2]
    mid2 = [(points[1][0] + points[2][0]) / 2, (points[1][1] + points[2][1]) / 2]
    
    # Calculate slopes of lines
    slope1 = (points[1][1] - points[0][1]) / (points[1][0] - points[0][0])
    slope2 = (points[2][1] - points[1][1]) / (points[2][0] - points[1][0])
    
    # Calculate perpendicular bisectors
    if slope1 != 0:  # Handle case of zero slope
        perp_slope1 = -1 / slope1
    else:
        perp_slope1 = np.inf
    if slope2 != 0:  # Handle case of zero slope
        perp_slope2 = -1 / slope2
    else:
        perp_slope2 = np.inf
    
    # Calculate y-intercepts
    if perp_slope1 != np.inf:
        b1 = mid1[1] - perp_slope1 * mid1[0]
    else:
        b1 = np.inf
    if perp_slope2 != np.inf:
        b2 = mid2[1] - perp_slope2 * mid2[0]
    else:
        b2 = np.inf
    
    # Calculate center point
    if perp_slope1 != perp_slope2:
        center_x = (b2 - b1) / (perp_slope1 - perp_slope2)
        center_y = perp_slope1 * center_x + b1
    else:
        center_x = (mid1[0] + mid2[0]) / 2
        center_y = (mid1[1] + mid2[1]) / 2
    
    # Calculate radius
    radius = np.sqrt((points[0][0] - center_x)**2 + (points[0][1] - center_y)**2)
    
    return (center_x, center_y), radius

def objective(params, points):
    center_x, center_y, radius = params
    return sum((distance(point, (center_x, center_y)) - radius)**2 for point in points)

def fit_circle(points):
    # Initial guess for center and radius
    initial_guess = [np.mean(points[:, 0]), np.mean(points[:, 1]), np.std(points)]

    # Use optimization to find the best-fit circle parameters
    result = minimize(objective, initial_guess, args=(points,), method='Nelder-Mead')
    
    # Extract center and radius from the optimization result
    center_x, center_y, radius = result.x
    
    return (center_x, center_y), radius
def fit_circle_convex(points):
    try:
        # Calculate the convex hull of the points
        hull = ConvexHull(points)
        # Get the vertices of the convex hull
        hull_vertices = points[hull.vertices]

        # Initial guess for center and radius (you may adjust this depending on your data)
        initial_guess = [np.mean(hull_vertices[:, 0]), np.mean(hull_vertices[:, 1]), np.std(points)]

        # Use optimization to find the best fit circle parameters
        result = minimize(residuals, initial_guess, args=(points,), method='Nelder-Mead')

        # Extract center and radius from the optimization result
        center_x, center_y, radius = result.x

        return (center_x, center_y), radius
    except QhullError as e: #Exception as e:
        print(f"Error occurred during optimization: {e}")
        return None

def find_intersection_point(A, B, O, C):
    # Calculate slopes
    m = (B[1] - A[1]) / (B[0] - A[0])
    n = (C[1] - O[1]) / (C[0] - O[0])

    # Calculate y-intercepts
    c1 = A[1] - m * A[0]
    c2 = O[1] - n * O[0]

    # Calculate intersection point
    x_intersect = (c2 - c1) / (m - n)
    y_intersect = m * x_intersect + c1

    return x_intersect, y_intersect

def identify_neighbor_points(points, threshold_distance):
    near_points = []
    for i in range(len(points)):
        p1 = points[i-1]
        p2 = points[i]
        dis = distance(p1, p2)
        if dis <= threshold_distance:
            near_points.append(i)
    return near_points

def find_key_points_old(points, threshold_distance):
    # Initialize the first and last points
    first_point = points[0]
    last_point = points[-1]

    # Initialize the key points list with the first and last points
    key_points = [first_point, last_point]

    while not np.all(last_point != points[-1]):
        print("DEBUG")
        # Find the point with the maximum distance to the segment connecting the first and last points
        max_distance = 0
        max_distance_point = None
        for point in points:
            dist = distance_to_segment(point, first_point, last_point)
            if dist > max_distance:
                max_distance = dist
                max_distance_point = point

        # Check if the maximum distance is greater than the threshold distance
        if max_distance > threshold_distance:
            # Update the last point to be the point with the maximum distance
            last_point = max_distance_point
        else:
            # Update the first point to be the point with the maximum distance
            first_point = max_distance_point

        # Add the new key point to the list
        key_points.append(max_distance_point)

    return key_points


def distance_point_to_line(point, line):
    """Calculate the perpendicular distance from a point to a line."""
    x0, y0 = point
    x1, y1 = line[0]
    x2, y2 = line[1]
    return np.abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1)**2 + (x2 - x1)**2)


def intersection_point(segment_point1, segment_point2, given_point):
    # Check if the segment is vertical
    if segment_point1[0] == segment_point2[0]:
        # If the segment is vertical, calculate intersection x-coordinate directly
        intersection_x = segment_point1[0]
        # Calculate the equation of the line passing through the given point
        intersection_y = given_point[1]
    else:
        # Calculate the equation of the line passing through the given point and parallel to the given segment
        segment_slope = (segment_point2[1] - segment_point1[1]) / (segment_point2[0] - segment_point1[0])
        segment_intercept = segment_point1[1] - segment_slope * segment_point1[0]

        # Equation of the line: y = mx + b
        # For the line passing through the given point, slope = segment_slope
        given_point_intercept = given_point[1] - segment_slope * given_point[0]

        # Calculate the intersection point
        intersection_x = (given_point_intercept - segment_intercept) / segment_slope
        intersection_y = segment_slope * intersection_x + segment_intercept
    
    return [intersection_x, intersection_y]


def identify_corner_points(points, O):
    corner_points = []
    non_corner_points = []
    for i in range(len(points)-2):
        A = points[i-1]
        C = points[i]
        B = points[i+1]
        
        angle = angle_between_vectors(C, A, B)
        #if angle >= 180 and angle <= 180:
        #    non_corner_points.append(i)
        #else:  
        D = find_intersection_point(A, B, O, C)
        dis_OC = distance(O,C)
        dis_OD = distance(O,D)
        if dis_OC > dis_OD:
            corner_points.append(i)
        else:
            non_corner_points.append(i)
    return corner_points, non_corner_points

def point_inside_polygon(point, polygon):
    """
    Check if a point is inside a polygon using ray casting.
    """
    x, y = point
    n = len(polygon)
    inside = False
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[(i + 1) % n]
        if (yi < y <= yj or yj < y <= yi) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
    return inside

def find_key_points(points, threshold_distance):
    
    index_points = np.linspace(0,len(points)-1,len(points), dtype=int)
    
    # Step 01: Create the segment connecting the first and last points
    line = [points[0], points[-1]]
    #print(f"DEBUG:{line}")
    
    # Initialize variables
    key_points = [points[0]]
    IsFinish = 0
    count = 0
    list_points = points
    list_index = index_points
    
    while IsFinish == 0 : #not np.all(last_point == points[-1]):
        if len(list_points) == 1: #(last_point == points[-1]).all():
            # Find the position of the value
            position = list_index[-1]+1 #np.where(points == first_point)
            #print(f"SPECIAL:{position}\n {first_point}\n {list_points}")
            if position -1 == len(points)-1:
                #print(f"END POINTS:{position}\n {first_point}\n {np.shape(list_points)}\n {np.shape(points)}")
                key_points.append(first_point)
                #return key_points
                break;
            else:
                #position = position[-1]
                #print(f"SPECIAL:{position}") #\n {first_point}\n {points}")
                list_points = points
                list_index = index_points
                list_points = list_points[position:,:]
                list_index = list_index[position:]
                
                #print(f"SPECIAL222:{position}\n {first_point}\n {list_points}\n {np.shape(points)}")

                first_point = list_points[0]
                key_points.append(first_point)
                #key_points.insert(0, first_point)
                line = [first_point, points[-1]]
            #IsFinish == 1
            #break
        max_distance = 0
        max_point = None
        max_index = 0
        count += 1
        #if count >= len(points):
            #return key_points
            #break
        #print(f"count: {count}, listpoints: {np.shape(list_points)}")
        # Step 02: Find the point with the maximum perpendicular distance to the line
        for index, point in enumerate(list_points):
            #intpoint = intersection_point(line[0], line[1], point)
            dist = perp_dist(point, line)
            #print(f"dist: {dist}")
            if index == 0:
                max_distance = dist
                max_point = point
                max_index = index
            #print(f"id: {index}, dist: {dist}, p1:{line[0]}, p2:{line[1]}, p:{point}, int:{intpoint}")
            if not np.isinf(dist) and not np.isnan(dist) and dist >= max_distance:
                max_distance = dist
                max_point = point
                max_index = index
        
        #print(f"id: {max_index}, max_dist: {max_distance}, max point: {max_point}")
        if not np.isinf(max_distance) and not np.isnan(max_distance) and not max_point is None:
            # Step 03: Determine whether to extend the segment or start a new one
            if max_distance > threshold_distance:
                last_point = max_point
                #key_points.append(last_point)
                line = [list_points[0], last_point]
                list_points = list_points[0:max_index,:]
                list_index = list_index[0:max_index]
                #print(f"max_dis > threshold, listpoints: {np.shape(list_points)}")
            elif max_distance <= threshold_distance:
                first_point = list_points[-1]
                #key_points.insert(0, first_point)
                line = [first_point, points[-1]]
                # Find the position of the value
                position = list_index[-1] #np.where(points == first_point)
                key_points.append(first_point)
                #position = position[-1]
                #print(f"{position}") #\n {first_point}\n {points}")
                list_points = points
                list_index = index_points
                list_points = list_points[position:,:]
                list_index = list_index[position:]
                #print(f"OK_max_dis <= threshold, from {list_index[0]} to {list_index[-1]} listpoints: {np.shape(list_points)}")
    return np.array(key_points)

# Define a function to calculate the perpendicular distance from a point to a line segment
def perp_dist(p, s):
    # p is a point (x, y), s is a segment ((x1, y1), (x2, y2))
    # Calculate the differences
    #print(f"point: {p}, line: {s}")
    dx = s[1][0] - s[0][0]
    dy = s[1][1] - s[0][1]
    # Check if the segment is a single poin
    if dx == 0 and dy == 0:
        return ((p[0] - s[0][0]) ** 2 + (p[1] - s[0][1]) ** 2) ** 0.5 # Return the Euclidean distance
    else:
        # Calculate the dot product
        dot = ((p[0] - s[0][0]) * dx + (p[1] - s[0][1]) * dy) / (dx ** 2 + dy ** 2)
        # Check if the projection is outside the segment
        if dot < 0:
            return ((p[0] - s[0][0]) ** 2 + (p[1] - s[0][1]) ** 2) ** 0.5 # Return the distance to the first endpoint
        elif dot > 1:
            return ((p[0] - s[1][0]) ** 2 + (p[1] - s[1][1]) ** 2) ** 0.5 # Return the distance to the second endpoint
        else:
            # Calculate the projection point
            px = s[0][0] + dot * dx
            py = s[0][1] + dot * dy
            #print(f"projection point: {px},{py}")
            return ((p[0] - px) ** 2 + (p[1] - py) ** 2) ** 0.5 # Return the distance to the projection point

def distance2circles(circle1, circle2):
    return np.sqrt((circle1.center[0] - circle2.center[0])**2 + (circle1.center[1] - circle2.center[1])**2)
def remove_similar_circles(circles, percentage_threshold=1):
    if not circles:
        # Handle the case where the circles list is empty
        return []
    min_radius = min(circle.radius for circle in circles)
    threshold_radius = min_radius * percentage_threshold
    threshold_distance = min_radius * percentage_threshold

    new_circles = []
    similar_circles = set()
    for i, circle1 in enumerate(circles):
        if i not in similar_circles:
            similar_circles.add(i)
            for j, circle2 in enumerate(circles):
                if i != j and j not in similar_circles:
                    # Check similarity based on radius and distance between centers
                    if (abs(circle1.radius - circle2.radius) < threshold_radius) and \
                       (distance2circles(circle1, circle2) < threshold_distance):
                        similar_circles.add(j)
            new_circles.append(circle1)
    return new_circles

def points_inside_polygon(polygon_points, xmin, xmax, ymin, ymax, spacing):
    # Create a grid of points with specified spacing distance
    # Generate x and y coordinates within the specified ranges with the given spacing
    x = np.linspace(xmin, xmax, int((xmax - xmin) / spacing) + 1)
    y = np.linspace(ymin, ymax, int((ymax - ymin) / spacing) + 1)

    # Create a meshgrid from the x and y coordinates
    xx, yy = np.meshgrid(x, y)

    # Flatten the meshgrid to get the grid points
    grid_points = np.vstack([xx.ravel(), yy.ravel()]).T

    # Create a Path object from the polygon points
    path = Path(polygon_points)

    # Check which grid points fall inside the polygon
    inside = path.contains_points(grid_points)

    # Filter and return the points inside the polygon
    points_inside = grid_points[inside]
    return points_inside

def pointsXYZ_inside_polygon(polygon_points, x_range, y_range, z_range):
    x = x_range
    y = y_range
    # Create a meshgrid from the x and y coordinates
    xx, yy = np.meshgrid(x, y)

    # Flatten the meshgrid to get the grid points
    grid_points = np.vstack([xx.ravel(), yy.ravel()]).T

    # Create a Path object from the polygon points
    path = Path(polygon_points)

    # Check which grid points fall inside the polygon
    inside = path.contains_points(grid_points)

    # Filter and return the points inside the polygon
    points_inside = grid_points[inside]
    z_inside = z_range[inside]
    return points_inside, z_inside

def min_distance_to_point(point, array_points):
    # Calculate the Euclidean distance between the point and each point in the array
    distances = np.linalg.norm(array_points - point, axis=1)
    # Find the minimum distance
    min_distance = np.min(distances)
    return min_distance

def calc_level(grid_points, bound_points):
    zlevel = np.zeros(len(grid_points))
    for i in range(len(grid_points)):
        check_point = grid_points[i,:]
        zlevel[i] = min_distance_to_point(check_point, bound_points)
    return zlevel

# Calculate the roundness value
def calc_roundness(list_circle, inscribed_circle):
    R_insc = inscribed_circle.radius
    sumRc = 0
    print(f"Radius of largest inscribed circle: {R_insc:0.4f}")
    for i in range(len(list_circle)):
        circle = list_circle[i]
        center = circle.center
        radius = circle.radius
        print(f"Radius of curvature of the corner {i+1}: {radius:0.4f}")
        sumRc += radius
    return sumRc/(len(list_circle)*R_insc)