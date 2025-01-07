import numpy as np
import matplotlib.pyplot as plt
import csv
import math
from scipy.interpolate import make_interp_spline
from matplotlib.patches import Circle


def plot_polygons(polygons, title, nrow, ncol, dpi=500):
    plt.rcParams["figure.dpi"] = dpi
    # Assume that the polygon profiles are in polar coordinates
    fig, axes = plt.subplots(nrow, ncol, figsize=(8, 8))
    for i, ax in enumerate(axes.flatten()):
        r = polygons[i, :np.shape(polygons)[1]]  # The distance from the centroid
        X_MIN = 0
        X_MAX = 2 * math.pi
        theta = np.linspace(X_MIN, X_MAX, np.shape(polygons)[1])

        # Insert the first element as the new last element
        r = np.append(r, r[0])
        theta = np.append(theta, X_MAX)

        x = r * np.cos(theta)  # The x-coordinate
        y = r * np.sin(theta)  # The y-coordinate

        fill_color = 'saddlebrown'
        #ax.plot(x, y, color=line_color)  # Plot the polygon with specified line color
        ax.fill(x, y, color=fill_color, alpha=0.5)  
        ax.set_aspect('equal')  # Set the aspect ratio
        ax.axis('off')  # Turn off the axis

    plt.suptitle(title, fontsize=12,fontweight='bold')  # Set the title
    plt.tight_layout()  # Adjust the spacing
    plt.show()  # Show the plot
def plot_two_polygons(polygons01, polygons02, title01, title02, dpi=500):
    plt.rcParams["figure.dpi"] = dpi
    # Assume that the polygon profiles are in polar coordinates
    fig, axes = plt.subplots(1,2, figsize=(8, 8))
    
    r01 = polygons01 # The distance from the centroid
    r02 = polygons02  # The distance from the centroid
    X_MIN = 0
    X_MAX = 2 * math.pi
    theta = np.linspace(X_MIN, X_MAX, np.shape(polygons01)[0])

    # Insert the first element as the new last element
    #r01 = np.append(r01, r01[0])
    #r02 = np.append(r02, r02[0])

    #theta = np.append(theta, X_MAX)

    x = r01 * np.cos(theta)  # The x-coordinate
    y = r01 * np.sin(theta)  # The y-coordinate

    fill_color = 'saddlebrown'
        #ax.plot(x, y, color=line_color)  # Plot the polygon with specified line color
    axes[0].fill(x, y, color=fill_color, alpha=0.5)
    axes[0].set_title(title01)

    x = r02 * np.cos(theta)  # The x-coordinate
    y = r02 * np.sin(theta)  # The y-coordinate
    axes[1].fill(x, y, color=fill_color, alpha=0.5)
    axes[1].set_title(title02)
    axes[0].set_aspect('equal')  # Set the aspect ratio
    axes[0].axis('off')  # Turn off the axis
    axes[1].set_aspect('equal')  # Set the aspect ratio
    axes[1].axis('off')  # Turn off the axis

    #plt.suptitle(title, fontsize=12,fontweight='bold')  # Set the title
    plt.tight_layout()  # Adjust the spacing
    plt.show()  # Show the plot

def plot_single_polygon(polygons, title, dpi=500):
    plt.rcParams["figure.dpi"] = dpi
    # Assume that the polygon profiles are in polar coordinates
    fig, ax = plt.subplots(figsize=(8, 6))
    
    r = polygons  # The distance from the centroid
    theta = np.linspace(0, 2 * math.pi, len(polygons))  # Angle from 0 to 2pi

    # Insert the first element as the new last element
    r = np.append(r, r[0])
    theta = np.append(theta, 2 * math.pi)

    x = r * np.cos(theta)  # The x-coordinate
    y = r * np.sin(theta)  # The y-coordinate

    fill_color = 'saddlebrown'
    ax.fill(x, y, color=fill_color, alpha=0.5)
    ax.set_title(title)
    ax.set_aspect('equal')  # Set the aspect ratio
    ax.axis('off')  # Turn off the axis

    plt.tight_layout()  # Adjust the spacing
    plt.show()  # Show the plot

def plot_save_polygons(polygons, title, nrow, ncol, title_font_size=16, line_color='blue', title_bold=True, resolution=1000, save_path='output.png', dpi=500):
    plt.rcParams["figure.dpi"] = dpi
    # Assume that the polygon profiles are in polar coordinates
    fig, axes = plt.subplots(nrow, ncol, figsize=(8, 8))
    for i, ax in enumerate(axes.flatten()):
        r = polygons[i, :np.shape(polygons)[1]]  # The distance from the centroid
        X_MIN = 0
        X_MAX = 2 * math.pi
        theta = np.linspace(X_MIN, X_MAX, np.shape(polygons)[1])

        # Insert the first element as the new last element
        r = np.append(r, r[0])
        theta = np.append(theta, X_MAX)

        x = r * np.cos(theta)  # The x-coordinate
        y = r * np.sin(theta)  # The y-coordinate
        fill_color = 'saddlebrown'
        #ax.plot(x, y, color=line_color)  # Plot the polygon with specified line color
        ax.fill(x, y, color=fill_color, alpha=0.5)  
        ax.set_aspect('equal')  # Set the aspect ratio
        ax.axis('off')  # Turn off the axis

    title_obj = plt.suptitle(title, fontsize=title_font_size, fontweight='bold' if title_bold else 'normal')  
    plt.tight_layout()  # Adjust the spacing
    plt.savefig(save_path, dpi=dpi)  # Save the image with specified DPI
    plt.close()  # Close the plot
    
def plot_smooth_polygons(polygons, title, ncol, nrow, dpi=500):
    title_font_size=12
    line_color='blue'
    title_bold=True
    #resolution=1000
    smooth_factor=3
    # Assume that the polygon profiles are in polar coordinates
    plt.rcParams["figure.dpi"] = dpi
    fig, axes = plt.subplots(nrow, ncol, figsize=(8, 8))
    for i, ax in enumerate(axes.flatten()):
        r = polygons[i, :np.shape(polygons)[1]]  # The distance from the centroid
        X_MIN = 0
        X_MAX = 2 * math.pi
        theta = np.linspace(X_MIN, X_MAX, np.shape(polygons)[1])

        # Insert the first element as the new last element
        r = np.append(r, r[0])
        theta = np.append(theta, X_MAX)

        # Use cubic spline interpolation
        spl = make_interp_spline(theta, r, k=smooth_factor)
        theta_smooth = np.linspace(X_MIN, X_MAX, np.shape(polygons)[1])
        r_smooth = spl(theta_smooth)

        x_smooth = r_smooth * np.cos(theta_smooth)  # The x-coordinate
        y_smooth = r_smooth * np.sin(theta_smooth)  # The y-coordinate
        ax.plot(x, y, color="blue")  # Plot the polygon
        ax.plot(x_smooth, y_smooth, color="red")  # Plot the smoothed polygon
        ax.set_aspect('equal')  # Set the aspect ratio
        ax.axis('off')  # Turn off the axis

    title_obj = plt.suptitle(title, fontsize=title_font_size, fontweight='bold' if title_bold else 'normal')  
    plt.tight_layout()  # Adjust the spacing
    plt.show()  # Show the plot

def plot_fit_circle(points, corner_points, non_corner_points, centroid_particle, center_circle, radius_circle, dpi=500, marker_size = 5):
    # Plot the points
    plt.rcParams["figure.dpi"] = dpi
    plt.figure(figsize=(8, 6))
    fill_color = 'saddlebrown'

    plt.fill(points[:,0], points[:,1], color=fill_color, alpha=0.5)

    #plt.scatter(non_corner_points[:,0], non_corner_points[:,1], marker='x', facecolors='blue', edgecolors='blue', label='Non-corner points')  # Plot non-corner points
    #plt.scatter(corner_points[:,0], corner_points[:,1], marker='o', facecolors='none', edgecolors='red', label='Corner points')  # Plot corner points
    plt.scatter(non_corner_points[:,0], non_corner_points[:,1], s =  marker_size, marker='d', facecolors='blue', edgecolors='black', label='Non-corner points', linewidths=0.5)  # Plot non-corner points
    plt.scatter(corner_points[:,0], corner_points[:,1], s =  marker_size, marker='o', facecolors='red', edgecolors='black', label='Corner points', linewidths=0.5)  # Plot corner points
    plt.plot(centroid_particle[0], centroid_particle[1], marker='o',color = "blueviolet", label='Centroid of particle')  # Plot corner points
    # Plot the circle
    circle = plt.Circle(center_circle, radius_circle, color='g', fill=False, label='Best Fit Circle', linewidth=2)
    plt.gca().add_artist(circle)

    plt.plot(center_circle[0], center_circle[1], '.',color = "g", label='Center of best-fit circle')  # Plot corner points

    plt.title('Identifying Corner Points and Best-fit Circle', fontname="Times New Roman", fontsize=14)
    plt.xlabel('X-coordinate', fontname="Times New Roman", fontsize=12)
    plt.ylabel('Y-coordinate', fontname="Times New Roman", fontsize=12)

    # Show the plot
    font_properties = {'family': "Times New Roman", 'size': 10}
    plt.legend(prop=font_properties, loc = 'upper right',frameon=False)
    plt.axis('equal')  # Ensure aspect ratio is equal
    #plt.grid(True)

    plt.tick_params(axis='both', which='major', labelsize=12)

    # Set font name for x-axis and y-axis tick labels
    xmin = -1.5
    xmax = 1.5
    xint = 0.5
    ymin = -1.5
    ymax = 1.5
    yint = 0.5

    plt.xticks(np.arange(xmin, xmax+xint, xint),fontname="Times New Roman", fontsize=12)  # Set x-axis tick positions at intervals of 1
    plt.yticks(np.arange(ymin, ymax+yint, yint),fontname="Times New Roman", fontsize=12)  # Set y-axis tick positions at intervals of 0.2

    #plt.ylim(xmin, xmax) # <- changed plt by ax
    #plt.xlim(ymin, ymax)

    # Turn off the axis
    plt.axis('off')

    plt.show()
    
def plot_geometry(list_geometry, nrow, ncol, save_path='out.png', dpi=500):
    # Set font name for x-axis and y-axis tick labels
    xmin = -1
    xmax = 1
    xint = 0.5
    ymin = -1
    ymax = 1
    yint = 0.5
    # Assume that the polygon profiles are in polar coordinates
    plt.rcParams["figure.dpi"] = dpi
    fig, axes = plt.subplots(nrow, ncol, figsize=(10, 10))
    for i, ax in enumerate(axes.flatten()):
        
        element = list_geometry[i]
        
        roundness = element.roundness
        list_circle = element.list_circle
        insc_circle = element.inscribed_circle
        points = element.points
        
        # Plot corner circles
        for j in range(len(list_circle)):
            circle = list_circle[j]
            center = circle.center
            radius = circle.radius
            # Add a label at the point
            #ax.text(center[0], center[1], '$C_{' + str(i+1) + '}$', verticalalignment='bottom', horizontalalignment='right', fontname="Arial", fontsize=10)

            # Plot the circle
            circle = Circle(center, radius, color='g', fill=False, label='Best Fit Circle', linewidth=2)
            #plt.gca().add_artist(circle)
            ax.add_patch(circle)
            #ax.plot(center[0], center[1], '.',color = "g", label='Center of best-fit circle')  # Plot corner points
        
        # Plot inscribed circle
        center_insc = insc_circle.center
        radius_insc = insc_circle.radius
        # Plot the inscribed circle
        insc_circle = Circle(center_insc, radius_insc, color='r', fill=False, label='Best Fit Circle', linewidth=3)
        ax.add_patch(insc_circle)

        #ax.plot(center_insc[0], center_insc[1], '+',color = "r", markerfacecolor='none', label='Center of best-fit circle')  # Plot corner points
        #ax.plot(center_insc[0], center_insc[1], 'o',color = "r", markerfacecolor='none', label='Center of best-fit circle')  # Plot corner points
        
        # Insert roundness ratio
        #ax.text(0, -1, f'$R_w$={roundness:0.4f}', color='b', verticalalignment='top', horizontalalignment='center', fontname="Arial", fontsize=10)
        ax.set_title(f'$R_w$={roundness:0.4f}', fontsize=12, color='b')  # Set title for each subplot
        fill_color = 'saddlebrown'
        ax.plot(points[:,0],points[:,1], color='black', linewidth=2)  # Plot the polygon with specified line color
        #ax.fill(points[:,0],points[:,1], color=fill_color, alpha=0.5)  
        
        ax.set_xlim(xmin, xmax)  # Set xlim for each subplot
        ax.set_ylim(ymin, ymax)  # Set ylim for each subplot
        ax.set_aspect('equal')  # Set the aspect ratio
        ax.axis('off')  # Turn off the axis

    #title_obj = plt.suptitle(title, fontsize=title_font_size, fontweight='bold' if title_bold else 'normal')  
    plt.tight_layout()  # Adjust the spacing
    plt.savefig(save_path, dpi=dpi)  # Save the image with specified DPI
    plt.show()  # Close the plot
    plt.close()