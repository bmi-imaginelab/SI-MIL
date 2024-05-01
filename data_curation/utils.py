import cv2
import numpy as np
import json
from openslide    import open_slide, ImageSlide
import matplotlib.pyplot as plt
import json
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
from skimage.measure import regionprops, label, regionprops_table
import math
import pickle
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import Delaunay
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from shapely.geometry import Polygon
from scipy.stats import kurtosis, skew

def extract_delaunay_features(centroids_list):
    if len(centroids_list)==0:
        return [0]*6

    perimeter_list = []
    area_list = []

    for temp_region in Delaunay(centroids_list).simplices:
        temp_coordinates = centroids_list[temp_region]
        pgon = Polygon(zip(temp_coordinates[:,0], temp_coordinates[:,1]))
        area_list.append(pgon.area)
        perimeter_list.append(pgon.length)

    area_list = np.array(area_list)
    perimeter_list = np.array(perimeter_list)

    return [area_list.mean(), area_list.std(), area_list.min()/area_list.max(), perimeter_list.mean(), perimeter_list.std(), perimeter_list.min()/perimeter_list.max()]


def extract_voronoi_features(centroids_list):
    if len(centroids_list)==0:
        return [0]*6

    vor = Voronoi(centroids_list)

    vor_regions_final = []
    vor_vertices_index_final = []

    min_point = centroids_list.min(0)-100
    max_point = centroids_list.max(0)+100

    for i in range(len(vor.vertices)):
        temp_vertex = vor.vertices[i]
        min_point = centroids_list.min(0)
        max_point = centroids_list.max(0)
        
        if temp_vertex[0]>=min_point[0] and temp_vertex[1]>=min_point[1] and temp_vertex[0]<=max_point[0] and temp_vertex[1]<=max_point[1]:
            vor_vertices_index_final.append(i)
            

    vor_vertices_index_final = np.array(vor_vertices_index_final)

    for i in vor.regions:
        if len(i)>=1:
            ignore = False
            for j in i:
                if j not in vor_vertices_index_final:
                    ignore = True

            if ignore is False:
                vor_regions_final.append(i)
            

    perimeter_list = []
    area_list = []

    for temp_region in vor_regions_final:
        temp_coordinates = vor.vertices[temp_region]
        pgon = Polygon(zip(temp_coordinates[:,0], temp_coordinates[:,1]))
        area_list.append(pgon.area)
        perimeter_list.append(pgon.length)

    area_list = np.array(area_list)
    perimeter_list = np.array(perimeter_list)

    if area_list.shape[0] != 0:
       return [area_list.mean(), area_list.std(), area_list.min()/area_list.max(), perimeter_list.mean(), perimeter_list.std(), perimeter_list.min()/perimeter_list.max()]
    else:
        return [0]*6


def distance_metric(point1, point2, method='euclidean'):
    
    if method == 'euclidean':
        distance = ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5
        
    return distance

def eucledian_probab_graph(coordinates_array, alpha=0.4, r=0.2):
    
    number_of_nodes = coordinates_array.shape[0]
    dimension_of_dataset = coordinates_array.shape[1]  # currently in 2D
    
    Adjacancy_matrix = np.zeros((number_of_nodes, number_of_nodes))
    Distance_matrix = np.zeros((number_of_nodes, number_of_nodes))
    edge_list = []


    for i in range(number_of_nodes-1):
        for j in np.arange(i+1,number_of_nodes):
            
            temp_distance = distance_metric(coordinates_array[i], coordinates_array[j])
            
            if temp_distance**(-1*alpha) > r:
                
                Adjacancy_matrix[i,j] = 1
                Adjacancy_matrix[j,i] = 1
                
                Distance_matrix[i,j] = temp_distance
                Distance_matrix[j,i] = temp_distance
                
                edge_list.append((i,j))
                edge_list.append((j,i))
                
    return Adjacancy_matrix, Distance_matrix, edge_list
            

def extract_statistics(feature_list):
    feature_list = np.array(feature_list)
    if len(feature_list.shape) == 1:
        return [np.mean(feature_list), np.std(feature_list), np.median(feature_list), kurtosis(feature_list), skew(feature_list)]

    else:
        temp_feature = np.mean(feature_list, axis = 0).reshape((-1,1))
        temp_feature = np.concatenate((temp_feature, np.std(feature_list, axis=0).reshape((-1,1))), axis=1)
        temp_feature = np.concatenate((temp_feature, np.median(feature_list, axis=0).reshape((-1,1))), axis=1)
        temp_feature = np.concatenate((temp_feature, kurtosis(feature_list, axis=0).reshape((-1,1))), axis=1)
        temp_feature = np.concatenate((temp_feature, skew(feature_list, axis=0).reshape((-1,1))), axis=1)
       
        return list(temp_feature.reshape(-1))


def calculate_area(coordinates):
    try:
        return ConvexHull(coordinates).area
    except:
        return 0 # for colinear points