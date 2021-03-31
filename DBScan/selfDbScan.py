# GOKHAN HAS - 161044067
# DATA MINING ASSIGNMENT #01
# DBSCAN ALGORITHM

# import the libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# read data from csv
def read_data():
    data = pd.read_csv('winequality-red.csv')
    arr = StandardScaler().fit_transform((data.iloc[:, [4, 6]].values))
    return arr


def find_neighbor(n, arr, epsilon):
    dist = np.sqrt(np.sum((arr-n)**2, axis=1))
    return np.where(dist <= epsilon)[0]


def get_random_list(arr):
    random_list = np.random.choice(arr.shape[0], size=arr.shape[0], replace=False)
    random_list = random_list.tolist()
    return random_list


def region_query(arr, epsilon, minPoint):
    region_list = []
    for i in range(len(arr)):
        region = find_neighbor(arr[i], arr, epsilon)
        if len(region) >= minPoint:
            region_list.append(1)
        else:
            region_list.append(0)
    return region_list


def db_scan_algorithm(arr, epsilon, minPoint):
    cluster_number = 1
    # (1) mark all objects as unvisited;
    cluster_array = np.empty(arr.shape[0])
    cluster_array.fill(-1)

    # (3)     randomly select an unvisited object p;
    random_list = get_random_list(arr)
    neighborhoods = region_query(arr, epsilon, minPoint)

    # (16) until no object is unvisited;
    while len(random_list) != 0:
        # (4)     mark p as visited;
        random_element_index = random_list.pop(0)
        if neighborhoods[random_element_index] == 1:
            # (5)     if the ε-neighborhood of p has at least MinPts objects
            neighbor_list = (find_neighbor(arr[random_element_index], arr, epsilon)).tolist()
            for p in neighbor_list:
                if p == random_element_index:
                    neighbor_list.remove(p)
            # (6)         create a new cluster C, and add p to C;
            cluster_array[random_element_index] = cluster_number
            # (7)         let N be the set of objects in the ε-neighborhood of p;
            cluster_array[neighbor_list] = cluster_number
            # (8)         for each point p' in N
            while len(neighbor_list) != 0:
                p0_index = neighbor_list.pop(0)
                # (9)             if p' is unvisited
                if neighborhoods[p0_index] == 1:
                    # remove the stack
                    # (10)                mark p' as visited;
                    if p0_index in random_list:
                        random_list.remove(p0_index)
                        # (11)      if the ε-neighborhood of p0 has at least MinPts points,
                        #           add those points to N;
                        second_neighbor_list = (find_neighbor(arr[p0_index], arr, epsilon)).tolist()
                        for p in second_neighbor_list:
                            if p == p0_index:
                                second_neighbor_list.remove(p)
                        # (12)            if p0 is not yet a member of any cluster, add p0 to C;
                        cluster_array[p0_index] = cluster_number
                        cluster_array[second_neighbor_list] = cluster_number
                        for p in second_neighbor_list:
                            # (15)    else mark p as noise;
                            if p not in neighbor_list:
                                neighbor_list.append(p)
                        # (13)        end for
            cluster_number += 1
    return cluster_array, cluster_number


def draw_plot(cluster_arr, cluster_num):
    data_frame = pd.DataFrame()
    data_frame['x'] = arr[:, 0]
    data_frame['y'] = arr[:, 1]
    data_frame['cluster_arr'] = cluster_arr

    print(cluster_arr)
    count = 0
    for i in range(len(cluster_arr)):
        if cluster_arr[i] == -1:
            count = count + 1
    print(count)
    plt.title("Number of cluster detected: %d" % (cluster_num - 1))
    plt.scatter(data_frame['x'][cluster_arr == -1], data_frame['y'][cluster_arr == -1], s=15, marker='*', c='black')
    plt.scatter(data_frame['x'][cluster_arr != -1], data_frame['y'][cluster_arr != -1], marker='o', c=cluster_arr[cluster_arr != -1])
    plt.show();


if __name__ == '__main__':
        arr = read_data()
        cluster_arr, cluster_num = db_scan_algorithm(arr, 0.3, 4)
        print("Number of cluster detected: ", cluster_num - 1)
        draw_plot(cluster_arr, cluster_num)


"""
Algorithm: DBSCAN: a density-based clustering algorithm.
Input:
D: a data set containing n objects,
ε: the radius parameter, and
MinPts: the neighborhood density threshold.

Output: A set of density-based clusters.
Method:
(1) mark all objects as unvisited;
(2) do
(3)     randomly select an unvisited object p;
(4)     mark p as visited;
(5)     if the ε-neighborhood of p has at least MinPts objects
(6)         create a new cluster C, and add p to C;
(7)         let N be the set of objects in the ε-neighborhood of p;
(8)         for each point p' in N
(9)             if p' is unvisited
(10)                mark p' as visited;
(11)                if the ε-neighborhood of p0 has at least MinPts points,
                    add those points to N;
(12)            if p0 is not yet a member of any cluster, add p0 to C;
(13)        end for
(14)        output C;
(15)    else mark p as noise;
(16) until no object is unvisited;
"""