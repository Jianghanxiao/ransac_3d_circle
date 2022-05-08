import open3d as o3d
import numpy as np
import scipy.linalg
from numpy import dot
from numpy.linalg import norm
import itertools


# Borrow the idea from https://github.com/sergarrido/random/blob/master/circle3d/circle3d.cpp
def get_circle(points, index):
    # random_index = np.random.permutation(len(points))[:3]
    three_points = points[np.array(list(index))]
    v1 = three_points[1] - three_points[0]
    v2 = three_points[2] - three_points[0]

    v1v1 = v1.dot(v1)
    v2v2 = v2.dot(v2)
    v1v2 = v1.dot(v2)

    base = 0.5 / (v1v1 * v2v2 - v1v2 * v1v2 + 1e-9)
    k1 = base * v2v2 * (v1v1 - v1v2)
    k2 = base * v1v1 * (v2v2 - v1v2)
    c = three_points[0] + v1 * k1 + v2 * k2

    radius = np.linalg.norm(c - three_points[0])

    axis = np.cross(v1, v2)
    axis = axis / np.linalg.norm(axis)

    vote = 0
    vote_index = []
    for i in range(len(points)):
        dist = np.linalg.norm(points[i] - c)
        if (
            np.absolute(radius - dist) < 0.01
            and np.absolute(
                np.arccos(np.absolute(np.dot(axis, (points[i] - c) / dist)))
                - 90.0 / 180 * np.pi
            )
            < 15.0 / 180 * np.pi
        ):
            vote += 1
            vote_index.append(i)

    return axis, c, vote, vote_index


def ransac(points, iter=None):
    best_value = 0
    best_vote_index = None
    best_axis = None
    best_origin = None
    if iter == None:
        all_combiantions = list(itertools.combinations(list(range(len(points))), r=3))
        iter = len(all_combiantions)
    for i in range(iter):
        axis, origin, value, vote_index = get_circle(points, all_combiantions[i])
        if value > best_value:
            best_value = value
            best_axis = [axis]
            best_origin = [origin]
            best_vote_index = vote_index
        if value == best_value:
            best_axis.append(axis)
            best_origin.append(origin)
    return np.median(np.array(best_axis), axis=0), np.median(np.array(best_origin), axis=0), best_value, best_vote_index


def get_arrow(origin=[0, 0, 0], end=None, color=[0, 0, 0]):
    """
    Creates an arrow from an origin point to an end point,
    or create an arrow from a vector vec starting from origin.
    Args:
        - end (): End point. [x,y,z]
        - vec (): Vector. [i,j,k]
    """
    vec_Arr = np.array(end) - np.array(origin)
    vec_len = np.linalg.norm(vec_Arr) * 0.3
    mesh_arrow = o3d.geometry.TriangleMesh.create_arrow(
        cone_height=0.2 * vec_len,
        cone_radius=0.001,
        cylinder_height=0.8 * vec_len,
        cylinder_radius=0.01,
    )
    mesh_arrow.paint_uniform_color(color)
    rot_mat = caculate_align_mat(vec_Arr)
    mesh_arrow.rotate(rot_mat, center=np.array([0, 0, 0]))
    mesh_arrow.translate(np.array(origin))
    return mesh_arrow

def get_cross_prod_mat(pVec_Arr):
    # pVec_Arr shape (3)
    qCross_prod_mat = np.array([
        [0, -pVec_Arr[2], pVec_Arr[1]],
        [pVec_Arr[2], 0, -pVec_Arr[0]],
        [-pVec_Arr[1], pVec_Arr[0], 0],
    ])
    return qCross_prod_mat

def caculate_align_mat(pVec_Arr):
    scale = np.linalg.norm(pVec_Arr)
    pVec_Arr = pVec_Arr / scale
    # must ensure pVec_Arr is also a unit vec.
    z_unit_Arr = np.array([0, 0, 1])
    z_mat = get_cross_prod_mat(z_unit_Arr)
    z_c_vec = np.matmul(z_mat, pVec_Arr)
    z_c_vec_mat = get_cross_prod_mat(z_c_vec)
    if np.dot(z_unit_Arr, pVec_Arr) == -1:
        qTrans_Mat = -np.eye(3, 3)
    elif np.dot(z_unit_Arr, pVec_Arr) == 1:   
        qTrans_Mat = np.eye(3, 3)
    else:
        qTrans_Mat = np.eye(3, 3) + z_c_vec_mat + np.matmul(z_c_vec_mat,
                                                        z_c_vec_mat)/(1 + np.dot(z_unit_Arr, pVec_Arr))
    qTrans_Mat *= scale
    return qTrans_Mat


def evalAxisDir(pred_axis, gt_axis):
    pred_axis = np.array(pred_axis)
    gt_axis = np.array(gt_axis)

    if np.sum(pred_axis**2) == 0:
        raise ValueError("Pred Axis is not a vector")

    if np.sum(gt_axis**2) == 0:
        raise ValueError("GT Axis is not a vector")

    axis_similarity = dot(gt_axis, pred_axis) / (
        norm(gt_axis) * norm(pred_axis)
    )
    if axis_similarity < 0:
        axis_similarity = -axis_similarity
    axis_similarity = min(axis_similarity, 1.0)
    ## dtAxis used for evaluation metric MD
    axis_similarity = np.arccos(axis_similarity) / np.pi * 180

    return axis_similarity

# Evaluate the origin, return the distance from pred origin to the gt axis line
def evalAxisOrig(pred_axis, gt_axis, pred_origin, gt_origin):
    p = pred_origin - gt_origin
    axis_line_similarity = np.linalg.norm(
        np.cross(p, gt_axis)
    ) / np.linalg.norm(gt_axis)

    return axis_line_similarity

if __name__ == "__main__":
    points = np.load("hand_microwave.npy")

    # A = np.c_[points[:,0], points[:,1], np.ones(points.shape[0])]
    # C,_,_,_ = scipy.linalg.lstsq(A, points[:,2])

    # plane_normal = np.array([C[0], C[1], -1])
    # plane_normal = plane_normal / np.linalg.norm(plane_normal)

    # filter_points = []
    # for i in range(len(points)):
    #     point = points[i]
    #     if C[0]*point[0] + C[1]*point[1] + C[2] - point[2] < 0.01:
    #         filter_points.append(points[i])
    
    # print(f"Original points: {len(points)}; Filtered points: {len(filter_points)}")
    # points = np.array(filter_points)

    filter_points = []
    for i in range(len(points)):
        if i == 0:
            filter_points.append(points[i])
        else:
            if np.linalg.norm(points[i] - points[i-1]) <= 1e-5:
                continue
            filter_points.append(points[i])


    print(f"Original points: {len(points)}; Filtered points: {len(filter_points)}")
    points = np.array(filter_points)

    axis, origin, value, vote_index = ransac(points)
    
    hand_mesh = []
    for i in range(len(points)):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.compute_vertex_normals()
        T = np.eye(4)
        T[:3, -1] = points[i]
        sphere.transform(T)
        if i in vote_index:
            sphere.paint_uniform_color([1., 0., 0.])
        else:
            sphere.paint_uniform_color([0., 0., 0.])
        hand_mesh.append(sphere)
    
    # Draw the gt axis and origin
    gt_axis = np.array([-0.09538594633340836, 0.9817603230476379, 0.16446362435817719])
    gt_origin = np.array([-0.10314829647541046, -0.02397972345352173, -1.9173485040664673])
    gt_axis_mesh = get_arrow(gt_origin - gt_axis, gt_origin + gt_axis, color=[0, 1, 0])

    # Draw the pred axis and origin
    # print(axis)
    # print(plane_normal)
    # axis = plane_normal
    pred_axis_mesh = get_arrow(origin - axis, origin + axis, color=[1, 0, 0])
    
    print(value)
    print(f"Axis Dir Error: {evalAxisDir(axis, gt_axis)}")
    print(f"Axis Origin Error: {evalAxisOrig(axis, gt_axis, origin, gt_origin)}")

    print(f"Predicted Axis: {axis}; Predicted Origin: {origin}")

    o3d.visualization.draw_geometries(hand_mesh + [gt_axis_mesh, pred_axis_mesh])

