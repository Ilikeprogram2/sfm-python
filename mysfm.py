import sys
sys.path.append(r'E:\3Dreconstruction\sift')
import cv2
import numpy as np
import os
import math
from scipy.linalg import lstsq
from scipy.optimize import least_squares
import open3d

config = {'image_dir':r'E:\3Dreconstruction\sfm\images\xiaoyang',
          'K':np.array([[3140.63, 0, 1631.5],[0, 3140.63, 1223.5],[0, 0, 1]],dtype=np.float32),
          'x':1,
          'y':2
          }


###################
# 读取image，进行sift特征提取，image两两匹配关键点
###################
def read_images(image_dir:str):
    """输入图像根目录，读取所有彩色图像"""
    images = []
    for imagename in os.listdir(image_dir):
        filename = os.path.join(image_dir,imagename)
        image = cv2.imread(filename)
        if image is None:
            continue
        images.append(image)

    return np.array(images)

def feature_detect_fast(images):
    keypoint_allimage = []  # 记录所有可用image的关键点
    descriptor_allimage = []  # 对应的描述子
    color_allimage = []  # 对应关键点的颜色--3通道
    usefulimages = []  # 关键点大于10个的image
    for image in images:
        image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)  # sift检测使用单通道image
        sift = cv2.xfeatures2d.SIFT_create(0, 4, 0.04, 10)  # 关键点上限、每个otcave的image数量、不稳定点阈值、边缘响应阈值
        keypoints,descriptors = sift.detectAndCompute(image_gray, None)
        #keypoints,descriptors = mysift.featuredetectwithsift(image_gray)  # 关键点中的坐标pt 是 （水平坐标，纵向坐标）（w,h） 跟image的array是相反的
        if len(keypoints) <= 10:
            continue
        usefulimages.append(image)
        keypoint_allimage.append(keypoints)
        descriptor_allimage.append(descriptors)
        colors = np.zeros((len(keypoints),3))
        for i,keypoint in enumerate(keypoints):
            colors[i] = image[int(keypoint.pt[1])][int(keypoint.pt[0])]
        color_allimage.append(colors)

    return np.array(keypoint_allimage,dtype=object),np.array(descriptor_allimage,dtype=object),\
        np.array(color_allimage,dtype=object),np.array(usefulimages,dtype=object)

def feature_detect_withsift(images):
    keypoint_allimage = []  # 记录所有可用image的关键点
    descriptor_allimage = []  # 对应的描述子
    color_allimage = []  # 对应关键点的颜色--3通道
    usefulimages = []  # 关键点大于10个的image
    for image in images:
        image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)  # sift检测使用单通道image
        keypoints,descriptors = mysift.featuredetectwithsift(image_gray)  # 关键点中的坐标pt 是 （水平坐标，纵向坐标）（w,h） 跟image的array是相反的
        if len(keypoints) <= 10:
            continue
        usefulimages.append(image)
        keypoint_allimage.append(keypoints)
        descriptor_allimage.append(descriptors)
        colors = np.zeros((len(keypoints),3))
        for i,keypoint in enumerate(keypoints):
            colors[i] = image[int(keypoint.pt[1])][int(keypoint.pt[0])]
        color_allimage.append(colors)

    return np.array(keypoint_allimage,dtype=object),np.array(descriptor_allimage,dtype=object),\
        np.array(color_allimage,dtype=object),np.array(usefulimages,dtype=object)

def feature_match_2image(query,train):
    """实现两个image的关键点描述子最近邻匹配，返回array，每个匹配结果包含queryIdx、trainIdx、distance"""
    FLANN = 0
    index_params = dict(algorithm=FLANN,trees=5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matchs = flann.knnMatch(query,train,k=2)

    # 筛选好的配对
    good_match=[]
    for m,n in matchs:
        if m.distance < 0.7 * n.distance:
            # 好的匹配
            good_match.append(m)

    return np.array(good_match,dtype=object)

def feature_match_allimage(descriptor_allimage):
    """实现前后紧挨的两个image的关键点匹配，返回二元image组 的 匹配结果和imageindex"""
    match_allimage = []
    match_index = []
    for i in range(len(descriptor_allimage)-1):
        matchs = feature_match_2image(descriptor_allimage[i],descriptor_allimage[i+1])
        if len(matchs)<=10:
            continue
        match_index.append((i,i+1))
        match_allimage.append(matchs)

    return np.array(match_allimage,dtype=object),np.array(match_index)

###########################
# 通过最小化重投影误差，求解最优相机内外参。
# 这里降低难度，内参直接给定
###########################
def find_transform(K,p1,p2):
    """输入图1的特征点坐标p1和图2特征点坐标p2，个数得相同且对应，以及内参矩阵K，
    输出相机相对姿态（image2 to image1）和mask（指出outlier点和inlier点）"""
    #focal_length = 0.5*(K[0,0]+K[1,1])
    #principle_point = (K[0,2],K[1,2])
    E,mask = cv2.findEssentialMat(p1,p2,K,cv2.RANSAC,0.95,1.0)  # 八点法计算本质矩阵，mask是指符合对极几何的特征点
    pass_count,R,T,mask = cv2.recoverPose(E,p1,p2,K,mask)  # 本质矩阵分解为旋转矩阵和平移向量，mask指符合深度z为正数的特征点

    return R,T,mask

def get_matched_points(kp1,kp2,match):
    src_pts = np.asarray([kp1[m.queryIdx].pt for m in match])  # 关键点坐标是（w,h）
    dst_pts = np.asarray([kp2[m.trainIdx].pt for m in match])
    return src_pts,dst_pts  # N*2

def get_matched_colors(c1,c2,match):
    color_src_pts = np.asarray([c1[m.queryIdx] for m in match])
    color_dst_pts = np.asarray([c2[m.trainIdx] for m in match])
    return color_src_pts,color_dst_pts  # N*3

def maskout_points(p,mask):
    p_copy = []
    for i in range(len(mask)):
        if mask[i] > 0:
            p_copy.append(p[i])

    return np.array(p_copy)

def init_structure(K,keypoint_allimage,color_allimage,match_allimage,match_index):
    """用第一个配对的match进行初始化"""
    firstmatch_indx = match_index[0]
    p1,p2 = get_matched_points(keypoint_allimage[firstmatch_indx[0]],keypoint_allimage[firstmatch_indx[1]],match_allimage[0])  # N *2
    c1,c2 = get_matched_colors(color_allimage[firstmatch_indx[0]],color_allimage[firstmatch_indx[1]],match_allimage[0])  # N *3

    if find_transform(K,p1,p2):
        R, T, mask = find_transform(K,p1,p2)  # c2 to c1 的旋转矩阵 和 平移向量
    else:
        R, T, mask = np.array([]),np.array([]),np.array([])

    p1 = maskout_points(p1,mask)  # 满足对极几何的点
    p2 = maskout_points(p2,mask)  # 同上
    color1 = maskout_points(c1,mask)  #
    color2 = maskout_points(c2,mask)

    colors = []
    for i in range(len(keypoint_allimage)):
        colors.append(0)
    colors[firstmatch_indx[0]] = color1
    colors[firstmatch_indx[1]] = color2

    # 第一个相机坐标to世界坐标的变换矩阵,也就是记第一个相机位置为世界坐标系原点
    R0 = np.eye(3)  # c1 to world
    T0 = np.zeros((3,1))
    point_3d = triangulation(K,R0,T0,R,T,p1,p2)  # 3 * N
    point_3d = point_3d.T  # N * 3

    rotations,motions = [],[]
    for i in range(len(keypoint_allimage)):
        rotations.append(0)
        motions.append(0)

    rotations[firstmatch_indx[0]] = R0
    rotations[firstmatch_indx[1]] = R
    motions[firstmatch_indx[0]] = T0
    motions[firstmatch_indx[1]] = T

    correspond_struct_idx = []  # 记录每个image中每个关键点是否满足对极几何,满足的关键点对按顺序标为 0 1 2 3 ...，不满足的是-1
    for keypoints in keypoint_allimage:
        correspond_struct_idx.append(np.ones(len(keypoints))*-1)
    correspond_struct_idx = np.array(correspond_struct_idx)

    idx = 0
    firstmatch = match_allimage[0]
    for i,match in enumerate(firstmatch):
        if mask[i] == 0:
            continue
        correspond_struct_idx[firstmatch_indx[0]][int(match.queryIdx)] = idx
        correspond_struct_idx[firstmatch_indx[1]][int(match.trainIdx)] = idx
        idx += 1

    structure = []  # 记录各个image中得到的世界点坐标
    for i in range(len(keypoint_allimage)):
        structure.append(0)
    structure[firstmatch_indx[0]] = point_3d
    structure[firstmatch_indx[1]] = point_3d

    return structure,correspond_struct_idx,colors,rotations,motions


def triangulation(K,R1,T1,R2,T2,p1,p2):
    """"
    输入的R是camera2world
    利用三角测量计算得到p1和p2特征点对应的世界点坐标 3*N
    """
    if len(T1.shape) == 2:
        T1 = T1.flatten()
    if len(T2.shape) == 2:
        T2 = T2.flatten()
    proj1 = np.zeros((3,4))
    proj2 = np.zeros((3,4))
    proj1[:,:3] = np.float32(R1)
    proj1[:, 3] = np.float32(T1)
    proj2[:,:3] = np.float32(R2)
    proj2[:, 3] = np.float32(T2)
    K = np.float32(K)
    proj1 = K @ proj1  # 常规矩阵乘法
    proj2 = K @ proj2
    s = cv2.triangulatePoints(proj1,proj2,p1.T,p2.T)  # 三角测量，输入点要求2*N，返回4*N 真实世界齐次坐标
    point_3d = s[:3,:] / s[3,:]
    return np.asarray(point_3d)  # 3*N

##################
# 点云融合
##################
def fusion_structure(matches,match_index,struct_indices,next_struct_indices,structure,point_3d,colors,next_colors1,next_colors2):
    structure[match_index[1]] = np.empty((0,3))
    colors[match_index[1]] = np.empty((0,3))
    idx = 0
    for i,match in enumerate(matches):
        query_Idx = match.queryIdx
        train_Idx = match.trainIdx
        struct_idx = struct_indices[query_Idx]
        if struct_idx>=0:  # 这个特征点对应的世界坐标已存在于structure中的前一个元素中
            next_struct_indices[train_Idx] = idx
            idx = idx + 1
            structure[match_index[1]] = np.append(structure[match_index[1]],[structure[match_index[0]][int(struct_idx)]],axis=0)
            colors[match_index[1]] = np.append(colors[match_index[1]],[colors[match_index[0]][int(struct_idx)]],axis=0)
            continue
        structure[match_index[1]] = np.append(structure[match_index[1]],[point_3d[i]],axis = 0)
        colors[match_index[1]] = np.append(colors[match_index[1]], [next_colors2[i]], axis=0)
        next_struct_indices[train_Idx] = idx
        idx += 1
    return struct_indices,next_struct_indices,structure,colors

def get_objpoints_and_imgpoints(matches,struct_indices,structure,key_points):
    """matches 是下一个配对的image index，struct_indices是配对image中query特征点是否满足对极几何，structure是query满足对极几何特征点对应的世界点，key_points是train的关键点"""
    object_points = []
    image_points = []
    for match in matches:  # matches是一个配对image
        query_idx = match.queryIdx
        train_idx = match.trainIdx
        struct_idx = struct_indices[query_idx]
        if struct_idx<0:
            continue
        # 满足对极几何
        object_points.append(structure[int(struct_idx)])  # query中满足对极几何特征点对应的世界点
        image_points.append(key_points[train_idx].pt)  # train中与query配对的坐标，（w,h）

    return np.array(object_points),np.array(image_points)

################
# bundle adjustment
################
def get_3dpos_v1(pos,ob,r,t,K):
    """判断一个点的重投影误差是不是很大，返回重投影误差较小的点"""
    p,J = cv2.projectPoints(pos.reshape(1,1,3),r,t,K,np.array([]))
    p = p.reshape(2)
    e = ob - p
    if abs(e[0]) > config['x'] or abs(e[1]) > config['y']:
        return None
    return pos
def get_3dpos(pos,ob,r,t,K):  # pos是结构点世界坐标（一维array），ob是对应的关键点坐标
    dtype = np.float32
    def F(x):
        p,J = cv2.projectPoints(x.reshape(1,1,3),r,t,K,np.array([]))
        p = p.reshape(2)
        err = ob -p
        return err
    res = least_squares(F,pos)  # 最小二乘法优化函数，结构点世界坐标是优化对象，并且pos是作为初始值，pos就是待优化参数
    return res.x  # 最小二乘得到的pos结果

def optimization_function(x,keypoint,K):
    def F(x):
        pos = x[0:3]
        r = x[3:6]
        t = x[6:9]
        p,J = cv2.projectPoints(pos.reshape(1,1,3),r,t,K,np.array([]))
        p  =p.reshape(2)
        err = keypoint - p
        return err
    res = least_squares(F,x)
    return res.x

def bundle_adjustment(rotations,motions,K,correspond_struct_idx,keypoint_allimage,structure):
    for i in range(len(rotations)):
        r,_ = cv2.Rodrigues(rotations[i])
        rotations[i] = r  # 变成旋转向量
    for i in range(1,len(correspond_struct_idx)):
        point3d_ids = correspond_struct_idx[i]
        keypoints = keypoint_allimage[i]
        r = rotations[i]
        t = motions[i]
        for j in range(len(point3d_ids)):
            point3d_id = int(point3d_ids[j])
            if point3d_id < 0:
                continue
            new_point = get_3dpos(structure[i][point3d_id],keypoints[j].pt,r,t,K)  # 得到优化后的世界点坐标
            #if new_point is None:
            #    continue
            structure[i][point3d_id] = new_point

    return structure

#####################
# 画图
#####################
def point_clear1(points_3d,colors):
    center = np.mean(points_3d, axis=0)  # todo center似乎很大
    distance = np.linalg.norm(points_3d - center, axis=1)
    distanceup = np.sort(distance)
    threshold_dis = distanceup[int(0.9 * len(distance))]
    outlier_indice = np.where(distance > threshold_dis)[0]
    new_structure = np.delete(points_3d, outlier_indice, axis=0)
    new_color = np.delete(colors, outlier_indice, axis=0)
    return new_structure,new_color

def point_clear2(point_3d_all,color_all):
    upt = 11
    downt = -11
    condition = ((point_3d_all[:, 0] > upt) | (point_3d_all[:, 0] < downt)) | \
                ((point_3d_all[:, 1] > upt) | (point_3d_all[:, 1] < downt)) | \
                ((point_3d_all[:, 2] > upt) | (point_3d_all[:, 2] < downt))
    new_structure = point_3d_all[~condition]
    new_color = color_all[~condition]
    return new_structure,new_color

def showresult(structure,colors):  # N*3
    point_3d_all,color_all = structure[1],colors[1]/255.0
    for i in range(2,len(structure)):
        point_3d_all = np.append(point_3d_all,structure[i],axis=0)
        color_all = np.append(color_all,colors[i]/255.0,axis=0)

    for i in range(len(color_all)):
        color_all[i, :] = color_all[i, :][[2, 1, 0]]

    # 剔除点云中远离点云中心的异常点
    new_structure, new_color = point_3d_all,color_all



    point_cloud = open3d.geometry.PointCloud()  # 创建点云对象
    point_cloud.points = open3d.utility.Vector3dVector(new_structure)  # 添加点云坐标
    point_cloud.colors = open3d.utility.Vector3dVector(new_color)  # 添加点云颜色


    point_cloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])  # 上下镜像
    window = open3d.visualization.Visualizer()  # 创建窗口对象
    window.create_window(window_name='crazyhorse')
    window.get_render_option().point_size = 1  # 设置点云大小
    #window.get_view_control().set_up([0,0,1])
    #window.get_view_control().set_front([0, 1, 0])

    window.get_render_option().background_color = np.asarray([0,0,0])  # 设置背景颜色
    window.add_geometry(point_cloud)

    window.run()
    window.destroy_window()

    #open3d.visualization.draw_geometries([point_cloud])



def main():
    K = config['K']
    images = read_images(config['image_dir'])
    keypoint_allimage,descriptor_allimage,color_allimage,usefulimages = feature_detect_fast(images)
    match_allimage,match_index = feature_match_allimage(descriptor_allimage)
    structure, correspond_struct_idx, colors, rotations, motions = init_structure(K,keypoint_allimage,color_allimage,match_allimage,match_index)

    for i in range(1,len(match_allimage)):
        match_index_now = match_index[i]
        # match（i，j），从 i 获得关键点世界坐标 ， 从 j 获得与i关键点匹配的关键点坐标
        object_points,image_points = get_objpoints_and_imgpoints(match_allimage[i],
                                                                 correspond_struct_idx[match_index_now[0]],
                                                                 structure[match_index_now[0]],
                                                                 keypoint_allimage[match_index_now[1]])
        _, r, T, _ = cv2.solvePnPRansac(object_points, image_points, K, np.array([]))  # 通过最小化重投影误差 估计train相机的外参 c2w 这是BA的相机外参初始化
        R, _ = cv2.Rodrigues(r)  # 旋转向量->旋转矩阵
        rotations[match_index_now[1]] = R
        motions[match_index_now[1]] = T

        # 世界点坐标的初始化
        p1,p2 = get_matched_points(keypoint_allimage[match_index_now[0]],keypoint_allimage[match_index_now[1]],match_allimage[i])
        c1,c2 = get_matched_colors(color_allimage[match_index_now[0]],color_allimage[match_index_now[1]],match_allimage[i])
        new_point_3d = triangulation(K,rotations[match_index_now[0]],motions[match_index_now[0]],R,T,p1,p2)
        new_point_3d = new_point_3d.T  # N * 3

        # 点云融合
        correspond_struct_idx[match_index_now[0]], correspond_struct_idx[match_index_now[1]], structure, colors = fusion_structure(match_allimage[i],match_index_now,
                                                                                                     correspond_struct_idx[match_index_now[0]],
                                                                                                     correspond_struct_idx[match_index_now[1]],
                                                                                                     structure,
                                                                                                     new_point_3d,
                                                                                                     colors, c1, c2 )
    structure = bundle_adjustment(rotations, motions, K, correspond_struct_idx, keypoint_allimage, structure)

    showresult(structure,colors)




if __name__ == '__main__':
    main()