
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import sys
import time

def read_pts(fname):
    with open(fname,'r') as myfile:
        data = myfile.readlines()
    return np.loadtxt(data)

def read_labels(fname):
    with open(fname,'r') as myfile:
        data = myfile.readlines()
    return np.loadtxt(data).reshape((len(data),1))

def create_kd_tree(pts):
    pt_cld = []
    # find the kd-tree leafs
    kd_orients = []
    nodes = [pt_cld]
    temp_list = []
    while len(nodes[0]) != 1:
        temp_list.clear()
        for node in nodes:
            ranges = np.amax(node,axis = 0) - np.amin(node,axis = 0)
            split_dir = np.argmax(ranges)
            kd_orients.append(split_dir)
            sorted_node = node[node[:,split_dir].argsort()]
            num_pts = len(sorted_node)
            temp_list.append(sorted_node[0:np.int(num_pts/2),:])
            temp_list.append(sorted_node[np.int(num_pts/2):num_pts,:])
        nodes.clear()
        nodes.extend(temp_list)

    kd_leaves = np.array(nodes)
    kd_orients = np.array(kd_orients)
    # np.save('kd_tree.npy',kd_tree)
    print(str(time.time()-start_t) + 's elapsed.')
    return (kd_leaves,kd_orients)

if len(sys.argv)>1:
    pts = read_pts(sys.argv[1])
    labels = read_pts(sys.argv[2])
    num_parts = len(np.unique(labels))

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    colors = ['b','r','g','y']
    markers = ['o','^','+','x']
    xs, ys, zs = pts[:,0],pts[:,1],pts[:,2]
    for i in range(num_parts):
        indices = np.where(labels==i+1)
        ax.scatter(xs[indices],ys[indices],zs[indices],color=colors[i],marker='o')
    plt.show()
    exit()

# loc_of_label  = 9 # change this based on depth of directory where ModelNet data is stored
#
# flist_mn10test = [line.rstrip('\n') for line in open('test10.txt')]
# MNET10_test = []
# MNET10_test_orients = []
# MNET10_test_labels = []
# for model in flist_mn10test:
#     tree = create_kd_tree(model)
#     MNET10_test.append(tree[0])
#     MNET10_test_orients.append(tree[1])
#     label = model.split('/')[loc_of_label]
#     MNET10_test_labels.append(LABELS_DICT[label])
#
#
# np.save('mnet10_test.npy',np.array(MNET10_test))
# np.save('mnet10_test_orients.npy',np.array(MNET10_test_orients))
# np.save('mnet10_test_labels.npy',np.array(MNET10_test_labels))
