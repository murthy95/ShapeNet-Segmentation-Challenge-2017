'''
Run in Python 3; list.clear() is used
'''

import numpy as np

def read_pts(fname):
    with open(fname,'r') as myfile:
        data = myfile.readlines()
    return np.loadtxt(data)

def read_labels(fname):
    with open(fname,'r') as myfile:
        data = myfile.readlines()
    return np.loadtxt(data).reshape((len(data),1))

def create_kd_tree(pts):
    # to keep track of which points the leaves correspond to
    inds = np.array(range(len(pts)))
    ind_nodes = [inds]
    # find the kd-tree leafs
    kd_orients = []
    nodes = [pts]
     # 'nodes' would be sets of points, whose size reduces as the loop runs, ultimately having leaves
    child_list = []  # this will store the 'children' for an iteration, and replace node after the iteration
    ind_child_list = []
    while len(nodes[0]) != 1:
        child_list.clear()
        ind_child_list.clear()
        for node,ind_node in zip(nodes,ind_nodes):  # split each node
            ranges = np.amax(node,axis = 0) - np.amin(node,axis = 0)
            split_dir = np.argmax(ranges)
            kd_orients.append(split_dir)
            # indices to be used for sorting; will use it on ind_of_original as well
            sort_ind = node[:,split_dir].argsort()

            sorted_node = node[sort_ind]
            sorted_ind_node = ind_node[sort_ind]

            # checking if splitting into two is possible; if not, duplicate
            num_pts = len(sorted_node)
            if(num_pts%2 == 0):
                first_split_end = np.int(num_pts/2)
            else:   # middle point will be sent into both children
                first_split_end = np.int(num_pts/2)+1

            child_list.append(sorted_node[0:first_split_end,:])
            child_list.append(sorted_node[np.int(num_pts/2):num_pts,:])
            ind_child_list.append(sorted_ind_node[0:first_split_end])
            ind_child_list.append(sorted_ind_node[np.int(num_pts/2):num_pts])
        nodes.clear()
        nodes.extend(child_list)
        ind_nodes.clear()
        ind_nodes.extend(ind_child_list)

    kd_leaves = np.array(nodes).reshape((len(nodes),3))
    kd_orients = np.array(kd_orients)
    kd_inds = np.array(ind_nodes).reshape((len(ind_nodes),))
    # return (kd_leaves,kd_orients)
    return kd_leaves,kd_inds
