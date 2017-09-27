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

     # 'nodes' would be sets of points, whose size reduces as the loop runs, ultimately having leaves
    nodes = [pts]
    # this will store the 'children' for an iteration, and replace node after the iteration
    child_list = []
    ind_child_list = []

    while len(nodes[0]) != 1:
        child_list.clear()
        ind_child_list.clear()
        for node,ind_node in zip(nodes,ind_nodes):  # split each node
            if len(node) == 1: # it was the middle of a subtree of odd numbers
                child_list.append(node)
                ind_child_list.append(ind_node)
                continue

            ranges = np.amax(node,axis = 0) - np.amin(node,axis = 0)
            split_dir = np.argmax(ranges)
            # indices to be used for sorting; will use it on ind_of_original as well
            sort_ind = node[:,split_dir].argsort()

            sorted_node = node[sort_ind]
            sorted_ind_node = ind_node[sort_ind]

            # checking if splitting into two is possible; if not, duplicate
            num_pts = len(sorted_node)
            half1_end = np.int(num_pts/2)

            odd_num = False
            if(num_pts%2 == 0):
                half2_start = half1_end
            else:   # middle point will be sent into both children
                half2_start = half1_end + 1
                odd_num = True

            child_list.append(sorted_node[0:half1_end,:])
            ind_child_list.append(sorted_ind_node[0:half1_end])
            if odd_num:
                child_list.append(sorted_node[half1_end,:].reshape((1,3)))
                ind_child_list.append(sorted_ind_node[half1_end])
            child_list.append(sorted_node[half2_start:num_pts,:])
            ind_child_list.append(sorted_ind_node[half2_start:num_pts])
        nodes.clear()
        nodes.extend(child_list)
        ind_nodes.clear()
        ind_nodes.extend(ind_child_list)

    kd_leaves = np.array(nodes).reshape((len(nodes),3))
    kd_inds = np.array(ind_nodes).reshape((len(ind_nodes),))
    return kd_leaves,kd_inds
