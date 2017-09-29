from __future__ import division
from kd_helpers import *

import numpy as np

import sys, glob

id_name_map = {
 "Airplane":"02691156",
 "Bag":"02773838",
 "Cap":"02954340",
 "Car":"02958343",
 "Chair":"03001627",
 "Earphone":"03261776",
 "Guitar":"03467517",
 "Knife":"03624134",
 "Lamp":"03636649",
 "Laptop":"03642806",
 "Motorbike":"03790512",
 "Mug":"03797390",
 "Pistol":"03948459",
 "Rocket":"04099429",
 "Skateboard":"04225987",
 "Table":"04379243"}

num_parts = {
"Airplane":4,
"Bag":2 ,
"Cap":2 ,
"Car":4 ,
"Chair":4 ,
"Earphone":3 ,
"Guitar":3 ,
"Knife":2 ,
"Lamp": 4,
"Laptop":2 ,
"Motorbike":6 ,
"Mug":2 ,
"Pistol":3 ,
"Rocket":3 ,
"Skateboard":3 ,
"Table":3 }

if len(sys.argv)>1:
    category = sys.argv[1]
    category_id = id_name_map[category]

    npy_path = './data/prepared_old_train/' + category + '_' + category_id + '_y_train.npy'
    Y = np.load(npy_path)
    Y_flat = Y.flatten()

    data_path = "./data/train_label/" + category_id + "/*"
    files = sorted(glob.glob(data_path))

    num_files = len(files)
    data_file = files[np.random.randint(num_files)]
    labels = read_labels(data_file)

    for i in range(num_parts[category]):
        points = (labels == (i+1))
        as_int = [int(p) for p in points]
        print(str.format("Number of points for part {0} : {1}",i+1,sum(as_int)))

    tot = np.zeros(num_parts[category])
    vals = np.zeros(num_parts[category])
    for i in range(num_parts[category]):
        tot[i] = len(np.where(Y_flat == (i+1))[0])
        tot_where_present = 0
        for j in range(len(Y)):
            if len(np.where(Y[j].flatten() == (i+1))[0])>0:
                tot_where_present += len(Y[j].flatten())

        print(str.format("Total points for part {0} : {1}",i+1,tot[i]))
        print(str.format("Total points it's where present {0}",tot_where_present))
        vals[i] = (float)(tot[i])/tot_where_present
    print("Values : " + str(vals))
