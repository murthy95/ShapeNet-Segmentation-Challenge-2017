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

def get_fid(s):
    splits = s.split('/')
    fname = splits[len(splits)-1]
    return fname.split('.')[0]

if len(sys.argv)>1:
    category = sys.argv[1]
    category_id = id_name_map[category]
    exclusions = sys.argv[2].split(',')
    labels_to_exclude = [int(e) for e in exclusions]

    fraction = int(sys.argv[3])

    data_path = "./" + category + "/data/train_label/" + category_id + "/*"
    files = sorted(glob.glob(data_path))

    num_files = len(files)
    num_files_to_generate = int(num_files*fraction)

    # num_files_to_generate = 1

    count,i = 0,0
    inds = np.random.permutation(num_files)


    while count<num_files_to_generate and i<num_files:
        label_file = files[inds[i]]
        labels = np.reshape(read_labels(label_file),(-1,))
        reqd_inds = []
        for k in range(len(labels)):
            exclude = False
            for j in range(len(labels_to_exclude)):
                if labels[k] == labels_to_exclude[j]:
                    exclude = True
                    break
            if exclude == False:
                reqd_inds.append(k)

        reqd_inds = np.array(reqd_inds,dtype=int)
        datafile_prefix = "./" + category + "/data/train_data/" + category_id + "/" + get_fid(label_file)
        labelfile_prefix = "./" + category + "/data/train_label/" + category_id + "/" + get_fid(label_file)

        if len(reqd_inds.flatten())>512:
            count += 1

            pts = read_pts(datafile_prefix + ".pts")


            new_pts = pts[reqd_inds]
            new_labels = labels[reqd_inds]

            np.savetxt(datafile_prefix + "_minor.pts", new_pts, delimiter=' ', newline='\r\n', fmt='%f')
            np.savetxt(labelfile_prefix + "_minor.seg",new_labels,fmt='%1.f')

            # np.savetxt("_minor.pts", new_pts, delimiter=' ', newline='\r\n', fmt='%f')
            # np.savetxt("_minor.seg",new_labels,fmt='%1.f')

            print(datafile_prefix + "_minor")
        i+=1

    print(str.format("Requested {0} files",num_files_to_generate))
    print(str.format("Generated {0} files",count))
