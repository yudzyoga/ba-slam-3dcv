import json
import glob
import cv2
import os
import numpy as np

class utils:
    def __init__(self):
        pass

    @staticmethod
    def import_json(path):
        with open(path, 'r') as openfile:
            json_object = json.load(openfile)
        return json_object

    @staticmethod
    def export_json(path, json_data):
        with open(path, "w") as outfile:
            json.dump(json_data, outfile)

    @staticmethod
    def read_image(path, bgr=False):
        img = cv2.imread(path)
        if bgr:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    @staticmethod
    def read_txt(path):
        data = None
        with open(path, "r") as openfile:
            data = openfile.readlines()
        return data

    @staticmethod
    def save_txt(path, data_list):
        with open(path, 'w') as fp:
            for item in data_list:
                # write each item on a new line
                fp.write("%s\n" % item)

    @staticmethod
    def get_dataset_info(dataset_name, PROJECT_DIR, detailed=False):
        # set empty
        dataset = {
            "data": {
                "path": "",
                "image": "",
                "correspondence": "None"
            }, 
            "parameters": {}            
        }

        dataset_path = os.path.join(PROJECT_DIR, "dataset", dataset_name)
        info = utils.import_json(os.path.join(dataset_path, "info.json"))
        dataset["data"]["path"] = dataset_path
        
        if detailed:
            # load data
            dataset["data"]["image"] = glob.glob(os.path.join(dataset_path, info["data"]["img"], "*.jpg"))
            if "correspondence" not in list(info["data"].keys()):
                info["data"]["correspondences"] = os.path.join(dataset_path, "correspondences")
                try:
                    os.mkdir(info["data"]["correspondences"])
                except:
                    pass
                dataset["data"]["correspondence"] = []
            else:
                dataset["data"]["correspondence"] = sorted(glob.glob(os.path.join(info["data"]["correspondences"], "*.txt")))
        else:
            dataset["data"]["path"] = dataset_path
            dataset["data"]["image"] = os.path.join(dataset_path, info["data"]["img"])
            if "correspondences" not in info["data"].keys():
                info["data"]["correspondences"] = os.path.join(dataset_path, "correspondences")
                try:
                    os.mkdir(info["data"]["correspondences"])
                except:
                    pass
                dataset["data"]["correspondence"] = info["data"]["correspondences"]
            else:
                dataset["data"]["correspondence"] = info["data"]["correspondences"]

        # load other details
        for k, v in info["parameters"].items():
            if "json" in v:
                dataset["parameters"][k] = utils.import_json(os.path.join(dataset_path, v))
            else:
                dataset["parameters"][k] = info["parameters"][k]
        return dataset
    
    @staticmethod
    def is_correspondence_exist(PROJECT_DIR, dataset_name, idx_l, idx_r):
        path1 = os.path.join(PROJECT_DIR, "dataset", dataset_name, "correspondences", f"{idx_l}_{idx_r}.txt")
        path2 = os.path.join(PROJECT_DIR, "dataset", dataset_name, "correspondences", f"{idx_r}_{idx_l}.txt")
        return os.path.exists(path1), os.path.exists(path2), os.path.exists(path1) or os.path.exists(path2)

    @staticmethod
    def triangulate_point(origin1, origin2, direction1, direction2):
        # # centers
        # p = np.array([2, 0, 0])
        # q = np.array([-2, 0, 0])

        # # directions
        # r = np.array([-1, 1, 0])
        # s = np.array([1, 1, 0])

        """
            # centers
            p = origin1, q = origin2
            # directions
            r = direction1, s = direction2

            f = p + lmbd * r
            g = q + mu * s
        """
        p, q, r, s = origin1, origin2, direction1, direction2
        r = r / np.linalg.norm(r)
        s = s / np.linalg.norm(s)

        A = np.array([
            [r.T.dot(r), -s.T.dot(r)],
            [r.T.dot(s), -s.T.dot(s)]
        ])

        b = np.array([
            (q-p).T.dot(r),
            (q-p).T.dot(s)
        ])

        lmbd, mu = np.linalg.inv(A).dot(b)
        
        f = p + lmbd * r
        g = q + mu * s
        h = 0.5 *  (f + g)
        dist = np.linalg.norm(f - g)
        return h, dist
    