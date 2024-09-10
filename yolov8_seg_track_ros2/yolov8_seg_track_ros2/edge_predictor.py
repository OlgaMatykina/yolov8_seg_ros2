from vlsat.src.utils.config import Config
from vlsat.src.model.model import MMGNet
from vlsat.src.utils import op_utils
import torch
from itertools import product
import numpy as np


class EdgePredictor:
    def __init__(self, config_path, ckpt_path, relationships_list):
        self.config = Config(config_path)
        self.config.exp = ckpt_path
        self.config.MODE = "eval"
        self.padding = 0.2
        self.model = MMGNet(self.config)
        # init device
        if torch.cuda.is_available() and len(self.config.GPU) > 0:
            self.config.DEVICE = torch.device("cuda")
        else:
            self.config.DEVICE = torch.device("cpu")
        self.model.load(best=True)
        self.model.model.eval()
        with open(relationships_list, "r") as f:
            self.relationships_list = f.readlines()
        
        self.rel_id_to_rel_name = {
            i: name.strip()
            for i, name in enumerate(self.relationships_list[1:])
        }

    def preprocess_poinclouds(self, points, num_points):
        assert len(points) > 1, "Number of objects should be at least 2"
        edge_indices = list(product(list(range(len(points))), list(range(len(points)))))
        edge_indices = [i for i in edge_indices if i[0]!=i[1]]

        num_objects = len(points)
        dim_point = points[0].shape[-1]

        instances_box = dict()
        obj_points = torch.zeros([num_objects, num_points, dim_point])
        descriptor = torch.zeros([num_objects, 11])

        obj_2d_feats = np.zeros([num_objects, 512])

        for i, pcd in enumerate(points):
            # get node point
            min_box = np.min(pcd, 0) - self.padding
            max_box = np.max(pcd, 0) + self.padding
            instances_box[i] = (min_box, max_box)
            choice = np.random.choice(len(pcd), num_points, replace=True)
            pcd = pcd[choice, :]
            descriptor[i] = op_utils.gen_descriptor(torch.from_numpy(pcd))
            pcd = torch.from_numpy(pcd.astype(np.float32))
            pcd = self.zero_mean(pcd)
            obj_points[i] = pcd

        edge_indices = torch.tensor(edge_indices, dtype=torch.long).permute(1, 0)
        obj_2d_feats = torch.from_numpy(obj_2d_feats.astype(np.float32))    
        obj_points = obj_points.permute(0, 2, 1)
        batch_ids = torch.zeros((num_objects, 1))
        return obj_points, obj_2d_feats, edge_indices, descriptor, batch_ids

    def predict_relations(self, obj_points, obj_2d_feats, edge_indices, descriptor, batch_ids):
        obj_points = obj_points.to(self.config.DEVICE)
        obj_2d_feats = obj_2d_feats.to(self.config.DEVICE)
        edge_indices = edge_indices.to(self.config.DEVICE)
        descriptor = descriptor.to(self.config.DEVICE)
        batch_ids = batch_ids.to(self.config.DEVICE)
        with torch.no_grad():
            rel_cls_3d = self.model.model(
                obj_points, obj_2d_feats, edge_indices, descriptor, batch_ids=batch_ids
            )
        return rel_cls_3d

    def save_relations(self, tracking_ids, timestamps, class_names, predicted_relations, edge_indices):
        saved_relations = []
        for k in range(predicted_relations.shape[0]):
            idx_1 = edge_indices[0][k].item()
            idx_2 = edge_indices[1][k].item()

            id_1 = tracking_ids[idx_1]
            id_2 = tracking_ids[idx_2]

            timestamp_1 = timestamps[idx_1]
            timestamp_2 = timestamps[idx_2]

            class_name_1 = class_names[idx_1]
            class_name_2 = class_names[idx_2]

            rel_id = torch.argmax(predicted_relations, dim=1)[k].item()
            rel_name = self.rel_id_to_rel_name[rel_id]

            rel_dict = {
                "id_1": id_1,
                "timestamp_1": timestamp_1,
                "class_name_1": class_name_1,
                "rel_name": rel_name,
                "id_2": id_2,
                "timestamp_2": timestamp_2,
                "class_name_2": class_name_2,
                "rel_id": rel_id,
                
            }
            saved_relations.append(rel_dict)

        return saved_relations

    def zero_mean(self, point):
        mean = torch.mean(point, dim=0)
        point -= mean.unsqueeze(0)
        return point