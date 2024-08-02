"""
Script to run gradslam over a sequence from ICL and visualize self-similarity
over fused feature maps (assumes we point-and-click a 3D map point to compute
self-similarity scores againt the other points in the map).
"""

import json
import os
import pathlib
from dataclasses import dataclass
from typing import List, Union


import open3d as o3d
import pandas as pd
import numpy as np

import open_clip

import torch
import tyro
from gradslam.structures.pointclouds import Pointclouds

from pytorch3d.ops import ball_query, knn_points
from typing_extensions import Literal
import pandas as pd

torch.set_grad_enabled(False)

colors = {
    0:[252, 65, 3],
    3:[252, 248, 3],
    11:[90, 252, 3],
    12:[3, 252, 194],
    13:[3, 223, 252],
    18:[3, 49, 252],
    19:[152, 3, 252],
    20:[231, 3, 252],
    29:[252, 3, 123],
    31:[36, 32, 32],
    37:[219, 206, 206],
    40:[255, 252, 222],
    44:[222, 255, 223],
    47:[222, 255, 251],
    59:[222, 232, 255],
    60:[245, 222, 255],
    63:[133, 53, 73],
    64:[53, 80, 133],
    65:[53, 133, 88],
    76:[133, 110, 53],
    78:[30, 230, 173],
    79:[180, 180, 180],
    80:[180, 0, 60],
    91:[180, 180, 0],
    92:[60, 180, 60],
    93:[60, 210, 60],
    95:[90, 180, 60],
    97:[60, 180, 60],
    98:[60, 210, 60],
}

def compute_confmatrix(
    labels_pred, labels_gt, idx_pred_to_gt, idx_gt_to_pred, class_names
):
    labels_gt = labels_gt[idx_pred_to_gt]
    num_classes = len(class_names)

    confmatrix = torch.zeros(num_classes, num_classes,
                             device=labels_pred.device)
    for class_gt_int in range(num_classes):
        tensor_gt_class = torch.eq(labels_gt, class_gt_int).long()
        for class_pred_int in range(num_classes):
            tensor_pred_class = torch.eq(labels_pred, class_pred_int).long()
            tensor_pred_class = torch.mul(tensor_gt_class, tensor_pred_class)
            count = torch.sum(tensor_pred_class)
            confmatrix[class_gt_int, class_pred_int] += count

    return confmatrix

def compute_metrics(confmatrix, class_names):
    if isinstance(confmatrix, torch.Tensor):
        confmatrix = confmatrix.cpu().numpy()

    num_classes = len(class_names)
    ious = np.zeros((num_classes))
    precision = np.zeros((num_classes))
    recall = np.zeros((num_classes))
    f1score = np.zeros((num_classes))

    for _idx in range(num_classes):
        ious[_idx] = confmatrix[_idx, _idx] / (
            max(
                1,
                confmatrix[_idx, :].sum()
                + confmatrix[:, _idx].sum()
                - confmatrix[_idx, _idx],
            )
        )
        recall[_idx] = confmatrix[_idx, _idx] / \
            max(1, confmatrix[_idx, :].sum())
        precision[_idx] = confmatrix[_idx, _idx] / \
            max(1, confmatrix[:, _idx].sum())
        f1score[_idx] = (
            2 * precision[_idx] * recall[_idx] /
            max(1, precision[_idx] + recall[_idx])
        )

    fmiou = (ious * confmatrix.sum(1) / confmatrix.sum()).sum()
    print(f"iou: {ious}")
    print(f"miou: {ious.mean()}")
    print(f"Acc>0.15: {(ious > 0.15).sum()}")
    print(f"Acc>0.25: {(ious > 0.25).sum()}")
    print(f"Acc>0.50: {(ious > 0.50).sum()}")
    print(f"Acc>0.75: {(ious > 0.75).sum()}")
    print(f"precision: {precision}")
    print(f"recall: {recall}")
    print(f"f1score: {f1score}")

    mdict = {}
    mdict["iou"] = ious.tolist()
    mdict["miou"] = ious.mean().item()
    mdict["fmiou"] = fmiou.item()
    mdict["num_classes"] = num_classes
    mdict["acc0.15"] = (ious > 0.15).sum().item()
    mdict["acc0.25"] = (ious > 0.25).sum().item()
    mdict["acc0.50"] = (ious > 0.50).sum().item()
    mdict["acc0.75"] = (ious > 0.75).sum().item()
    mdict["precision"] = precision.tolist()
    mdict["recall"] = recall.tolist()
    mdict["f1score"] = f1score.tolist()
    mdict["class_names"] = class_names

    return mdict
def compute_pred_gt_associations(pred, gt):
    # pred: predicted pointcloud
    # gt: GT pointcloud
    from pytorch3d.ops import ball_query, knn_points

    # pred = pointclouds.points_padded.cuda().contiguous()
    # gt = pts_gt.unsqueeze(0).cuda().contiguous()
    b, l, d = pred.shape
    lengths_src = torch.ones(b, dtype=torch.long, device=pred.device) * l
    b, l, d = gt.shape
    lengths_tgt = torch.ones(b, dtype=torch.long, device=pred.device) * l
    src_nn = knn_points(
        pred,
        gt,
        lengths1=lengths_src,
        lengths2=lengths_tgt,
        return_nn=True,
        return_sorted=True,
        K=1,
    )
    idx_pred_to_gt = src_nn.idx.squeeze(0).squeeze(-1)
    tgt_nn = knn_points(
        gt,
        pred,
        lengths1=lengths_tgt,
        lengths2=lengths_src,
        return_nn=True,
        return_sorted=True,
        K=1,
    )
    idx_gt_to_pred = tgt_nn.idx.squeeze(0).squeeze(-1)

    return idx_pred_to_gt, idx_gt_to_pred

@dataclass
class ProgramArgs:
    """Commandline args for this script"""

    # Path to saved pointcloud to visualize

    device: str = "cuda:0"

    # Similarity computation and visualization params
    viz_type: Literal["topk", "thresh"] = "thresh"
    similarity_thresh: float = 0.6
    topk: int = 10000

    # CLIP model config
    open_clip_model = "ViT-H-14"
    open_clip_pretrained_dataset = "laion2b_s32b_b79k"

    def to_dict(self) -> dict:
        """Convert the ProgramArgs object to a dictionary"""
        attrs = {}
        for attr in vars(self):
            temp = getattr(self, attr)
            if isinstance(temp, pathlib.PosixPath):
                temp = str(temp)
            attrs[attr] = temp
        return attrs

    def to_json(self) -> str:
        """Convert the ProgramArgs object to a JSON string"""
        return json.dumps(self.to_dict(), indent=2)


if __name__ == "__main__":

    print(torch.cuda.is_available())
    # Process command-line args (if any)
    args = tyro.cli(ProgramArgs)

    class_names = [
    "other", #
    "backpack",
    "base cabinet",
    "basket",
    "bathtub",
    "beam",
    "beanbag",
    "bed",
    "bench",
    "bike",
    "bin",
    "blanket",
    "blinds",
    "book",
    "bottle",
    "box",
    "bowl",
    "camera",
    "cabinet",
    "candle",
    "chair",
    "chopping board",
    "clock",
    "cloth",
    "clothing",
    "coaster",
    "comforter",
    "computer keyboard",
    "cup",
    "pillow",
    "curtain",
    "ceiling",
    "cooktop",
    "countertop",
    "desk",
    "desk organizer",
    "desktop computer",
    "door",
    "exercise ball",
    "faucet",
    "floor",
    "handbag",
    "hair dryer",
    "handrail",
    "indoor plant",
    "knife block",
    "kitchen utensil",
    "lamp",
    "laptop",
    "major appliance",
    "mat",
    "microwave",
    "monitor",
    "mouse",
    "nightstand",
    "pan",
    "panel",
    "paper towel",
    "phone",
    "picture",
    "pillar",
    "pillow",
    "pipe",
    "plant stand",
    "plate",
    "pot",
    "rack",
    "refrigerator",
    "remote control",
    "scarf",
    "sculpture",
    "shelf",
    "shoe",
    "shower stall",
    "sink",
    "small appliance",
    "sofa",
    "stair",
    "stool",
    "switch",
    "table",
    "table runner",
    "tablet",
    "tissue paper",
    "toilet",
    "toothbrush",
    "towel",
    "tv screen",
    "tv stand",
    "umbrella",
    "utensil holder",
    "vase",
    "vent",
    "wall",
    "wall cabinet",
    "wall plug",
    "wardrobe",
    "window",
    "rug",
    "logo",
    "bag",
    "set of clothing",
]

    object_map = [3,11,12,13,18,19,20,29,31,37,40,44,47,59,60,63,64,65,76,78,79,80,91,92, 93, 95, 97, 98]
    print(f"size classes: {len(class_names )}, size objs: {len(object_map )}")
    # Prompt user whether or not to continue
    # print("Excluding classes: ", [(i, class_names[i]) for i in classes_id])

    # print(open_clip.list_pretrained())
    # Compute the CLIP embedding for each class
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k")
    # clip_model, clip_preprocess = clip.load(
    #     "ViT-L/14", "laion2b_s32b_b79k"
    # )
    clip_model = clip_model
    clip_model.eval()
    clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

    scenes = ["office0", "office1", "office2", "office3",
              "office4", "room0", "room1", "room2"]
    scenes = ["room1"]

    replica_semantic_root = "/mnt/hdd5/Datasets/Replica"

    # Get the set of classes that are used for evaluation
    all_class_index = np.arange(len(class_names))

    conf_matrices = {}
    conf_matrix_all = 0
    gt_class_id = set()
    for scene in scenes:

        

        args.load_path = "/datasets/Replica/" + scene + "-map-test"
        cf_pose_path = f"/datasets/Replica/{scene}/traj.txt"
        cf_poses = np.loadtxt(cf_pose_path)
        cf_poses = torch.from_numpy(cf_poses.reshape(-1, 4, 4)).float()
        
        # gt_mesh = pytorch3d.io.load_ply(gt_pc_path)
        # gt_xyz = gt_mesh[0]
        pointclouds = Pointclouds.load_pointcloud_from_h5(args.load_path)
        cf_xyz = pointclouds.points_padded[0]
        cf_xyz = cf_xyz @ cf_poses[0, :3, :3].t() + cf_poses[0, :3, 3]

        model, _, _ = open_clip.create_model_and_transforms(
            args.open_clip_model, args.open_clip_pretrained_dataset
        )

        model.eval()
        # Normalize the map
        map_embeddings = pointclouds.embeddings_padded
        map_embeddings_norm = torch.nn.functional.normalize(
            map_embeddings, dim=2)

      

        # pointclouds = Pointclouds.load_pointcloud_from_h5(gt_path)

        # gt_xyz = pointclouds.points_padded[0]
        scene_id_ = scene[:-1]+"_"+ scene[-1]
        gt_pc_path = f"/datasets/Replica/gt_pcd/{scene_id_}/Sequence_1/saved-maps-gt"
        gt_pose_path = f"/datasets/Replica/gt_pcd/{scene_id_}/Sequence_1/traj_w_c.txt"

        gt_map = Pointclouds.load_pointcloud_from_h5(gt_pc_path)
        gt_poses = np.loadtxt(gt_pose_path)
        gt_poses = torch.from_numpy(gt_poses.reshape(-1, 4, 4)).float()

        gt_xyz = gt_map.points_padded[0]
        gt_color = gt_map.colors_padded[0]
        gt_embedding = gt_map.embeddings_padded[0]  # (N, num_class)
        gt_class = gt_embedding.argmax(dim=1)  # (N,)
        local_ids = sorted(gt_class.unique().cpu().numpy())[1:]
        local_classes = []
        for ind in local_ids:
            local_classes.append(class_names[ind])  

        dict_classes = {}
        for ind, id_local in enumerate(local_ids):
            dict_classes[ind] = id_local
        prompts = [f"There is the {c} in the scene." for c in local_classes]

        # text = clip_tokenizer(prompts)
        # text = text
        # class_feats = clip_model.encode_text(text)
        # class_feats /= class_feats.norm(dim=-1, keepdim=True)  # (num_classes, D)
        # object_class_sim = map_embeddings_norm.float(
        # ) @ class_feats.T  # (num_objects, num_classes)
        # gt_class = class_all2existing[gt_class]  # (N,)
        # assert gt_class.min() >= 0
        # assert gt_class.max() < len(REPLICA_EXISTING_CLASSES)

        # transform pred_xyz and gt_xyz according to the first pose in gt_poses
        gt_xyz = gt_xyz @ gt_poses[0, :3, :3].t() + gt_poses[0, :3, 3]


        cf_pcd = o3d.geometry.PointCloud()
        cf_pcd.points = o3d.utility.Vector3dVector(cf_xyz.cpu().numpy())
        cf_pcd.colors = o3d.utility.Vector3dVector([[1, 0 , 0] for i in range(len(cf_pcd.points))])

        gt_pc = o3d.geometry.PointCloud()
        gt_pc.points = o3d.utility.Vector3dVector(gt_xyz.cpu().numpy())
        gt_pc.colors = o3d.utility.Vector3dVector([[0, 1 , 0] for i in range(len(gt_pc.points))])

        o3d.visualization.draw_geometries([gt_pc, cf_pcd])



        all_class_index = np.arange(len(class_names))
        # ignore_index = np.asarray([0, 3, 12, 13, 19, 20, 29, 37, 40, 44, 47, 59, 60, 63, 64, 65, 79, 80, 91, 92, 95, 97, 98])
        ignore_index = np.asarray([0])
        existing_index = gt_class.unique().cpu().numpy()
        print("Excluding classes: ", [(i) for i in existing_index])
        non_existing_index = np.setdiff1d(all_class_index, existing_index)
        ignore_index = np.append(ignore_index, non_existing_index)
        print(ignore_index)
        # object_class_sim[:, ignore_index] = -1e6
        # print(object_class_sim.shape)
        pred_class = object_class_sim.argmax(dim=-1).cpu()
        # print(pred_class.shape)
        pred_class = pred_class.squeeze(0)
        for i in range(pred_class.shape[0]):
            # print(int(pred_class.numpy()[i]))
            pred_class[i] = object_map[int(pred_class.numpy()[i])]

        pred_class = pred_class.unsqueeze(0) 
        pred_xyz = pointclouds.points_padded[0]

        pred_pcd = o3d.geometry.PointCloud()
        pred_pcd.points = o3d.utility.Vector3dVector(pred_xyz.numpy())
        colors_class = []
        keep_index = np.setdiff1d(all_class_index, ignore_index)

        print(
            f"{len(keep_index)} classes remains. They are: ",
            [(i, class_names[i]) for i in keep_index],
        )
        print(pred_class.shape)
        # while True :
        #     ind = int(input())
        #     colors_class = []
        #     for cl in pred_class.squeeze(0).numpy():
        #         if cl == ind:
        #             # print("ignore")
        #             colors_class.append([255,0, 255])
        #         else:
        #             colors_class.append([0, 0, 255])
        #     pred_pcd.colors = o3d.utility.Vector3dVector(np.array(colors_class)/255)
        #     o3d.visualization.draw_geometries([pred_pcd])
        for cl in pred_class.squeeze(0).numpy():
            if cl in ignore_index:
                # print("ignore")
                colors_class.append(colors[0])
            else:
                colors_class.append(colors[cl])
        pred_pcd.colors = o3d.utility.Vector3dVector(np.array(colors_class)/255)
        o3d.visualization.draw_geometries([pred_pcd])

        gt_pcd = o3d.geometry.PointCloud()
        gt_pcd.points = o3d.utility.Vector3dVector(gt_xyz.numpy())
        gt_class_id.update(gt_class.squeeze(0).numpy())
        print(gt_class_id)
        # colors_class = []
        # # print(gt_class.shape)
        # for cl in gt_class.squeeze(0).numpy():
        #     if cl in ignore_index:
        #         # print("ignore")
        #         colors_class.append(colors[0])
        #     else:
        #         colors_class.append(colors[cl])
        # gt_pcd.colors = o3d.utility.Vector3dVector(np.array(colors_class)/255)
        # o3d.visualization.draw_geometries([gt_pcd])

    #     scene_id_ = scene[:-1]+"_"+ scene[-1]
    #     print(
    #             "Using only the classes that exists in GT of this scene: ",
    #             len(existing_index),
    #         )
    #     keep_index = np.setdiff1d(all_class_index, ignore_index)

    #     print(
    #         f"{len(keep_index)} classes remains. They are: ",
    #         [(i, class_names[i]) for i in keep_index],
    #     )
    #     slam_path = os.path.join(
    #     "/datasets/Replica", scene, "rgb_cloud"
    #     )
    #     slam_pointclouds = Pointclouds.load_pointcloud_from_h5(slam_path)
    #     slam_xyz = slam_pointclouds.points_padded[0]

    #     # To ensure fair comparison, build the prediction point cloud based on the slam results
    #     # Search for NN of slam_xyz in pred_xyz
    #     slam_nn_in_pred = knn_points(
    #         slam_xyz.unsqueeze(0).cuda().contiguous().float(),
    #         pred_xyz.unsqueeze(0).cuda().contiguous().float(),
    #         lengths1=None,
    #         lengths2=None,
    #         return_nn=True,
    #         return_sorted=True,
    #         K=1,
    #     )
    #     idx_slam_to_pred = slam_nn_in_pred.idx.squeeze(0).squeeze(-1)

    #     # # predicted point cloud in open3d
    #     # print("Before resampling")
    #     # pred_pcd = o3d.geometry.PointCloud()
    #     # pred_pcd.points = o3d.utility.Vector3dVector(pred_xyz.numpy())
    #     # pred_pcd.colors = o3d.utility.Vector3dVector(class2color[pred_class.numpy()])
    #     # o3d.visualization.draw_geometries([pred_pcd])

    #     # Resample the pred_xyz and pred_class based on slam_nn_in_pred
    #     pred_xyz = slam_xyz
    #     print(pred_class.shape)
    #     pred_class = pred_class.squeeze(0)[idx_slam_to_pred.cpu()]
    #     # pred_color = pred_color[idx_slam_to_pred.cpu()]

    #     # # predicted point cloud in open3d
    #     # print("After resampling")
    #     # pred_pcd = o3d.geometry.PointCloud()
    #     # pred_pcd.points = o3d.utility.Vector3dVector(pred_xyz.numpy())
    #     # colors_class = []
    #     # for cl in pred_class.numpy():
    #     #     if cl in ignore_index:
    #     #         # print("ignore")
    #     #         colors_class.append(colors[0])
    #     #     else:
    #     #         colors_class.append(colors[cl])
    #     # pred_pcd.colors = o3d.utility.Vector3dVector(np.array(colors_class))
    #     # o3d.visualization.draw_geometries([pred_pcd])

    #     # Compute the associations between the predicted and ground truth point clouds
    #     idx_pred_to_gt, idx_gt_to_pred = compute_pred_gt_associations(
    #         pred_xyz.unsqueeze(0).cuda().contiguous().float(),
    #         gt_xyz.unsqueeze(0).cuda().contiguous().float(),
    #     )

    #     # Only keep the points on the 3D reconstructions that are mapped to
    #     # GT point that is in keep_index
    #     label_gt = gt_class[idx_pred_to_gt.cpu()]
    #     # pred_keep_idx = torch.isin(label_gt, torch.from_numpy(keep_index))
    #     # pred_class = pred_class[pred_keep_idx]
    #     # idx_pred_to_gt = idx_pred_to_gt[pred_keep_idx]
    #     idx_gt_to_pred = None  # not to be used

    #     # Compute the confusion matrix
    #     confmatrix = compute_confmatrix(
    #         pred_class.cuda(),
    #         gt_class.cuda(),
    #         idx_pred_to_gt,
    #         idx_gt_to_pred,
    #         class_names,
    #     )

    #     # assert confmatrix.sum(0)[ignore_index].sum() == 0
    #     assert confmatrix.sum(1)[ignore_index].sum() == 0


    #     conf_matrix = confmatrix.detach().cpu()
    #     conf_matrix_all += conf_matrix

    #     conf_matrices[scene] = {
    #         "conf_matrix": conf_matrix,
    #         "keep_index": keep_index
    #     }

    # # Remove the rows and columns that are not in keep_class_index
    # conf_matrices["all"] = {
    #     "conf_matrix": conf_matrix_all,
    #     "keep_index": conf_matrix_all.sum(axis=1).nonzero().reshape(-1)
    # }

    # results_scene = []
    # results_classes = []
    # for scene_id, res in conf_matrices.items():
    #     conf_matrix = res["conf_matrix"]
    #     keep_index = res["keep_index"]
    #     conf_matrix = conf_matrix[keep_index, :][:, keep_index]
    #     keep_class_names = [class_names[i] for i in keep_index]

    #     mdict = compute_metrics(conf_matrix, keep_class_names)
    #     results_scene.append(
    #         {
    #             "scene_id": scene_id,
    #             "miou": mdict["miou"],  # * 100.0,
    #             "mrecall": np.mean(mdict["recall"]),  # * 100.0,
    #             "mprecision": np.mean(mdict["precision"]),  # * 100.0,
    #             "mf1score": np.mean(mdict["f1score"]),  # * 100.0,
    #             "fmiou": mdict["fmiou"],  # * 100.0,
    #         }
    #     )
    #     dict_result = {"scene_id": scene_id}
    #     for id_name, name in enumerate(keep_class_names):
    #         dict_result[f"{name}_iou"] = mdict["iou"][id_name]
    #         dict_result[f"{name}_recall"] = mdict["recall"][id_name]
    #         # dict_result[f"{name}_recall"] = mdict["recall"][id_name]
    #     results_classes.append(dict_result)

    # df_result = pd.DataFrame(results_scene)
    # # print(df_result)
    # df_result.to_csv(f"result_scene.csv", index=False)
    # df_result = pd.DataFrame(results_classes)
    # df_result.to_csv(f"result_classes.csv", index=False)




