import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import svd
from mmcv.runner import force_fp32
from torch.nn import functional as F
from torch.nn.functional import smooth_l1_loss, l1_loss, mse_loss
from pcdet.models.backbones_3d.util.voxelize import Voxelization
from spconv.pytorch.ops import get_indice_pairs_implicit_gemm
from spconv.core import ConvAlgo
from .voxel_encoder import DynamicScatterVFE
from pcdet.utils import loss_utils
from pcdet.ops.sst.transform import bbox3d2result
from pcdet.ops.sst.sst_ops import get_inner_win_inds, scatter_v2
from pcdet.models.backbones_3d.depth_wise_conv import DepthSepaConv

eps = 1e-9


class MultiSubVoxelDynamicVoxelNetSSL(nn.Module):
    r"""VoxelNet using `dynamic voxelization <https://arxiv.org/abs/1910.06528>`_."""

    def __init__(
        self,
        loss_ratio_low,
        loss_ratio_med,
        loss_ratio_top,
        loss_ratio_low_nor,
        loss_ratio_med_nor,
        loss_ratio_top_nor,
        random_mask_ratio,
        voxel_size,
        point_cloud_range,
        grid_size,
        sub_voxel_ratio_low,
        sub_voxel_ratio_med,
        sub_voxel_size_low,
        sub_voxel_size_med,
        sub_voxel_size_top,
        loss=dict(type="SmoothL1Loss", reduction="mean", loss_weight=1.0),
        spatial_shape=[1, 400, 400],
        voxel_encoder=None,
        backbone=None,
        nor_usr_sml1=None,
        cls_loss_ratio_low=None,
        cls_loss_ratio_med=None,
        vis=False,
        cls_sub_voxel=True,
        normalize_sub_voxel=True,
        use_focal_mask=None,
        norm_curv=True,
        mse_loss=True,
        use_chamfer=True,
        mean_vfe=False
    ):
        super(MultiSubVoxelDynamicVoxelNetSSL, self).__init__()
        
        self.spatial_shape = spatial_shape

        self.nor_usr_sml1 = nor_usr_sml1
        self.norm_curv = norm_curv

        self.loss_ratio_med = loss_ratio_med
        self.loss_ratio_low = loss_ratio_low
        self.loss_ratio_top = loss_ratio_top

        self.loss_ratio_low_nor = loss_ratio_low_nor
        self.loss_ratio_med_nor = loss_ratio_med_nor
        self.loss_ratio_top_nor = loss_ratio_top_nor

        self.cls_loss_ratio_low = cls_loss_ratio_low
        self.cls_loss_ratio_med = cls_loss_ratio_med

        self.cls_sub_voxel = cls_sub_voxel
        self.vis = vis
        self.random_mask_ratio = random_mask_ratio
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.grid_size = grid_size
        self.mean_vfe = mean_vfe
        
        self.sub_voxel_size_low = sub_voxel_size_low
        self.sub_voxel_size_med = sub_voxel_size_med
        self.sub_voxel_ratio_low = sub_voxel_ratio_low
        self.sub_voxel_ratio_med = sub_voxel_ratio_med
        
        voxel_layer = dict(
            voxel_size=self.voxel_size,
            max_num_points=-1,
            point_cloud_range=self.point_cloud_range,
            max_voxels=(-1, -1),
        )
        sub_voxel_layer_low = dict(
            voxel_size=self.sub_voxel_size_low,
            max_num_points=-1,
            point_cloud_range=self.point_cloud_range,
            max_voxels=(-1, -1),
            )
        sub_voxel_layer_med = dict(
            voxel_size=self.sub_voxel_size_med,
            max_num_points=-1,
            point_cloud_range=self.point_cloud_range,
            max_voxels=(-1, -1),
            )
        
        self.voxel_layer = Voxelization(**voxel_layer)
        self.sub_voxel_layer_low = Voxelization(**sub_voxel_layer_low)
        self.sub_voxel_layer_med = Voxelization(**sub_voxel_layer_med)
        
        self.voxel_encoder = DynamicScatterVFE(
            num_point_features=4,
            point_cloud_range=self.point_cloud_range,
            voxel_size=self.voxel_size,
            feat_channels=[64, 128],
            with_distance=False,
            with_cluster_center=True,
            with_voxel_center=True,
            norm_cfg=dict(type="naiveSyncBN1d", eps=1e-3, momentum=0.01),
        )
    
        self.voxel_encoder_middle = DynamicScatterVFE(
            num_point_features=4,
            point_cloud_range=self.point_cloud_range,
            voxel_size=self.sub_voxel_size_med,
            feat_channels=[32, 64],
            with_distance=False,
            with_cluster_center=True,
            with_voxel_center=True,
            norm_cfg=dict(type="naiveSyncBN1d", eps=1e-3, momentum=0.01),
        )
        
        self.voxel_encoder_low = DynamicScatterVFE(
            num_point_features=4,
            point_cloud_range=self.point_cloud_range,
            voxel_size=self.sub_voxel_size_low,
            feat_channels=[32, 64],
            with_distance=False,
            with_cluster_center=True,
            with_voxel_center=True,
            norm_cfg=dict(type="naiveSyncBN1d", eps=1e-3, momentum=0.01),
        )
        
        self.drop_points_th = 100
        self.pred_dims= 3
        
        self.layer_norm = nn.LayerNorm(normalized_shape=224)
        
        self.reg_loss = loss
        self.use_chamfer = use_chamfer
        self.mse_loss = mse_loss
        self.use_focal_mask = use_focal_mask
        self.normalize_sub_voxel = normalize_sub_voxel
        self.build_losses()
        
    def build_losses(self):
        self.add_module(
            "cls_loss", loss_utils.CrossEntropyLoss(use_sigmoid=True, loss_weight=1.0)
        )
        # self.add_module('nor_loss', loss_utils.WeightedSmoothL1Loss())

    def forward(self, batch_dict, **kwargs):
        """Training forward function.
        Args:
            points (list[torch.Tensor]): Point cloud of each sample.
            img_metas (list[dict]): Meta information of each sample
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        if self.cls_sub_voxel:
            (
                x,
                centroid_low,
                centroid_low_mask,
                centroid_med,
                centroid_med_mask,
                centroid_high,
                centroid_normal_low,
                centroid_normal_med,
                centroid_normal_high,
            ) = self.extract_feat(batch_dict)
            return (
                x,
                centroid_low,
                centroid_low_mask,
                centroid_med,
                centroid_med_mask,
                centroid_high,
                centroid_normal_low,
                centroid_normal_med,
                centroid_normal_high,
            )
            
    def extract_feat(self, batch_dict, vis=False):
        """Extract features from points."""
        
        points = batch_dict["points"]
        batch_size = batch_dict["batch_size"]
        res_coors = []

        for batch_idx in range(batch_size):
            mask = points[:, 0] == batch_idx
            batched_points = points[mask, 1:]
            res_coors.append(batched_points)
    
        voxels, coors = self.voxelize(res_coors)
        sub_voxels_low, sub_coors_low = self.sub_voxelize_low(res_coors)
        sub_voxels_med, sub_coors_med = self.sub_voxelize_med(res_coors)
        
        voxel_features, feature_coors, low_level_point_feature, points_inverse_indices = self.voxel_encoder(voxels, coors, return_inv=True)
    
        if not self.mean_vfe:
            voxel_mid_features, feature_mid_coors = self.voxel_encoder_middle(sub_voxels_med, sub_coors_med)
            voxel_low_features, feature_low_coors = self.voxel_encoder_low(sub_voxels_low, sub_coors_low)
        else:
            voxel_mid_features, feature_mid_coors = self.voxel_encoder_middle(sub_voxels_med, sub_coors_med)
            voxel_low_features, feature_low_coors = self.voxel_encoder_low(sub_voxels_low, sub_coors_low)
        
        '''
        # merge 3 size voxel mean
        top_in_mid_indices = self.find_coordinate_indices(feature_coors, feature_mid_coors, mode="mid")
        top_in_low_indices = self.find_coordinate_indices(feature_coors, feature_low_coors, mode="low")
        
        feature_med_2 = voxel_mid_features[top_in_mid_indices]
        feature_low_2 = voxel_low_features[top_in_low_indices]
        
        feature_mid_coors_2 = feature_mid_coors[top_in_mid_indices]
        feature_low_coors_2 = feature_low_coors[top_in_low_indices]
        
        voxel_med_mean, new_med_coor, _ = scatter_v2(feature_med_2, feature_mid_coors_2, mode='max', fusion=True)
        voxel_low_mean, new_low_coor, _ = scatter_v2(feature_low_2, feature_low_coors_2, mode='max', fusion=True)
        
        merge_voxel_top_features = torch.cat([voxel_features, voxel_med_mean, voxel_low_mean], dim=1)
        '''
        #LN_feats = self.layer_norm(merge_voxel_top_features)
        
        '''
        # Hierarchical 3 size voxel mean
        mid_in_low_indices = self.find_coordinate_indices(feature_mid_coors, feature_low_coors, mode="mid")
        top_in_mid_indices = self.find_coordinate_indices(feature_coors, feature_mid_coors, mode="mid")
        
        low_to_mid_features = voxel_low_features[mid_in_low_indices]
        #mid_to_top_features = voxel_mid_features[top_in_mid_indices]
        
        feature_low_coors_2 = feature_low_coors[mid_in_low_indices]
        feature_mid_coors_2 = feature_mid_coors[top_in_mid_indices]
        
        low_to_mid_mean, _, _ = scatter_v2(low_to_mid_features, feature_low_coors_2, mode='avg', fusion=True)
        
        merge_voxel_mid_features = (voxel_mid_features + low_to_mid_mean) / 2
        
        mid_to_top_mean, _, _ = scatter_v2(merge_voxel_mid_features, feature_mid_coors_2, mode='avg', fusion=True)
        
        merge_voxel_top_features = (voxel_features + mid_to_top_mean) / 2
        '''
        
        # merge 3 size voxel mean for corss attention
        top_in_mid_indices = self.find_coordinate_indices(feature_coors, feature_mid_coors, mode="mid")
        top_in_low_indices = self.find_coordinate_indices(feature_coors, feature_low_coors, mode="low")
        
        feature_med_2 = voxel_mid_features[top_in_mid_indices]
        feature_low_2 = voxel_low_features[top_in_low_indices]
        
        feature_mid_coors_2 = feature_mid_coors[top_in_mid_indices]
        feature_low_coors_2 = feature_low_coors[top_in_low_indices]
        
        voxel_med_mean, new_med_coor, _ = scatter_v2(feature_med_2, feature_mid_coors_2, mode='avg', fusion=True)
        voxel_low_mean, new_low_coor, _ = scatter_v2(feature_low_2, feature_low_coors_2, mode='avg', fusion=True)
        
        merge_voxel_mid_low_features = torch.cat([voxel_med_mean, voxel_low_mean], dim=1)
        
        ids_keep, ids_mask = self.get_vanilla_mask_index(feature_coors, batch_size)
        device = voxel_features.device #depthwise_voxel_feat.device
        gt_dict = self.get_ground_truth(batch_size, device, low_level_point_feature, coors, feature_coors, voxel_features)
        gt_dict["ids_mask"] = ids_mask
        
        (
            centroids_low,
            centroid_voxel_coors_low,
            labels_count_low,
        ) = self.get_centroid_per_voxel(voxels[:, [2, 1, 0]], sub_coors_low)
        (
            centroids_med,
            centroid_voxel_coors_med,
            labels_count_med,
        ) = self.get_centroid_per_voxel(voxels[:, [2, 1, 0]], sub_coors_med)
        (
            centroids_high,
            centroid_voxel_coors_high,
            labels_count_high,
        ) = self.get_centroid_per_voxel(voxels[:, [2, 1, 0]], coors)

        (
            centroids_med_for_curv,
            centroid_mask_med_for_curv,
        ) = self.get_multi_voxel_id_to_tensor_id_for_curv(
            feature_coors.long(),
            centroid_voxel_coors_med.long(),
            centroids_med,
            batch_size,
        )

        (
            out_inds,
            indice_num_per_loc,
            pair,
            pair_bwd,
            pair_mask,
            pair_mask_bwd_splits,
            mask_argsort_fwd_splits,
            mask_argsort_bwd_splits,
            masks,
        ) = get_indice_pairs_implicit_gemm(
            indices=feature_coors,
            batch_size=batch_size,
            spatial_shape=self.grid_size,
            algo=ConvAlgo.MaskImplicitGemm,
            ksize=[1, 3, 3],
            stride=[1, 1, 1],
            padding=[0, 1, 1],
            dilation=[1, 1, 1],
            out_padding=[0, 0, 0],
            subm=True,
            transpose=False,
            is_train=False,
        )

        centroids_normal, centroids_curv = self.cal_regular_voxel_nor_and_curv(
            centroids_med_for_curv,
            centroid_mask_med_for_curv,
            centroids_high,
            pair.long(),
        )

        if self.normalize_sub_voxel is not None:
            centroids_low = self.normalize_centroid_sub_voxel(
                centroid_voxel_coors_low[:, 1:], centroids_low, layer="low"
            )
            centroids_med = self.normalize_centroid_sub_voxel(
                centroid_voxel_coors_med[:, 1:], centroids_med, layer="med"
            )
            centroids_high = self.normalize_centroid_sub_voxel(
                centroid_voxel_coors_high[:, 1:], centroids_high, layer="top"
            )

        (
            centroids_low,
            centroid_mask_low,
            centroids_med,
            centroid_mask_med,
        ) = self.get_multi_voxel_id_to_tensor_id_ori(
            feature_coors.long(),
            centroid_voxel_coors_low.long(),
            centroid_voxel_coors_med.long(),
            centroids_low,
            centroids_med,
            ids_mask,
            batch_size,
        )

        with torch.no_grad():
            centroids_high = centroids_high[ids_mask]
            centroids_normal = centroids_normal[ids_mask]
            centroids_curv = centroids_curv[ids_mask]

        centroids_normal_med = None
        centroids_normal_high = None

        mask_coors = feature_coors[ids_mask]
        if self.normalize_sub_voxel is None:
            centroids_low = self.normalize_centroid(mask_coors[:, 1:], centroids_low)
            centroids_med = self.normalize_centroid(mask_coors[:, 1:], centroids_med)
            centroids_high = self.normalize_centroid(mask_coors[:, 1:], centroids_high)

        if self.vis:
            return (
                x,
                centroids_low,
                centroid_mask_low,
                centroids_med,
                centroid_mask_med,
                centroids_high,
                mask_coors[:, 1:],
            )
        else:
            return (
                [
                    voxel_features[ids_keep],
                    feature_coors[ids_keep],
                    mask_coors,
                    batch_size,
                    merge_voxel_mid_low_features[ids_keep],
                    feature_mid_coors,
                    voxel_low_features,
                    feature_low_coors,
                    gt_dict
                ],
                centroids_low,
                centroid_mask_low,
                centroids_med,
                centroid_mask_med,
                centroids_high,
                centroids_normal,
                centroids_normal_med,
                centroids_normal_high,
            )
    
    def find_coordinate_indices(self, coords, coord_list, mode="mid"):
        if mode == "mid":
            coord_list[:, 1:] = torch.div(coord_list[:, 1:], 2).floor()
            indices = torch.where(torch.all(coords.unsqueeze(1) == coord_list, dim=2))[1].tolist()
        if mode == "low":
            coord_list[:, 1:] = torch.div(coord_list[:, 1:], 4).floor()
            indices = torch.where(torch.all(coords.unsqueeze(1) == coord_list, dim=2))[1].tolist()
        return indices
    
    def get_ground_truth(
        self,
        batch_size,
        device,
        low_level_point_feature,
        point_coors,
        voxel_coors,
        voxel_feats
        ):
        
        gt_dict = {}
        vx, vy, vz = self.grid_size[::-1]
        max_num_voxels = batch_size * vx * vy * vz

        point_indices = self.get_voxel_indices(point_coors)
        voxel_indices = self.get_voxel_indices(voxel_coors)

        # Get points per voxel
        if self.use_chamfer:
            points_rel_center = low_level_point_feature[:, -3:] # ce_x, ce_y, ce_z
            assert self.pred_dims in [2, 3], "Either use x and y or x, y, and z"
            points_rel_center = points_rel_center[:, : self.pred_dims].clone()
            pointr_rel_norm = 2 / torch.tensor(self.voxel_size, device=device).view(
                1, -1
            )
            points_rel_center = (
                points_rel_center * pointr_rel_norm #  -1과 1 사이의 범위로 정규화
            )  # x,y,z all in range [-1, 1]

            shuffle = torch.argsort(
                torch.rand(len(point_indices))
            )  # Shuffle to drop random points
            restore = torch.argsort(shuffle)
            inner_voxel_inds = get_inner_win_inds(point_indices[shuffle])[ # 복셀이 속한 윈도우 내에서의 상대적인 위치 정보를 계산하고 이를 반환
                restore
            ]  # fixes one index per point per voxel
            drop_mask = inner_voxel_inds < self.drop_points_th # 작은 값인 포인트를 제거하고, drop_mask를 생성

            points_rel_center = points_rel_center[drop_mask]
            inner_voxel_inds = inner_voxel_inds[drop_mask].long()
            dropped_point_indices = point_indices[drop_mask].long()

            gt_points = torch.zeros(
                (max_num_voxels, self.drop_points_th, 3),
                device=device,
                dtype=points_rel_center.dtype,
            ) # gt_points는 각 복셀에 대한 포인트의 상대적 위치 정보를 저장
            gt_points_padding = torch.ones(
                (max_num_voxels, self.drop_points_th), device=device, dtype=torch.long
            ) # gt_points_padding는 패딩 정보를 저장하며, 포인트가 패딩되지 않은 경우 0, 패딩된 경우 1의 값을 가짐
            
            gt_points[dropped_point_indices, inner_voxel_inds] = points_rel_center
            
            gt_points_padding[dropped_point_indices, inner_voxel_inds] = 0  # not_padded -> 0, padded -> 1
            
            gt_dict["points_per_voxel"] = gt_points[voxel_indices]
            gt_dict["points_per_voxel_padding"] = gt_points_padding[voxel_indices]
            gt_dict["gt_points"] = low_level_point_feature[
                :, : self.pred_dims
            ]  # For visualization
            gt_dict["gt_point_coors"] = point_coors  # For visualization

            assert len(gt_dict["points_per_voxel"]) == len(
                voxel_feats
            ), "Wrong number of point collections"
            
        return gt_dict
    
    @torch.no_grad()
    def get_focal_mask_index(self, coors, gt_bboxes, gt_labels_3d):
        # TODO: this version is only a tricky implmentation of judging pillar in bboxes. Also having some error.
        batch_size = len(gt_bboxes)
        device = coors.device
        voxel_size = torch.tensor(self.voxel_size[:2], device=device)
        start_coors = torch.tensor(self.point_cloud_range[:2], device=device)

        ids_mask_list = []
        ids_keep_list = []
        previous_length = 0
        for i in range(batch_size):
            inds = torch.where(coors[:, 0] == i)
            # print('inds',inds.shape,inds.dtype)
            coors_per_batch = coors[inds][:, [3, 2]] * voxel_size + start_coors
            z_coors = torch.ones((coors_per_batch.shape[0], 1), device=device)
            coors_per_batch = torch.cat([coors_per_batch, z_coors], dim=1)

            valid_index = gt_labels_3d[i] != -1
            valid_gt_bboxes = gt_bboxes[i][valid_index]
            valid_gt_bboxes.tensor[:, 2] = 1
            valid_gt_bboxes.tensor[:, 5] = 2
            voxel_in_gt_bboxes = valid_gt_bboxes.points_in_boxes(coors_per_batch)

            fg_index = voxel_in_gt_bboxes != -1
            fg_index = torch.nonzero(fg_index)
            bg_index = voxel_in_gt_bboxes == -1
            bg_index = torch.nonzero(bg_index)

            L = fg_index.shape[0]
            len_keep = int(L * (1 - self.random_mask_ratio))
            ids_shuffle = torch.randperm(L, device=device)
            ids_mask_list.append(fg_index[ids_shuffle[len_keep:]] + previous_length)
            ids_keep_list.append(fg_index[ids_shuffle[:len_keep]] + previous_length)
            ids_keep_list.append(bg_index + previous_length)
            previous_length += coors_per_batch.shape[0]

        ids_keep_list = torch.cat(ids_keep_list).squeeze()
        ids_mask_list = torch.cat(ids_mask_list).squeeze()

        return ids_keep_list, ids_mask_list

    @torch.no_grad()
    def get_vanilla_mask_index(self, coors, batch_size):
        # TODO: this version is only a tricky implmentation of judging pillar in bboxes. Also having some error.
        '''
        device=coors.device

        ids_keep_list=[]
        ids_mask_list=[]
        for i in range(batch_size):
            inds = torch.where(coors[:, 0] == i)
            L=inds[0].shape[0]
            len_keep = int(L * (1 - self.random_mask_ratio))
            ids_shuffle=torch.randperm(L,device=device)
            ids_keep_list.append(inds[0][ids_shuffle[:len_keep]])
            ids_mask_list.append(inds[0][ids_shuffle[len_keep:]])

        ids_keep_list = torch.cat(ids_keep_list)
        ids_mask_list = torch.cat(ids_mask_list)
        return ids_keep_list, ids_mask_list
    
        '''
        device = coors.device
        voxel_coords_distance = (((coors[:, 2] - self.grid_size[1] / 2) * self.voxel_size[1])**2 + 
                                ((coors[:, 3] - self.grid_size[2] / 2) * self.voxel_size[0])**2)**0.5

        select_30 = voxel_coords_distance <= 30
        select_30to50 = (voxel_coords_distance > 30) & (voxel_coords_distance <= 50)
        select_50 = voxel_coords_distance > 50

        coors[:, 1:] = coors[:, 1:].to(device)  # 인덱스가 아닌 값들을 GPU로 이동

        # GPU에 채우기 값으로 텐서 초기화
        selected_values_30 = torch.full_like(coors, fill_value=-1, device=device)
        selected_values_30to50 = torch.full_like(coors, fill_value=-1, device=device)
        selected_values_50 = torch.full_like(coors, fill_value=-1, device=device)

        # 효율적인 할당을 위해 PyTorch 인덱싱 사용
        selected_values_30[select_30] = coors[select_30]
        selected_values_30to50[select_30to50] = coors[select_30to50]
        selected_values_50[select_50] = coors[select_50]

        coors_list = [selected_values_30, selected_values_30to50, selected_values_50]
        ids_keep_list = []
        ids_mask_list = []

        for i in range(batch_size):
            for idx, distance in enumerate(coors_list):
                inds = torch.where(distance[:, 0] == i)[0]
                L = inds.shape[0]
                len_keep = int(L * (1 - self.random_mask_ratio - 0.2 + 0.2 * idx))
                
                ids_shuffle = torch.randperm(L, device=device)
                ids_keep_list.append(inds[ids_shuffle[:len_keep]])
                ids_mask_list.append(inds[ids_shuffle[len_keep:]])
        
        ids_keep_list = torch.cat(ids_keep_list)
        ids_mask_list = torch.cat(ids_mask_list)
        
        return ids_keep_list, ids_mask_list

        
    @torch.no_grad()
    @force_fp32()
    def sub_voxelize_low(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            torch.Tensor: Concatenated points and coordinates.
        """
        coors = []
        # dynamic voxelization only provide a coors mapping
        for i, res in enumerate(points):
            res_coors = self.sub_voxel_layer_low(res)
            coors.append(res_coors)
        points = torch.cat(points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode="constant", value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return points, coors_batch

    @torch.no_grad()
    @force_fp32()
    def sub_voxelize_med(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            torch.Tensor: Concatenated points and coordinates.
        """
        coors = []
        # dynamic voxelization only provide a coors mapping
        for i, res in enumerate(points):
            res_coors = self.sub_voxel_layer_med(res)
            coors.append(res_coors)
        points = torch.cat(points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode="constant", value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return points, coors_batch


    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points and coordinates.
        """
        coors = []
        # dynamic voxelization only provide a coors mapping
        for res in points:
            res_coors = self.voxel_layer(res)
            coors.append(res_coors)
        points = torch.cat(points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode="constant", value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return points, coors_batch

    @torch.no_grad()
    def map_voxel_centroid_id(
        self, voxel_coor, centriod_coor, voxel_size, point_cloud_range, batch_size
    ):
        x_max = (point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]
        y_max = (point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]
        z_max = (point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2]
        all_len = x_max * y_max * z_max
        y_len = y_max * x_max
        voxel_id = (
            voxel_coor[:, 0] * all_len
            + voxel_coor[:, 1] * y_len
            + voxel_coor[:, 2] * x_max
            + voxel_coor[:, 3]
        )
        centroid_id = (
            centriod_coor[:, 0] * all_len
            + centriod_coor[:, 1] * y_len
            + centriod_coor[:, 2] * x_max
            + centriod_coor[:, 3]
        )
        voxel_id = torch.sort(voxel_id)[1]
        centroid_id = torch.sort(centroid_id)[1]
        centroid_to_voxel = voxel_id.new_zeros(voxel_id.shape)
        voxel_to_centroid = voxel_id.new_zeros(voxel_id.shape)
        centroid_to_voxel[voxel_id] = centroid_id
        voxel_to_centroid[centroid_id] = voxel_id
        return centroid_to_voxel, voxel_to_centroid

    @torch.no_grad()
    def map_voxel_centroids_to_sub_voxel(
        self, voxel_coors, voxel_centroids, voxel_coors_low, voxel_coors_med, batch_size
    ):
        # TODO: this version doesn't support the ori voxel's height isn't the whole pillar
        sub_voxel_num_low = voxel_coors_low.shape[0]
        sub_voxel_num_med = voxel_coors_med.shape[0]
        voxel_num = voxel_coors.shape[0]
        grid_shape = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        # per_sub_voxel_num_low = self.sub_voxel_ratio_low[0] * self.sub_voxel_ratio_low[1] * self.sub_voxel_ratio_low[2]
        # per_sub_voxel_num_med = self.sub_voxel_ratio_med[0] * self.sub_voxel_ratio_med[1] * self.sub_voxel_ratio_med[2]
        #
        # centroid_target_low = voxel_centroids.new_zeros((sub_voxel_num_low , 3))
        # centroid_target_med = voxel_centroids.new_zeros((sub_voxel_num_med , 3))

        hash_table = voxel_coors.new_zeros(batch_size * grid_shape, dtype=torch.int64)
        voxel_id = torch.arange(voxel_coors.shape[0], device=voxel_coors.device)
        tensor_id = (
            voxel_coors[:, 0] * grid_shape
            + voxel_coors[:, 2] * self.grid_size[1]
            + voxel_coors[:, 3]
        )
        hash_table[tensor_id] = voxel_id

        tensor_id_low = (
            voxel_coors_low[:, 0] * grid_shape
            + voxel_coors_low[:, 2] // self.sub_voxel_ratio_low[1] * self.grid_size[1]
            + voxel_coors_low[:, 3] // self.sub_voxel_ratio_low[2]
        )
        tensor_id_low = hash_table[tensor_id_low]
        centroid_target_low = voxel_centroids[tensor_id_low]

        tensor_id_med = (
            voxel_coors_med[:, 0] * grid_shape
            + voxel_coors_med[:, 2] // self.sub_voxel_ratio_med[1] * self.grid_size[1]
            + voxel_coors_med[:, 3] // self.sub_voxel_ratio_med[2]
        )
        tensor_id_med = hash_table[tensor_id_med]
        centroid_target_med = voxel_centroids[tensor_id_med]

        return centroid_target_low, centroid_target_med

    @torch.no_grad()
    def map_voxel_to_sub_voxel(
        self,
        voxel_coors,
        voxel_centroids,
        voxel_coors_low,
        voxel_coors_med,
        voxel_centroids_low,
        voxel_centroids_med,
        batch_size,
    ):
        # TODO: this version doesn't support the ori voxel's height isn't the whole pillar
        sub_voxel_num_low = voxel_coors_low.shape[0]
        sub_voxel_num_med = voxel_coors_med.shape[0]
        voxel_num = voxel_coors.shape[0]
        grid_shape = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        per_sub_voxel_num_low = (
            self.sub_voxel_ratio_low[0]
            * self.sub_voxel_ratio_low[1]
            * self.sub_voxel_ratio_low[2]
        )
        per_sub_voxel_num_med = (
            self.sub_voxel_ratio_med[0]
            * self.sub_voxel_ratio_med[1]
            * self.sub_voxel_ratio_med[2]
        )

        centroid_target_low = voxel_centroids.new_zeros(
            (voxel_num * per_sub_voxel_num_low, 3)
        )
        centroid_target_med = voxel_centroids.new_zeros(
            (voxel_num * per_sub_voxel_num_med, 3)
        )

        centroid_to_low = voxel_centroids.new_zeros(
            (voxel_num * per_sub_voxel_num_low, 3)
        )
        centroid_to_med = voxel_centroids.new_zeros(
            (voxel_num * per_sub_voxel_num_med, 3)
        )

        sub_voxel_low_grid_xy = (
            self.sub_voxel_ratio_low[1] * self.sub_voxel_ratio_low[2]
        )
        sub_voxel_med_grid_xy = (
            self.sub_voxel_ratio_med[1] * self.sub_voxel_ratio_med[2]
        )

        hash_table = voxel_coors.new_zeros(batch_size * grid_shape, dtype=torch.int64)
        voxel_id = torch.arange(voxel_coors.shape[0], device=voxel_coors.device)
        tensor_id = (
            voxel_coors[:, 0] * grid_shape
            + voxel_coors[:, 2] * self.grid_size[1]
            + voxel_coors[:, 3]
        )
        hash_table[tensor_id] = voxel_id

        tensor_id_low = (
            voxel_coors_low[:, 0] * grid_shape
            + voxel_coors_low[:, 2] // self.sub_voxel_ratio_low[1] * self.grid_size[1]
            + voxel_coors_low[:, 3] // self.sub_voxel_ratio_low[2]
        )
        tensor_id_low = hash_table[tensor_id_low]
        centroid_for_low = voxel_centroids[tensor_id_low]

        target_id_low = (
            tensor_id_low * per_sub_voxel_num_low
            + (voxel_coors_low[:, 1] % self.sub_voxel_ratio_low[0])
            * sub_voxel_low_grid_xy
            + (voxel_coors_low[:, 2] % self.sub_voxel_ratio_low[1])
            * self.sub_voxel_ratio_low[2]
            + voxel_coors_low[:, 3] % self.sub_voxel_ratio_low[2]
        )
        centroid_target_low[target_id_low] = voxel_centroids_low
        centroid_target_low = centroid_target_low.view(
            voxel_num, per_sub_voxel_num_low, 3
        )

        centroid_to_low[target_id_low] = centroid_for_low
        centroid_to_low = centroid_to_low.view(voxel_num, per_sub_voxel_num_low, 3)

        tensor_id_med = (
            voxel_coors_med[:, 0] * grid_shape
            + voxel_coors_med[:, 2] // self.sub_voxel_ratio_med[1] * self.grid_size[1]
            + voxel_coors_med[:, 3] // self.sub_voxel_ratio_med[2]
        )
        tensor_id_med = hash_table[tensor_id_med]
        centroid_for_med = voxel_centroids[tensor_id_med]

        tensor_id_med = (
            tensor_id_med * per_sub_voxel_num_med
            + (voxel_coors_med[:, 1] % self.sub_voxel_ratio_med[0])
            * sub_voxel_med_grid_xy
            + voxel_coors_med[:, 2]
            % self.sub_voxel_ratio_med[1]
            * self.sub_voxel_ratio_med[2]
            + voxel_coors_med[:, 3] % self.sub_voxel_ratio_med[2]
        )
        centroid_target_med[tensor_id_med] = voxel_centroids_med
        centroid_target_med = centroid_target_med.view(
            voxel_num, per_sub_voxel_num_med, 3
        )

        centroid_to_med[tensor_id_med] = centroid_for_med
        centroid_to_med = centroid_to_med.view(voxel_num, per_sub_voxel_num_med, 3)

        return (
            centroid_to_low,
            centroid_to_med,
            centroid_target_low,
            centroid_target_med,
        )

    @torch.no_grad()
    def map_voxel_and_center_to_sub_voxel(
        self,
        voxel_coors,
        voxel_centroids,
        voxel_coors_low,
        voxel_coors_med,
        voxel_centroids_low,
        voxel_centroids_med,
        batch_size,
    ):
        # TODO: this version doesn't support the ori voxel's height isn't the whole pillar

        sub_voxel_low_grid_xy = (
            self.sub_voxel_ratio_low[1] * self.sub_voxel_ratio_low[2]
        )
        sub_voxel_med_grid_xy = (
            self.sub_voxel_ratio_med[1] * self.sub_voxel_ratio_med[2]
        )

        voxel_num = voxel_coors.shape[0]
        grid_shape = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        per_sub_voxel_num_low = (
            self.sub_voxel_ratio_low[0]
            * self.sub_voxel_ratio_low[1]
            * self.sub_voxel_ratio_low[2]
        )
        per_sub_voxel_num_med = (
            self.sub_voxel_ratio_med[0]
            * self.sub_voxel_ratio_med[1]
            * self.sub_voxel_ratio_med[2]
        )

        centroid_target_low = voxel_centroids.new_zeros(
            (voxel_num * per_sub_voxel_num_low, 3)
        )
        centroid_target_med = voxel_centroids.new_zeros(
            (voxel_num * per_sub_voxel_num_med, 3)
        )

        centroid_to_low = voxel_centroids.new_zeros(
            (voxel_num * per_sub_voxel_num_low, 3)
        )
        centroid_to_med = voxel_centroids.new_zeros(
            (voxel_num * per_sub_voxel_num_med, 3)
        )

        hash_table = voxel_coors.new_zeros(batch_size * grid_shape, dtype=torch.int64)
        voxel_id = torch.arange(voxel_coors.shape[0], device=voxel_coors.device)
        tensor_id = (
            voxel_coors[:, 0] * grid_shape
            + voxel_coors[:, 2] * self.grid_size[1]
            + voxel_coors[:, 3]
        )
        hash_table[tensor_id] = voxel_id

        tensor_id_low = (
            voxel_coors_low[:, 0] * grid_shape
            + voxel_coors_low[:, 2] // self.sub_voxel_ratio_low[1] * self.grid_size[1]
            + voxel_coors_low[:, 3] // self.sub_voxel_ratio_low[2]
        )
        tensor_id_low = hash_table[tensor_id_low]
        centroid_for_low = voxel_centroids[tensor_id_low]

        target_id_low = (
            tensor_id_low * per_sub_voxel_num_low
            + (voxel_coors_low[:, 1] % self.sub_voxel_ratio_low[0])
            * sub_voxel_low_grid_xy
            + (voxel_coors_low[:, 2] % self.sub_voxel_ratio_low[1])
            * self.sub_voxel_ratio_low[2]
            + voxel_coors_low[:, 3] % self.sub_voxel_ratio_low[2]
        )
        centroid_target_low[target_id_low] = voxel_centroids_low
        centroid_target_low = centroid_target_low.view(
            voxel_num, per_sub_voxel_num_low, 3
        )

        centroid_to_low[target_id_low] = centroid_for_low
        centroid_to_low = centroid_to_low.view(voxel_num, per_sub_voxel_num_low, 3)

        tensor_id_med = (
            voxel_coors_med[:, 0] * grid_shape
            + voxel_coors_med[:, 2] // self.sub_voxel_ratio_med[1] * self.grid_size[1]
            + voxel_coors_med[:, 3] // self.sub_voxel_ratio_med[2]
        )
        tensor_id_med = hash_table[tensor_id_med]
        centroid_for_med = voxel_centroids[tensor_id_med]

        tensor_id_med = (
            tensor_id_med * per_sub_voxel_num_med
            + (voxel_coors_med[:, 1] % self.sub_voxel_ratio_med[0])
            * sub_voxel_med_grid_xy
            + voxel_coors_med[:, 2]
            % self.sub_voxel_ratio_med[1]
            * self.sub_voxel_ratio_med[2]
            + voxel_coors_med[:, 3] % self.sub_voxel_ratio_med[2]
        )
        centroid_target_med[tensor_id_med] = voxel_centroids_med
        centroid_target_med = centroid_target_med.view(
            voxel_num, per_sub_voxel_num_med, 3
        )

        centroid_to_med[tensor_id_med] = centroid_for_med
        centroid_to_med = centroid_to_med.view(voxel_num, per_sub_voxel_num_med, 3)

        return (
            centroid_to_low,
            centroid_to_med,
            centroid_target_low,
            centroid_target_med,
        )

    @torch.no_grad()
    @force_fp32()
    def cal_voxel_curv(
        self,
        voxel_points,
        voxel_flag,
        centroids,
        centroid_to_voxel_id,
        voxel_to_centroid_id,
    ):
        N, max_points, _ = voxel_points.shape
        voxel_cetroids = (
            centroids[centroid_to_voxel_id].unsqueeze(dim=1).repeat(1, max_points, 1)
        )
        voxel_cetroids[~voxel_flag] = 0
        voxel_points = voxel_points - voxel_cetroids

        cov = voxel_points.transpose(-2, -1) @ voxel_points
        est_normal = torch.svd(cov)[2][..., -1]
        est_normal = est_normal / torch.norm(est_normal, p=2, dim=-1, keepdim=True)
        return est_normal[voxel_to_centroid_id]

    @torch.no_grad()
    @force_fp32()
    def cal_regular_voxel_curv(
        self,
        centroid_for_low,
        centroid_for_med,
        centroid_target_low,
        centroid_target_med,
    ):
        # N,max_points,_=voxel_points.shape
        # voxel_cetroids=centroids[centroid_to_voxel_id].unsqueeze(dim=1).repeat(1,max_points,1)
        # voxel_cetroids[~voxel_flag]=0
        voxel_points = centroid_target_low - centroid_for_low
        cov = voxel_points.transpose(-2, -1) @ voxel_points
        est_normal_low = torch.svd(cov)[2][..., -1]
        est_normal_low = est_normal_low / torch.norm(
            est_normal_low, p=2, dim=-1, keepdim=True
        )

        voxel_points = centroid_target_med - centroid_for_med
        cov = voxel_points.transpose(-2, -1) @ voxel_points
        est_normal_med = torch.svd(cov)[2][..., -1]
        est_normal_med = est_normal_med / torch.norm(
            est_normal_med, p=2, dim=-1, keepdim=True
        )

        return est_normal_low, est_normal_med

    @torch.no_grad()
    @force_fp32()
    def cal_regular_voxel_nor_and_curv(
        self, centroid_low, centroid_low_mask, voxel_centroid, indice_pairs
    ):
         # N,max_points,_=voxel_points.shape
        # voxel_cetroids=centroids[centroid_to_voxel_id].unsqueeze(dim=1).repeat(1,max_points,1)
        # voxel_cetroids[~voxel_flag]=0
        sub_voxel_num=centroid_low.shape[1]
        around_num,voxel_num=indice_pairs.shape
        around_mask=(indice_pairs==-1)
        centroid_low_around = centroid_low[indice_pairs]
        centroid_low_mask_around = centroid_low_mask[indice_pairs]

        centroid_low_around[around_mask]=0
        centroid_low_mask_around[around_mask]=False

        centroid_low_around=centroid_low_around.transpose(0,1).contiguous().view(voxel_num,-1,3)
        centroid_low_mask_around=centroid_low_mask_around.transpose(0,1).contiguous().view(voxel_num,-1)

        voxel_centroid_around = voxel_centroid.unsqueeze(dim=1).repeat(1, sub_voxel_num * around_num, 1)

        voxel_centroid_around[~centroid_low_mask_around]=0
        voxel_points=centroid_low_around - voxel_centroid_around
        cov = voxel_points.transpose(-2, -1) @ voxel_points
        svd = torch.svd(cov)
        est_normal = svd[2][..., -1]

        if self.norm_curv:
            est_normal = est_normal/torch.norm(est_normal,p=2, dim=-1, keepdim=True)

        est_curv = svd[1].to(dtype=torch.float64)
        est_curv = est_curv+eps

        est_curv = est_curv / est_curv.sum(dim=-1,keepdim=True)


        return est_normal,est_curv

    @torch.no_grad()
    def normalize_centroid(self, coors, centroids):
        device = coors.device
        voxel_size = torch.tensor(self.voxel_size[::-1], device=device)
        start_coors = torch.tensor(self.point_cloud_range[:3][::-1], device=device)
        # print('test shape',coors.shape,voxel_size.shape,start_coors.shape)
        coors_ = coors * voxel_size + start_coors
        centroids = (centroids - coors_.unsqueeze(dim=1)) / voxel_size
        return centroids

    @torch.no_grad()
    def normalize_centroid_sub_voxel(self, coors, centroids, layer=None):
        device = coors.device
        if layer == "low":
            voxel_size = torch.tensor(self.sub_voxel_size_low[::-1], device=device)
        elif layer == "med":
            voxel_size = torch.tensor(self.sub_voxel_size_med[::-1], device=device)
        else:
            voxel_size = torch.tensor(self.voxel_size[::-1], device=device)

        start_coors = torch.tensor(self.point_cloud_range[:3][::-1], device=device)
        # print('test shape',coors.shape,voxel_size.shape,start_coors.shape)
        coors = coors * voxel_size + start_coors
        centroids = (centroids - coors) / voxel_size
        return centroids

    @torch.no_grad()
    def get_multi_voxel_id_to_tensor_id_for_curv(
        self,
        voxel_coors,
        voxel_coors_med,
        voxel_centroids_med,
        batch_size,
    ):
        # TODO: this version doesn't support the ori voxel's height isn't the whole pillar
        voxel_num = voxel_coors.shape[0]
        grid_shape = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        per_sub_voxel_num_med = (
            self.sub_voxel_ratio_med[0]
            * self.sub_voxel_ratio_med[1]
            * self.sub_voxel_ratio_med[2]
        )
        centroid_target_med = voxel_coors.new_zeros(
            (voxel_num * per_sub_voxel_num_med, 3), dtype=torch.float32
        )
        centroid_target_mask_med = voxel_coors.new_zeros(
            voxel_num * per_sub_voxel_num_med, dtype=torch.bool
        )

        sub_voxel_med_grid_xy = (
            self.sub_voxel_ratio_med[1] * self.sub_voxel_ratio_med[2]
        )
        hash_table = voxel_coors.new_zeros(batch_size * grid_shape, dtype=torch.int64)

        voxel_id = torch.arange(voxel_coors.shape[0], device=voxel_coors.device)
        tensor_id = (
            voxel_coors[:, 0] * grid_shape
            + voxel_coors[:, 2] * self.grid_size[1]
            + voxel_coors[:, 3]
        )
        hash_table[tensor_id] = voxel_id
        tensor_id_med = (
            voxel_coors_med[:, 0] * grid_shape
            + voxel_coors_med[:, 2] // self.sub_voxel_ratio_med[1] * self.grid_size[1]
            + voxel_coors_med[:, 3] // self.sub_voxel_ratio_med[2]
        )
        tensor_id_med = hash_table[tensor_id_med]
        tensor_id_med = (
            tensor_id_med * per_sub_voxel_num_med
            + (voxel_coors_med[:, 1] % self.sub_voxel_ratio_med[0]) * sub_voxel_med_grid_xy
            + voxel_coors_med[:, 2] % self.sub_voxel_ratio_med[1] * self.sub_voxel_ratio_med[2]
            + voxel_coors_med[:, 3] % self.sub_voxel_ratio_med[2]
        )
        centroid_target_med[tensor_id_med] = voxel_centroids_med
        centroid_target_mask_med[tensor_id_med] = True
        centroid_target_med = centroid_target_med.view(
            voxel_num, per_sub_voxel_num_med, 3
        )
        centroid_target_mask_med = centroid_target_mask_med.view(
            voxel_num, per_sub_voxel_num_med
        )

        return centroid_target_med, centroid_target_mask_med

    @torch.no_grad()
    def get_multi_voxel_id_to_tensor_id_ori(
        self,
        voxel_coors,
        voxel_coors_low,
        voxel_coors_med,
        voxel_centroids_low,
        voxel_centroids_med,
        ids_masked,
        batch_size,
    ):
        # TODO: this version doesn't support the ori voxel's height isn't the whole pillar
        voxel_num = voxel_coors.shape[0]
        grid_shape = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        per_sub_voxel_num_low = (
            self.sub_voxel_ratio_low[0]
            * self.sub_voxel_ratio_low[1]
            * self.sub_voxel_ratio_low[2]
        )
        per_sub_voxel_num_med = (
            self.sub_voxel_ratio_med[0]
            * self.sub_voxel_ratio_med[1]
            * self.sub_voxel_ratio_med[2]
        )

        centroid_target_low = voxel_coors.new_zeros(
            (voxel_num * per_sub_voxel_num_low, 3), dtype=torch.float32
        )
        centroid_target_mask_low = voxel_coors.new_zeros(
            voxel_num * per_sub_voxel_num_low, dtype=torch.bool
        )

        centroid_target_med = voxel_coors.new_zeros(
            (voxel_num * per_sub_voxel_num_med, 3), dtype=torch.float32
        )
        centroid_target_mask_med = voxel_coors.new_zeros(
            voxel_num * per_sub_voxel_num_med, dtype=torch.bool
        )

        sub_voxel_low_grid_xy = (
            self.sub_voxel_ratio_low[1] * self.sub_voxel_ratio_low[2]
        )
        sub_voxel_med_grid_xy = (
            self.sub_voxel_ratio_med[1] * self.sub_voxel_ratio_med[2]
        )

        hash_table = voxel_coors.new_zeros(batch_size * grid_shape, dtype=torch.int64)
        voxel_id = torch.arange(voxel_coors.shape[0], device=voxel_coors.device)
        tensor_id = (
            voxel_coors[:, 0] * grid_shape
            + voxel_coors[:, 2] * self.grid_size[1]
            + voxel_coors[:, 3]
        )
        hash_table[tensor_id] = voxel_id

        tensor_id_low = (
            voxel_coors_low[:, 0] * grid_shape
            + voxel_coors_low[:, 2] // self.sub_voxel_ratio_low[1] * self.grid_size[1]
            + voxel_coors_low[:, 3] // self.sub_voxel_ratio_low[2]
        )
        tensor_id_low = hash_table[tensor_id_low]
        target_id_low = (
            tensor_id_low * per_sub_voxel_num_low
            + (voxel_coors_low[:, 1] % self.sub_voxel_ratio_low[0])
            * sub_voxel_low_grid_xy
            + (voxel_coors_low[:, 2] % self.sub_voxel_ratio_low[1])
            * self.sub_voxel_ratio_low[2]
            + voxel_coors_low[:, 3] % self.sub_voxel_ratio_low[2]
        )
        centroid_target_low[target_id_low] = voxel_centroids_low
        centroid_target_mask_low[target_id_low] = True
        centroid_target_low = centroid_target_low.view(
            voxel_num, per_sub_voxel_num_low, 3
        )[ids_masked]
        centroid_target_mask_low = centroid_target_mask_low.view(
            voxel_num, per_sub_voxel_num_low
        )[ids_masked]

        tensor_id_med = (
            voxel_coors_med[:, 0] * grid_shape
            + voxel_coors_med[:, 2] // self.sub_voxel_ratio_med[1] * self.grid_size[1]
            + voxel_coors_med[:, 3] // self.sub_voxel_ratio_med[2]
        )
        tensor_id_med = hash_table[tensor_id_med]
        tensor_id_med = (
            tensor_id_med * per_sub_voxel_num_med
            + (voxel_coors_med[:, 1] % self.sub_voxel_ratio_med[0])
            * sub_voxel_med_grid_xy
            + voxel_coors_med[:, 2]
            % self.sub_voxel_ratio_med[1]
            * self.sub_voxel_ratio_med[2]
            + voxel_coors_med[:, 3] % self.sub_voxel_ratio_med[2]
        )
        centroid_target_med[tensor_id_med] = voxel_centroids_med
        centroid_target_mask_med[tensor_id_med] = True
        centroid_target_med = centroid_target_med.view(
            voxel_num, per_sub_voxel_num_med, 3
        )[ids_masked]
        centroid_target_mask_med = centroid_target_mask_med.view(
            voxel_num, per_sub_voxel_num_med
        )[ids_masked]

        return (
            centroid_target_low,
            centroid_target_mask_low,
            centroid_target_med,
            centroid_target_mask_med,
        )

    @torch.no_grad()
    @force_fp32()
    def get_centroid_per_voxel(
        self, points: torch.Tensor, voxel_idxs: torch.Tensor, num_points_in_voxel=None
    ):
        """
        Args:
            points: (N, 3 + (f)) [bxyz + (f)]
            voxel_idxs: (N, 4) [bxyz]
            num_points_in_voxel: (N)
        Returns:
            centroids: (N', 4 + (f)) [bxyz + (f)] Centroids for each unique voxel
            centroid_voxel_idxs: (N', 4) [bxyz] Voxels idxs for centroids
            labels_count: (N') Number of points in each voxel
        """
        assert points.shape[0] == voxel_idxs.shape[0]
        voxel_idxs_valid_mask = (voxel_idxs >= 0).all(-1)

        # print('non zero test',voxel_idxs.shape,voxel_idxs_valid_mask.sum())

        voxel_idxs = voxel_idxs[voxel_idxs_valid_mask]

        points = points[voxel_idxs_valid_mask]

        centroid_voxel_idxs, unique_idxs, labels_count = voxel_idxs.unique(
            dim=0, sorted=False, return_inverse=True, return_counts=True
        )

        unique_idxs = unique_idxs.view(unique_idxs.size(0), 1).expand(
            -1, points.size(-1)
        )

        # Scatter add points based on unique voxel idxs
        if num_points_in_voxel is not None:
            centroids = torch.zeros(
                (centroid_voxel_idxs.shape[0], points.shape[-1]),
                device=points.device,
                dtype=torch.float,
            ).scatter_add_(0, unique_idxs, points * num_points_in_voxel.unsqueeze(-1))
            num_points_in_centroids = torch.zeros(
                (centroid_voxel_idxs.shape[0]), device=points.device, dtype=torch.int64
            ).scatter_add_(0, unique_idxs[:, 0], num_points_in_voxel)
            centroids = centroids / num_points_in_centroids.float().unsqueeze(-1)
        else:
            centroids = torch.zeros(
                (centroid_voxel_idxs.shape[0], points.shape[-1]),
                device=points.device,
                dtype=torch.float,
            ).scatter_add_(0, unique_idxs, points)
            centroids = centroids / labels_count.float().unsqueeze(-1)

        return centroids, centroid_voxel_idxs, labels_count

    @torch.no_grad()
    @force_fp32()
    def get_centroid_and_normal_per_voxel(
        self, points: torch.Tensor, voxel_idxs: torch.Tensor, num_points_in_voxel=None
    ):
        """
        Args:
            points: (N, 3 + (f)) [bxyz + (f)]
            voxel_idxs: (N, 4) [bxyz]
            num_points_in_voxel: (N)
        Returns:
            centroids: (N', 4 + (f)) [bxyz + (f)] Centroids for each unique voxel
            centroid_voxel_idxs: (N', 4) [bxyz] Voxels idxs for centroids
            labels_count: (N') Number of points in each voxel
        """
        assert points.shape[0] == voxel_idxs.shape[0]
        voxel_idxs_valid_mask = (voxel_idxs >= 0).all(-1)

        # print('non zero test',voxel_idxs.shape,voxel_idxs_valid_mask.sum())

        voxel_idxs = voxel_idxs[voxel_idxs_valid_mask]

        points = points[voxel_idxs_valid_mask]

        centroid_voxel_idxs, unique_idxs, labels_count = voxel_idxs.unique(
            dim=0, sorted=True, return_inverse=True, return_counts=True
        )
        N, C = points.shape
        ori_id = torch.arange(N)
        ori_id_reverse = torch.flip(ori_id, dims=[0])
        inverse_unique_idxs = torch.flip(unique_idxs, dims=[0])

        unique_idxs_ = unique_idxs.view(unique_idxs.size(0), 1).expand(-1, C).clone()

        # Scatter add points based on unique voxel idxs
        if num_points_in_voxel is not None:
            centroids = torch.zeros(
                (centroid_voxel_idxs.shape[0], points.shape[-1]),
                device=points.device,
                dtype=torch.float,
            ).scatter_add_(0, unique_idxs_, points * num_points_in_voxel.unsqueeze(-1))
            num_points_in_centroids = torch.zeros(
                (centroid_voxel_idxs.shape[0]), device=points.device, dtype=torch.int64
            ).scatter_add_(0, unique_idxs_[:, 0], num_points_in_voxel)
            centroids = centroids / num_points_in_centroids.float().unsqueeze(-1)
        else:
            centroids = torch.zeros(
                (centroid_voxel_idxs.shape[0], points.shape[-1]),
                device=points.device,
                dtype=torch.float,
            ).scatter_add_(0, unique_idxs_, points)
            centroids = centroids / labels_count.float().unsqueeze(-1)
        points_centroids = centroids[unique_idxs]
        sort_idx = torch.sort(unique_idxs)[1]
        sort_idx_inverse = ori_id_reverse[torch.sort(inverse_unique_idxs)[1]]
        edge_vec1 = points[sort_idx] - points_centroids[sort_idx]
        edge_vec2 = points[sort_idx_inverse] - points_centroids[sort_idx_inverse]
        nor = torch.cross(edge_vec1, edge_vec2, dim=-1)
        nor_len = torch.norm(nor, dim=-1, keepdim=True)
        nor_len[nor_len == 0] = 1
        nor = nor / nor_len

        unique_idxs = unique_idxs[sort_idx].view(unique_idxs.size(0), 1).expand(-1, C)
        sur_nor = torch.zeros(
            (centroid_voxel_idxs.shape[0], nor.shape[-1]),
            device=points.device,
            dtype=torch.float,
        ).scatter_add_(0, unique_idxs, nor)
        # nor_len=torch.norm(sur_nor, dim=-1, keepdim=True)
        # nor_len[nor_len == 0] = 1
        # sur_nor = sur_nor / nor_len
        sur_nor = sur_nor / labels_count.float().unsqueeze(-1)
        return centroids, sur_nor, centroid_voxel_idxs, labels_count

    @force_fp32(apply_to=("reg_pred", "centroid_target", "cls_pred"))
    def loss(
        self,
        centroid_low,
        centroid_low_mask,
        centroid_med,
        centroid_med_mask,
        centroid_high,
        centroid_normal_low,
        centroid_normal_med,
        centroid_normal_high,
        reg_pred_low,
        reg_pred_med,
        reg_pred_high,
        nor_pred_low,
        nor_pred_med,
        nor_pred_high,
        pred_dict,
        #simsiam_feature,
        cls_pred_low=None,
        cls_pred_med=None,
    ):
        centroid_low_mask = centroid_low_mask.view(-1)
        centroid_low = centroid_low.view(-1, 3)[centroid_low_mask]

        reg_pred_low = reg_pred_low.view(-1, 3)[centroid_low_mask]

        centroid_med_mask = centroid_med_mask.view(-1)
        centroid_med = centroid_med.view(-1, 3)[centroid_med_mask]

        reg_pred_med = reg_pred_med.view(-1, 3)[centroid_med_mask]

        pred_points_masked = pred_dict["pred_points_masked"]
        gt_points_masked = pred_dict["gt_points_masked"]
        gt_point_padding_masked = pred_dict["gt_point_padding_masked"]
        loss_chamfer_src_masked, loss_chamfer_dst_masked = self.chamfer_distance_loss(
        pred_points_masked, gt_points_masked, trg_padding=gt_point_padding_masked)
        
        #p1, p2, z1, z2 = simsiam_feature
        #criterion = nn.CosineSimilarity(dim=1) #.cuda(args.gpu)
        #loss_similarity = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
        
        if self.mse_loss:
            loss_reg_low = (reg_pred_low - centroid_low) ** 2
            loss_reg_low = loss_reg_low.mean(dim=-1)
            loss_reg_low = (
                loss_reg_low.sum() / loss_reg_low.shape[0] * self.loss_ratio_low
            )

            loss_reg_med = (reg_pred_med - centroid_med) ** 2
            loss_reg_med = loss_reg_med.mean(dim=-1)
            loss_reg_med = (
                loss_reg_med.sum() / loss_reg_med.shape[0] * self.loss_ratio_med
            )

            loss_reg_top = (reg_pred_high - centroid_high) ** 2
            loss_reg_top = loss_reg_top.mean(dim=-1)
            loss_reg_top = (
                loss_reg_top.sum() / loss_reg_top.shape[0] * self.loss_ratio_top
            )

            if self.nor_usr_sml1 is None:
                if nor_pred_low is None and nor_pred_med is None:
                    loss_nor_low = (
                        nor_pred_high - centroid_normal_low.to(nor_pred_high.device)
                    ) ** 2
                    loss_nor_low = loss_nor_low.mean(dim=-1)
                    loss_nor_low = (
                        loss_nor_low.sum()
                        / loss_nor_low.shape[0]
                        * self.loss_ratio_low_nor
                    )
                else:
                    loss_nor_low = (nor_pred_low - centroid_normal_low) ** 2
                    loss_nor_low = loss_nor_low.mean(dim=-1)
                    loss_nor_low = (
                        loss_nor_low.sum()
                        / loss_nor_low.shape[0]
                        * self.loss_ratio_low_nor
                    )

            else:
                if nor_pred_low is None and nor_pred_med is None:
                    loss_nor_low = (
                        self.nor_loss(nor_pred_high, centroid_normal_low)
                        * self.loss_ratio_low_nor
                    )
                else:
                    loss_nor_low = (
                        self.nor_loss(nor_pred_low, centroid_normal_low)
                        * self.loss_ratio_low_nor
                    )

        else:
            loss_reg_low = (
                self.reg_loss(reg_pred_low, centroid_low) * self.loss_ratio_low
            )
            loss_reg_med = (
                self.reg_loss(reg_pred_med, centroid_med) * self.loss_ratio_med
            )
            loss_reg_top = (
                self.reg_loss(reg_pred_high, centroid_high) * self.loss_ratio_top
            )

        if self.cls_sub_voxel:
            assert cls_pred_low is not None
            cls_pred_low = cls_pred_low.view(-1, 2)
            cls_pred_med = cls_pred_med.view(-1, 2)

            loss_cls_low = (
                self.cls_loss(cls_pred_low, centroid_low_mask.long())
                * self.cls_loss_ratio_low
            )
            loss_cls_med = (
                self.cls_loss(cls_pred_med, centroid_med_mask.long())
                * self.cls_loss_ratio_med
            )
            loss = dict(
                #loss_similarity=loss_similarity,
                loss_curv_around=loss_nor_low,
                loss_centroid_low=loss_reg_low,
                loss_centroid_med=loss_reg_med,
                loss_centroid_top=loss_reg_top,
                loss_cls_low=loss_cls_low,
                loss_cls_med=loss_cls_med,
                loss_chamfer_src_masked=loss_chamfer_src_masked,
                loss_chamfer_dst_masked=loss_chamfer_dst_masked
            )

        else:
            loss = dict(
                loss_centroid_low=loss_reg_low,
                loss_centroid_med=loss_reg_med,
                loss_centroid_top=loss_reg_top,
                loss_nor_low=loss_nor_low,
            )
        tb_dict = None
        return loss, tb_dict

    @staticmethod
    def chamfer_distance_loss(src, trg, trg_padding, criterion_mode="l2"):
        """

        :param src: predicted point positions (B, N, C)
        :param trg: gt point positions (B, M, C)
        :param trg_padding: Which points are padded (B, M)
        :type trg_padding: torch.Tensor([torch.bool])
        :param criterion_mode: way of calculating distance, l1, l2, or smooth_l1
        :return:
        """
        if criterion_mode == 'smooth_l1':
            criterion = smooth_l1_loss
        elif criterion_mode == 'l1':
            criterion = l1_loss
        elif criterion_mode == 'l2':
            criterion = mse_loss
        else:
            raise NotImplementedError
        # src->(B,N,C) dst->(B,M,C)
        src_expand = src.unsqueeze(2).repeat(1, 1, trg.shape[1], 1)  # (B,N M,C)
        trg_expand = trg.unsqueeze(1).repeat(1, src.shape[1], 1, 1)  # (B,N M,C)
        trg_padding_expand = trg_padding.unsqueeze(1).repeat(1, src.shape[1], 1)  # (B,N M)

        distance = criterion(src_expand, trg_expand, reduction='none').sum(-1)  # (B,N M)
        distance[trg_padding_expand] = float('inf')#torch.inf

        src2trg_distance, indices1 = torch.min(distance, dim=2)  # (B,N)
        trg2src_distance, indices2 = torch.min(distance, dim=1)  # (B,M)
        trg2src_distance[trg_padding] = 0

        loss_src = torch.mean(src2trg_distance)
        # Since there is different number of points in each voxel we want to have each voxel matter equally much
        # and to not have voxels with more points be more important to mimic
        loss_trg = trg2src_distance.sum(1) / (~trg_padding).sum(1)  # B
        loss_trg = loss_trg.mean()

        return loss_src, loss_trg

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        feats = self.extract_feats(points, img_metas)
        return feats

    @torch.no_grad()
    def cal_center_loss(self, centroid_target, centroid_target_mask, x):
        centroid_target_mask = centroid_target_mask.view(-1)
        centroid_target = centroid_target.view(-1, 3)[centroid_target_mask]
        x = x.view(-1, 3)[centroid_target_mask]
        center = x.new_ones(x.shape) * 0.5
        centroid_loss = self.loss(x, centroid_target) * self.loss_ratio
        center_loss = self.loss(x, center) * self.loss_ratio
        center_centroid_loss = self.loss(center, centroid_target) * self.loss_ratio
        print(
            "centroid_loss:",
            centroid_loss,
            "center_loss:",
            center_loss,
            "center_centroid_loss:",
            center_centroid_loss,
        )

    def simple_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function without augmentaiton."""
        x = self.extract_feat(points, img_metas)
        outs = self.bbox_head(x)
        # print(type(img_metas),type(imgs),type(rescale),type(outs))
        if self.centerpoint_head:
            bbox_list = self.bbox_head.get_bboxes(
                outs, img_metas=img_metas, rescale=rescale
            )
        else:
            bbox_list = self.bbox_head.get_bboxes(*outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

        return bbox_results
    
    def get_voxel_indices(self, coors):
        vx, vy, vz = tuple(self.grid_size[::-1])
        indices = (
            coors[:, 0] * vz * vy * vx
            + coors[:, 1] * vy * vx  # batch
            + coors[:, 2] * vx  # z
            + coors[:, 3]  # y  # x
        ).long()
        return indices
