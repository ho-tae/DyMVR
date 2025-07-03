import torch
import torch.nn as nn
from torch.nn import functional as F
from pcdet.models.backbones_3d.util.voxelize import Voxelization
from .voxel_encoder import DynamicScatterVFE
from pcdet.ops.sst.sst_ops import scatter_v2
from mmcv.runner import force_fp32

eps = 1e-9


class MultiFusionVoxel(nn.Module):
    r"""VoxelNet using `dynamic voxelization <https://arxiv.org/abs/1910.06528>`_."""

    def __init__(
        self,
        grid_size,
        voxel_size,
        point_cloud_range,
        sub_voxel_size_low,
        sub_voxel_size_med,
    ):
        super(MultiFusionVoxel, self).__init__()
        
        self.point_cloud_range = point_cloud_range
        
        self.voxel_size = voxel_size
        self.grid_size = grid_size
        self.sub_voxel_size_low = sub_voxel_size_low
        self.sub_voxel_size_med = sub_voxel_size_med
        
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
        
            
    def forward(self, batch_dict):
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
        voxel_features, feature_coors = self.voxel_encoder(voxels, coors)
    
        voxel_mid_features, feature_mid_coors = self.voxel_encoder_middle(sub_voxels_med, sub_coors_med)
        voxel_low_features, feature_low_coors = self.voxel_encoder_low(sub_voxels_low, sub_coors_low)
        
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
        
        return [voxel_features, merge_voxel_mid_low_features], feature_coors #[merge_voxel_mid_low_features], feature_coors # [merge_voxel_mid_low_features], feature_coors #[voxel_features, voxel_med_mean, voxel_low_mean], feature_coors
    
    def find_coordinate_indices(self, coords, coord_list, mode="mid"):
        if mode == "mid":
            #coord_list[:, 1] = 0
            coord_list[:, 1:] = torch.div(coord_list[:, 1:], 2).floor()
            indices = torch.where(torch.all(coords.unsqueeze(1) == coord_list, dim=2))[1].tolist()
        if mode == "low":
            #coord_list[:, 1] = 0
            coord_list[:, 1:] = torch.div(coord_list[:, 1:], 4).floor()
            indices = torch.where(torch.all(coords.unsqueeze(1) == coord_list, dim=2))[1].tolist()
        return indices
    
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