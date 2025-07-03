from .detector3d_template_voxel_mae import Detector3DTemplate_voxel_mae

class VoxelMAE_SST(Detector3DTemplate_voxel_mae):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            if cur_module == self.module_list[0]: # DynamicVFE
                voxel_features, feature_coors, low_level_point_feature, indices = cur_module(batch_dict)
            if cur_module == self.module_list[1]: # SSTInputLayerV2Masked
                x = cur_module(voxel_features, feature_coors, low_level_point_feature, indices, batch_dict["batch_size"])
            if cur_module == self.module_list[2]: # SSTv2
                x = cur_module(x)
            if cur_module == self.module_list[3]: # SSTv2decoder
                x = cur_module(x)
            if cur_module == self.module_list[4]: # reconstruction_head
                self.outs = cur_module(x)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.loss(self.outs[0])
        tb_dict = {
            'loss_num_points_masked': loss_rpn['loss_num_points_masked'].item(), #,'loss_rpn': loss_rpn.item(),
            'loss_chamfer_src_masked': loss_rpn['loss_chamfer_src_masked'].item(),
            'loss_chamfer_dst_masked': loss_rpn['loss_chamfer_dst_masked'].item(),
            **tb_dict
        }

        loss = sum(loss_rpn.values())
        return loss, tb_dict, disp_dict
