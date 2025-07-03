from .detector3d_template_voxel_mae import Detector3DTemplate_voxel_mae


class GeoMAE_SST(Detector3DTemplate_voxel_mae):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            if cur_module == self.module_list[0]:  # MultiSubVoxelDynamicVoxelNetSSL
                (
                    [
                        backbone_input1,
                        backbone_input2,
                        backbone_input3,
                        backbone_input4,
                        backbone_input5,
                        backbone_input6,
                        backbone_input7,
                        backbone_input8,
                        backbone_input9,
                    ],
                    centroid_low,
                    centroid_low_mask,
                    centroid_med,
                    centroid_med_mask,
                    centroid_high,
                    centroid_normal_low,
                    centroid_normal_med,
                    centroid_normal_high,
                ) = cur_module(batch_dict)
                
            if cur_module == self.module_list[1]:  # MultiMAESSTSPChoose
                (
                    reg_pred_low,
                    reg_pred_med,
                    reg_pred_high,
                    nor_pred_low,
                    nor_pred_med,
                    nor_pred_high,
                    cls_pred_low,
                    cls_pred_med,
                    pred_dict,
                    )= cur_module( #, simsiam_feature = cur_module(
                    backbone_input1,
                    backbone_input2,
                    backbone_input3,
                    backbone_input4,
                    backbone_input5,
                    backbone_input6,
                    backbone_input7,
                    backbone_input8,
                    backbone_input9,
                )
                self.outs = (
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
                    cls_pred_low,
                    cls_pred_med,
                )

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {"loss": loss}
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.vfe.loss(*self.outs)
        tb_dict = {
            #"loss_similarity": loss_rpn["loss_similarity"].item(),
            "loss_chamfer_src_masked": loss_rpn["loss_chamfer_src_masked"].item(),
            "loss_chamfer_dst_masked": loss_rpn["loss_chamfer_dst_masked"].item(),
            "loss_curv_around": loss_rpn["loss_curv_around"].item(),
            "loss_centroid_low": loss_rpn["loss_centroid_low"].item(),
            "loss_centroid_med": loss_rpn["loss_centroid_med"].item(),
            "loss_centroid_top": loss_rpn["loss_centroid_top"].item(),
            "loss_cls_low": loss_rpn["loss_cls_low"].item(),
            "loss_cls_med": loss_rpn["loss_cls_med"].item(),
        }

        loss = sum(loss_rpn.values())
        return loss, tb_dict, disp_dict
