import torch

pre_file = "/CMKD/output/CMKD/tools/cfgs/kitti_models/voxel_mae_sst/default/lr_0.0001/ckpt/trian3712_voxelmae_best.pth"
state_dict = torch.load(pre_file, map_location="cpu")
print(state_dict["model_state"].keys)


'''
backbone_list = []
for i in state_dict["model_state"].keys():
    if i[:8] == "backbone":
        backbone_list.append(i)
        print(i)
print(len(backbone_list))
'''