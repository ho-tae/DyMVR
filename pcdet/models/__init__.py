from collections import namedtuple

import numpy as np
import torch
from thop import profile
from .detectors import build_detector

try:
    import kornia
except:
    pass 
    # print('Warning: kornia is not installed. This package is only required by CaDDN')



def build_network(model_cfg, num_class, dataset):
    model = build_detector(
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )
    return model


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        elif key in ['frame_id', 'metadata', 'calib', 'index', 'cam_type']:
            continue
        elif key in ['images']:
            batch_dict[key] = kornia.image_to_tensor(val).float().cuda().contiguous()
        elif key in ['image_shape']:
            batch_dict[key] = torch.from_numpy(val).int().cuda()
        else:
            batch_dict[key] = torch.from_numpy(val).float().cuda()


def model_fn_decorator():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)
        ret_dict, tb_dict, disp_dict = model(batch_dict)
        #flops, params = profile(model, (batch_dict, ))
        #gflops = flops / 1e9
        #print(f"Total FLOPs: {flops:.2f} FLOPs")
        #print(f"GFLOPS: {gflops:.2f} GFLOPS")
        loss = ret_dict['loss'].mean()
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func

def model_fn_decorator_cmkd():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)
        ret_dict, tb_dict, disp_dict = model(batch_dict)

        loss = ret_dict['loss'].mean()

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
 
            if hasattr(model.module.model_img, 'update_global_step'):
                model.module.model_img.update_global_step()
            else:
                model.module.model_img.module.update_global_step()
        
        else:
            if hasattr(model.model_img, 'update_global_step'):
                model.model_img.update_global_step()
            else:
                model.model_img.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func


