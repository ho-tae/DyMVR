from .detector3d_template import Detector3DTemplate
from .PartA2_net import PartA2Net
from .point_rcnn import PointRCNN
from .pointpillar import PointPillar
from .pv_rcnn import PVRCNN
from .second_net import SECONDNet
from .second_net_iou import SECONDNetIoU
from .caddn import CaDDN
from .voxel_rcnn import VoxelRCNN
from .centerpoint import CenterPoint
from .pv_rcnn_plusplus import PVRCNNPlusPlus
from .cmkd import CMKD, CMKD_MONO, CMKD_LIDAR
from .voxel_mae_sst import VoxelMAE_SST
from .sst import SST
from .geo_mae_sst import GeoMAE_SST
from .sst_conv import SST_Conv
from .vf_sst import VF_SST

__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'SECONDNet': SECONDNet,
    'PartA2Net': PartA2Net,
    'PVRCNN': PVRCNN,
    'PointPillar': PointPillar,
    'PointRCNN': PointRCNN,
    'SECONDNetIoU': SECONDNetIoU,
    'CaDDN': CaDDN,
    'VoxelRCNN': VoxelRCNN,
    'CenterPoint': CenterPoint,
    'PVRCNNPlusPlus': PVRCNNPlusPlus,
    'CMKD': CMKD,
    'CMKD_MONO': CMKD_MONO,
    'CMKD_LIDAR': CMKD_LIDAR,
    'VoxelMAE_SST' :VoxelMAE_SST,
    'SST' : SST,
    'GeoMAE_SST' : GeoMAE_SST,
    'SST_Conv' : SST_Conv,
    'VF_SST' : VF_SST
}


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
