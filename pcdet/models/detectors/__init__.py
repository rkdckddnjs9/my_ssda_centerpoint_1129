from .detector3d_template import Detector3DTemplate
from .detector3d_template_v2 import Detector3DTemplateV2
from .PartA2_net import PartA2Net
from .point_rcnn import PointRCNN
from .pointpillar import PointPillar
from .pv_rcnn import PVRCNN
from .second_net import SECONDNet
from .second_net_iou import SECONDNetIoU
#from .caddn import CaDDN
from .voxel_rcnn import VoxelRCNN
from .pillarnet import PillarNet
from .centerpoint import CenterPoint
from .pv_rcnn_plusplus import PVRCNNPlusPlus
from .pv_rcnn_ssl import PVRCNN_SSL
from .second_net_iou_ssl import SECONDNetIoU_SSL
from .centerpoint_ssl import CenterPoint_SSL
from .pointpillar_ssl import PointPillar_SSL
from .centerpoint_rcnn import CenterPointRCNN
from .centerpoint_dp_ori import CenterPoint_PointPillar_SingelHead
from .centerpoint_rcnn_v2 import CenterPointRCNNV2 #dynamic + centerpoint(ori) + point head (pcdet) + pvrcnn head(pcdet)
from .centerpoint_rcnn_ssl import CenterPointRCNN_SSL
from .centerpoint_pp_rcnn import CenterPoint_PointPillar_RCNN
from .centerpoint_pp_rcnn_v2 import CenterPoint_PointPillar_RCNNV2
from . centerpoint_pp_rcnn_v2_ssl import CenterPoint_PointPillar_RCNNV2_SSL


from .pointpillar_rcnn import PointPillarRCNN

__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'Detector3DTemplateV2': Detector3DTemplateV2,
    'SECONDNet': SECONDNet,
    'PartA2Net': PartA2Net,
    'PVRCNN': PVRCNN,
    'PointPillar': PointPillar,
    'PointRCNN': PointRCNN,
    'SECONDNetIoU': SECONDNetIoU,
    #'CaDDN': CaDDN,
    'VoxelRCNN': VoxelRCNN,
    'PillarNet': PillarNet,
    'CenterPoint': CenterPoint,
    'CenterPoint_SSL': CenterPoint_SSL,
    'PointPillar_SSL': PointPillar_SSL,
    'PVRCNNPlusPlus': PVRCNNPlusPlus,
    'PVRCNN_SSL': PVRCNN_SSL,
    'SECONDNetIoU_SSL': SECONDNetIoU_SSL,
    'CenterPoint_PointPillar_SingelHead': CenterPoint_PointPillar_SingelHead,
    'CenterPointRCNN' : CenterPointRCNN,
    'CenterPointRCNNV2' : CenterPointRCNNV2,
    'CenterPointRCNN_SSL' : CenterPointRCNN_SSL,
    'PointPillarRCNN' : PointPillarRCNN,
    'CenterPoint_PointPillar_RCNN' : CenterPoint_PointPillar_RCNN,
    'CenterPoint_PointPillar_RCNNV2' : CenterPoint_PointPillar_RCNNV2,
    'CenterPoint_PointPillar_RCNNV2_SSL' : CenterPoint_PointPillar_RCNNV2_SSL
}


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
