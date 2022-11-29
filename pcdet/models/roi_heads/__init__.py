from .partA2_head import PartA2FCHead
from .pointrcnn_head import PointRCNNHead
from .pvrcnn_head import PVRCNNHead
from .second_head import SECONDHead
from .voxelrcnn_head import VoxelRCNNHead
from .roi_head_template import RoIHeadTemplate
from .roi_head_template_centerpoint import RoIHeadTemplate_CenterPoint
from .roi_head_template_centerpoint_pointpillar import RoIHeadTemplate_CenterPoint_PointPillar
from .roi_head import RoIHead
from .roi_head_pillar import RoIHeadPillar
from .roi_head_pillar_pn import RoIHeadPillarNet
from .roi_head_pillar_dn import RoIHeadDynamicPillar
from .roi_head_pillar_dn_v2 import RoIHeadDynamicPillarV2
from .roi_head_pillar_pn_v2 import RoIHeadPillarNetV2


__all__ = {
    'RoIHeadTemplate': RoIHeadTemplate,
    'PartA2FCHead': PartA2FCHead,
    'PVRCNNHead': PVRCNNHead,
    'SECONDHead': SECONDHead,
    'PointRCNNHead': PointRCNNHead,
    'VoxelRCNNHead': VoxelRCNNHead,
    'RoIHeadTemplate_CenterPoint': RoIHeadTemplate_CenterPoint,
    'RoIHeadTemplate_CenterPoint_PointPillar': RoIHeadTemplate_CenterPoint_PointPillar,
    'RoIHead' : RoIHead,
    'RoIHeadPillar' : RoIHeadPillar,
    'RoIHeadPillarNet' : RoIHeadPillarNet,
    'RoIHeadDynamicPillar' : RoIHeadDynamicPillar,
    'RoIHeadPillarNetV2' : RoIHeadPillarNetV2,
    'RoIHeadDynamicPillarV2' : RoIHeadDynamicPillarV2
}
