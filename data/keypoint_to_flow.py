r'''modified source code from DHPF https://github.com/juhongm999/dhpf'''

import torch
import pdb

class KeypointToFlow:
    def __init__(self, receptive_field_size=35, jsz=16, feat_size=16, img_size=256):
        self.feat_size = feat_size
        self.img_size = img_size
        self.box, self.feat_ids = self.receptive_fields(receptive_field_size, jsz, feat_size)
    
    def receptive_fields(self, receptive_field_size, jsz, feat_size):
        r"""Returns a set of receptive fields (N, 4) N=hw
        box         : [N, 4]
        feat_ids    : [N, 2] (height, width)
        """
        width = feat_size
        height = feat_size

        feat_ids = torch.tensor(list(range(width))).repeat(1, height).t().repeat(1, 2)
        feat_ids[:, 0] = torch.tensor(list(range(height))).unsqueeze(1).repeat(1, width).view(-1)

        box = torch.zeros(feat_ids.size()[0], 4)
        box[:, 0] = feat_ids[:, 1] * jsz - receptive_field_size // 2 + jsz // 2 
        box[:, 1] = feat_ids[:, 0] * jsz - receptive_field_size // 2 + jsz // 2 
        box[:, 2] = feat_ids[:, 1] * jsz + receptive_field_size // 2 + jsz // 2
        box[:, 3] = feat_ids[:, 0] * jsz + receptive_field_size // 2 + jsz // 2

        return box, feat_ids

    def neighbours(self, box, kps):
        r"""Returns boxes in one-hot format that covers given keypoints
        box : [hw, 4]
        kps : [2, n_pts]

        return
        nbr_onehot.shape    : [n_pts, hw]
        n_neighbours.shape  : [n_pts]
        n_points.shape      : [hw]
        """
        box_duplicate = box.unsqueeze(2).repeat(1, 1, len(kps.t())).transpose(0, 1) # [4, hw, n_pts]
        kps_duplicate = kps.unsqueeze(1).repeat(1, len(box), 1)                     # [2, hw, n_pts]

        xmin = kps_duplicate[0].ge(box_duplicate[0])    # [hw, n_pts]
        ymin = kps_duplicate[1].ge(box_duplicate[1])
        xmax = kps_duplicate[0].le(box_duplicate[2])
        ymax = kps_duplicate[1].le(box_duplicate[3])

        nbr_onehot = torch.mul(torch.mul(xmin, ymin), torch.mul(xmax, ymax)).t()   # [n_pts, hw]
        n_neighbours = nbr_onehot.sum(dim=1)                                       # [n_pts]
        n_points = nbr_onehot.sum(dim=0)                                           # [hw]

        return nbr_onehot, n_neighbours, n_points

    def __call__(self, batch):
        """
        src_kps : [k, 2]
        trg_kps : [k, 2]
        n_pts   : seleted keypoints
        우선, k=40으로 설정 => n_pts개만 keypoint 나머지 padding
        """
        src_kps, trg_kps, n_pts = batch['src_kps'].t(), batch['trg_kps'].t(), batch['n_pts']

        kp = trg_kps.narrow_copy(0, 0, n_pts)       # [n_pts, 2]
        kp_src = src_kps.narrow_copy(0, 0, n_pts)   # [n_pts, 2]

        src_nbr_onehot, n_neighbours, n_points = self.neighbours(self.box, kp.t())

        center = torch.stack(((self.box[:, 0] + self.box[:, 2])/2, (self.box[:, 1] + self.box[:, 3])/2), dim=1) # [hw, 2]
        center = center.unsqueeze(0).repeat(len(kp), 1, 1)          # [n_pts, hw, 2]

        src_idx = src_nbr_onehot.nonzero()          # [N, 2] N:겹치는 것을 제외한 idx / 2=[kp idx, spatial idx] 
        src_nn = center[src_idx[:,0],src_idx[:,1]]  # 
        kp_selected = kp[src_idx[:,0],:]            # [N, 2] 겹치는 것을 제외한 kp의 좌표

        vector_summator = torch.zeros_like(center)  # [n_pts, hw, 2] 
        vector_summator[src_idx[:, 0], src_idx[:, 1]] = kp_selected # []

        n_points_expanded = n_points.unsqueeze(1).repeat(1,2).float()   # [hw, 2]
        n_points_expanded[n_points_expanded == 0] = 1.                  # [hw, 2]

        source_averaged = (vector_summator.sum(dim=0) / n_points_expanded)[src_idx[:,1]]    # [hw, 2]

        flow = kp_src[src_idx[:,0],:] - source_averaged
        
        flow_index = self.feat_ids.index_select(dim=0, index=src_idx[:,1]) # [256, 2] - [N, 2]

        flow_map = torch.zeros(self.feat_size, self.feat_size, 2)   # [16, 16, 2]
        flow_map[flow_index[:,0],flow_index[:,1]] = flow / (self.img_size // self.feat_size)

        flow_map = flow_map.permute(2, 0, 1)     # [2, 16, 16]
        
        return flow_map