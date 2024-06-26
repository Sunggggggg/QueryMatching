import torch
import torch.nn as nn
import numpy as np
from .backbone import FeatureExtractionHyperPixel, SimpleResnet
from .pos_encoding import PositionEmbeddingSine
from .mod import FeatureL2Norm, unnormalise_and_convert_mapping_to_flow

class SimCLR_Loss(nn.Module):
    def __init__(self, batch_size, temperature):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)
        
        labels = torch.from_numpy(np.array([0]*N)).reshape(-1).to(positive_samples.device).long()
        
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        
        return loss

class QueryActivation(nn.Module):
    def __init__(self, model_dim, nhead=8, dropout=0.0) :
        super(QueryActivation, self).__init__()
        self.nhead = nhead
        self.scale = (model_dim // nhead)** -0.5

        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.proj = nn.Linear(model_dim, model_dim, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.prj_drop = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(model_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        torch.manual_seed(0)
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.proj.weight)
        if self.k_proj.bias is not None:
            nn.init.xavier_normal_(self.k_proj.bias)
        if self.v_proj.bias is not None:
            nn.init.xavier_normal_(self.v_proj.bias)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0.)

    def forward(self, query, key, value=None):
        if value is None:
            value = key
        B, N_q, C_q = query.shape
        B, N_k, C_k = key.shape
        B, N_v, C_v = value.shape
        _query = query

        query = self.q_proj(query).reshape(B, N_q, self.nhead, C_q//self.nhead).permute(0, 2, 1, 3)
        key = self.k_proj(key).reshape(B, N_k, self.nhead, C_k//self.nhead).permute(0, 2, 1, 3)        
        value = self.v_proj(value).reshape(B, N_v, self.nhead, C_v//self.nhead).permute(0, 2, 1, 3)

        attn = (query @ key.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ value).transpose(1, 2).reshape(B, N_q, -1) 
        x = self.proj(x)
        x = self.prj_drop(x)
        x = self.norm(x + _query)
        return x

class CATransformer(nn.Module):
    """ No Multi-head attention
    """
    def __init__(self, query_dim, key_dim, value_dim, dropout=0.0):
        super(CATransformer, self).__init__()
        self.scale = (value_dim)** -0.5
        self.q_proj = nn.Linear(query_dim, query_dim, bias=False)
        self.k_proj = nn.Linear(key_dim, key_dim, bias=False)
        self.v_proj = nn.Linear(value_dim, value_dim, bias=False)
        self.proj = nn.Linear(value_dim, value_dim, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.prj_drop = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(value_dim)

    def forward(self, query, key, value=None):
        B, N_q, C_q = query.shape

        query = self.q_proj(query)
        key = self.k_proj(key)   
        value = self.v_proj(value)
        _value = value

        attn = (query @ key.transpose(-2, -1)) * self.scale # [B, h, N, e] [B, h, e, N] = [B, h, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ value).transpose(1, 2).reshape(B, N_q, -1)  # [B, h, N, N] @ [B, h, N, e]  = [B, h, N, e] - [B, N, h, e] 
        x = self.proj(x)
        x = self.prj_drop(x)
        x = self.norm(x + _value)
        return x

class FFN(nn.Module):
    def __init__(self, model_dim, dim_feedforward=2048, dropout=0.0):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(model_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, model_dim)

        self.norm = nn.LayerNorm(model_dim)
        self.activation = nn.GELU()
    
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, query):
        query = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout(query)

        return query

class QueryAggregation(nn.Module):
    def __init__(self, num_queries=100, model_dim=256) :
        super(QueryAggregation, self).__init__()
        self.num_queries = num_queries
        self.model_dim = model_dim
        #####################################
        # Transformer 
        #####################################
        self.query_trans = CATransformer(query_dim=num_queries+model_dim, key_dim=num_queries+model_dim, value_dim=model_dim)
        self.cost_trans = CATransformer(query_dim=num_queries+model_dim, key_dim=num_queries+model_dim, value_dim=num_queries)

        #####################################
        # FFN
        #####################################
        self.ffn = FFN(model_dim=model_dim, dim_feedforward=model_dim*4)

        #####################################
        # Norm
        #####################################
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)
        self.norm4 = nn.LayerNorm(model_dim)
        self.norm5 = nn.LayerNorm(model_dim)
        self.norm6 = nn.LayerNorm(model_dim)
        self.norm7 = nn.LayerNorm(model_dim)
        self.norm8 = nn.LayerNorm(model_dim)

    def matual_nn_filter(self, correlation_matrix) :
        """Mutual nearest neighbor filtering (Rocco et al. NeurIPS'18)
        correlation_matrix : [B, Q1, Q2]
        """
        corr_src_max = torch.max(correlation_matrix, dim=2, keepdim=True)[0]
        corr_trg_max = torch.max(correlation_matrix, dim=1, keepdim=True)[0]
        corr_src_max[corr_src_max == 0] += 1e-30
        corr_trg_max[corr_trg_max == 0] += 1e-30

        corr_src = correlation_matrix / corr_src_max
        corr_trg = correlation_matrix / corr_trg_max

        return correlation_matrix * (corr_src * corr_trg)

    def forward(self, query1, query2, query1_pos, query2_pos, cost_vol_pos):
        """
        query1, query2          : [B, Q, e]
        query1_pos, query2_pos  : [Q, e]
        cost_vol_pos            : [Q, Q]
        """
        query1, query2 = self.norm1(query1), self.norm2(query2)
        cost_volume = query1 @ query2.permute(0, 2, 1)
        init_query1, init_query2 = query1, query2 

        # Positional embedding
        query1 = query1 + query1_pos
        query2 = query2 + query2_pos
        cost_volume = cost_volume + cost_vol_pos

        cost_feat1 = torch.cat([cost_volume, query1], dim=-1)                    # [B, Q1, (Q2+e)]
        cost_feat2 = torch.cat([cost_volume.permute(0, 2, 1), query2], dim=-1)   # [B, Q2, (Q1+e)]

        # Query Aggregation
        _query1, _query2 = query1, query2
        query1 = _query1 + self.query_trans(cost_feat1, cost_feat1, _query1)
        query2 = _query2 + self.query_trans(cost_feat2, cost_feat2, _query2)
        query1, query2 = self.norm3(query1), self.norm4(query2)

        _query1, _query2 = query1, query2
        query1 = _query1 + self.ffn(_query1)
        query2 = _query2 + self.ffn(_query2)
        query1, query2 = self.norm5(query1), self.norm6(query2)

        # Cost volme Aggregation
        cost_volume1 = self.cost_trans(cost_feat1, cost_feat1, cost_volume)
        cost_volume2 = self.cost_trans(cost_feat2, cost_feat2, cost_volume.permute(0, 2, 1))
        cost_volume = cost_volume + cost_volume1 + cost_volume2.permute(0, 2, 1)    # Refine cost vol
        correlation_matrix = self.matual_nn_filter(cost_volume)              # [B, Q1, Q2]
        
        # Refine Query
        _query1, _query2 = query1, query2
        query1 = _query1 + correlation_matrix @ _query2                  # [B, Q1, Q2] [B, Q2, e]
        query2 = _query2 + correlation_matrix.permute(0, 2, 1) @ _query1 # [B, Q2, Q1] [B, Q1, e]

        query1 = query1 + init_query1
        query2 = query2 + init_query2
        query1, query2 = self.norm7(query1), self.norm8(query2)
        return query1, query2

class QueryMatching(nn.Module):
    def __init__(self, 
                 backbone_name="simple_resnet",
                 hyperpixel_ids=[0,8,20,21,26,28,29,30], 
                 feat_dim=256, 
                 freeze=True,
                 depth=4,
                 num_queries=16,
                 feature_size=16,
                 ):
        super(QueryMatching, self).__init__()
        self.num_queries = num_queries
        self.depth = depth
        self.feature_size = feature_size
        #####################################
        # Backbone 
        #####################################
        if backbone_name == "simple_resnet" :
            self.feature_extraction = SimpleResnet(feat_dim, freeze)
            self.num_stage = 3   # Fix
        else :  # CATs Backbone
            self.feature_extraction = FeatureExtractionHyperPixel(hyperpixel_ids, feat_dim, freeze) # 8
            self.num_stage = len(hyperpixel_ids)
        self.total_depth = self.num_stage * depth
        self.feat_pos = PositionEmbeddingSine(feat_dim//2)

        #####################################
        # Query 
        #####################################
        self.query = nn.Parameter(torch.zeros(num_queries, feat_dim), requires_grad=True)
        self.query_emb = nn.Parameter(torch.rand(num_queries, feat_dim), requires_grad=True)
        self.query_emb1 = nn.Parameter(torch.rand(num_queries, 1), requires_grad=True)
        self.query_emb2 = nn.Parameter(torch.rand(num_queries, 1), requires_grad=True)
        
        #####################################
        # Query Aggregation
        #####################################
        self.query_SA = nn.ModuleList()
        self.query_CA = nn.ModuleList()
        self.query_aggregation = nn.ModuleList()

        for _ in range(self.depth):
            self.query_SA.append(QueryActivation(model_dim=feat_dim, nhead=8))    # Self Atttention

        for _ in range(self.total_depth):
            self.query_CA.append(QueryActivation(model_dim=feat_dim, nhead=8))
            self.query_aggregation.append(QueryAggregation(model_dim=feat_dim, num_queries=num_queries))

        #####################################
        # Scalar
        #####################################
        self.l2norm = FeatureL2Norm()
        self.x_normal = np.linspace(-1,1,self.feature_size)
        self.x_normal = nn.Parameter(torch.tensor(self.x_normal, dtype=torch.float, requires_grad=False))
        self.y_normal = np.linspace(-1,1,self.feature_size)
        self.y_normal = nn.Parameter(torch.tensor(self.y_normal, dtype=torch.float, requires_grad=False))
        self.flow_map = np.zeros((2, self.feature_size, self.feature_size))
        self.flow_map = nn.Parameter(torch.tensor(self.flow_map, dtype=torch.float, requires_grad=False))
        
        #####################################
        # Loss
        #####################################
        self.loss_func = SimCLR_Loss(self.num_queries, 0.5)

    def softmax_with_temperature(self, x, beta, d = 1):
        r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
        M, _ = x.max(dim=d, keepdim=True)
        x = x - M # subtract maximum value for stability
        exp_x = torch.exp(x/beta)
        exp_x_sum = exp_x.sum(dim=d, keepdim=True)
        return exp_x / exp_x_sum

    def soft_argmax(self, src_heatmap, tgt_heatmap, h, w, beta=0.02):
        B, Q, hw = src_heatmap.shape
        src_heatmap = self.softmax_with_temperature(src_heatmap, beta=beta) #[B, Q, hw]

        # 
        src_heatmap = src_heatmap.reshape(B, Q, h, w)

        src_grid_x = src_heatmap.sum(dim=2, keepdim=False)      # [B, Q, w]
        x_normal = self.x_normal.view(1, 1, w)              # [w]
        src_grid_x = (x_normal*src_grid_x).sum(dim=-1, keepdim=False)# [B, Q]

        src_grid_y = src_heatmap.sum(dim=3, keepdim=False)      # [B, Q, h]
        y_normal = self.y_normal.view(1, 1, h)              # [h]
        src_grid_y = (y_normal*src_grid_y).sum(dim=-1, keepdim=False)# [B, Q]

        _, tgt_max_idx = torch.max(tgt_heatmap, dim=-1)   # [B, Q]
        tgt_max_row = tgt_max_idx // h
        tgt_max_col = tgt_max_idx % h    # [B, Q]
        
        flow_map = self.flow_map.repeat(B, 1, 1, 1)   # [B, 2, h, w]
        for b in range(B):
            grid_x, grid_y = src_grid_x[b], src_grid_y[b]   # [Q]
            tgt_row, tgt_col = tgt_max_row[b], tgt_max_col[b]   # [Q]
            for sr, sc, tr, tc in zip(grid_x, grid_y, tgt_row, tgt_col):
                flow_map[b, 0, tr, tc] = sr
                flow_map[b, 1, tr, tc] = sc

        return flow_map

    def heatmap2flow(self, src_heatmap, tgt_heatmap, h, w, beta=0.02):
        r'''
        src_heatmap, tgt_heatmap [B, Q, hw]
        '''
        B = src_heatmap.shape[0]

        _, src_max_idx = torch.max(src_heatmap, dim=-1)   # [B, Q]
        src_max_row = src_max_idx // h
        src_max_col = src_max_idx % h    # [B, Q]

        _, tgt_max_idx = torch.max(tgt_heatmap, dim=-1)   # [B, Q]
        tgt_max_row = tgt_max_idx // h
        tgt_max_col = tgt_max_idx % h    # [B, Q]
        
        flow_map = self.flow_map.repeat(B, 1, 1, 1)   # [B, 2, h, w]
        for b in range(B):
            src_row, src_col = src_max_row[b], src_max_col[b]   # [Q]
            tgt_row, tgt_col = tgt_max_row[b], tgt_max_col[b]   # [Q]
            for sr, sc, tr, tc in zip(src_row, src_col, tgt_row, tgt_col):
                flow_map[b, 0, tr, tc] = sr - tr
                flow_map[b, 1, tr, tc] = sc - tc
        
        return flow_map

    def forward(self, source, target):
        B = source.shape[0]

        # Query
        keypoint_query = self.query.unsqueeze(0).repeat(B, 1, 1)
        query1, query2 = keypoint_query, keypoint_query

        # Query pos
        query_pos = self.query_emb.unsqueeze(0)         # [1, Q, e]
        rel_query1_pos = self.query_emb1.unsqueeze(0)   # [1, Q, 1]
        rel_query2_pos = self.query_emb2.unsqueeze(0)   # [1, Q, 1]
        query1_pos = query_pos + rel_query1_pos         # [1, Q, e]
        query2_pos = query_pos + rel_query2_pos         # [1, Q, e]
        cost_vol_pos =  rel_query1_pos.repeat(1, 1, self.num_queries) \
            + rel_query2_pos.permute(0, 2, 1).repeat(1, self.num_queries, 1) # [1, Q1, Q2]

        # Feature map
        src_feats, tgt_feats = self.feature_extraction(source), self.feature_extraction(target)  # [l, B, dim, H, W] small to large

        for d in range(self.depth):
            query1 = query1 + query1_pos
            query2 = query2 + query2_pos
            query1 = self.query_SA[d](query=query1, key=query1)
            query2 = self.query_SA[d](query=query2, key=query2)

            for s in range(self.num_stage):
                src_feat, tgt_feat = src_feats[s], tgt_feats[s]
                src_feat, tgt_feat = self.l2norm(src_feat), self.l2norm(tgt_feat)
                feat_pos = self.feat_pos(src_feat)

                query1 = query1 + query1_pos    # [B, Q, e]
                query2 = query2 + query2_pos
                src_feat = src_feat + feat_pos  # [B, e, h, w]
                tgt_feat = tgt_feat + feat_pos

                src_feat = src_feat.flatten(2).permute(0, 2, 1) # [B, hw, e]
                tgt_feat = tgt_feat.flatten(2).permute(0, 2, 1)

                level = d * self.num_stage + s
                query1 = self.query_CA[level](query=query1, key=src_feat)
                query2 = self.query_CA[level](query=query2, key=tgt_feat)
                query1, query2 = self.query_aggregation[level](query1, query2, query1_pos, query2_pos, cost_vol_pos)

        # Fuse Block
        contr_loss = torch.stack([self.loss_func(query1[b], query2[b]) for b in range(B)]).mean()
        src_feat, tgt_feat = self.l2norm(src_feats[0]), self.l2norm(tgt_feats[0])
        h, w = src_feat.shape[-2:]
        src_feat = src_feat.flatten(2) # [B, e, hw]
        tgt_feat = tgt_feat.flatten(2)
        src_heatmap = query1 @ src_feat # [B, Q, e][B, e, hw]=[B, Q, hw]
        tgt_heatmap = query2 @ tgt_feat # [B, Q, e][B, e, hw]=[B, Q, hw]

        #flow = self.heatmap2flow(src_heatmap, tgt_heatmap, h, w)
        flow = self.soft_argmax(src_heatmap, tgt_heatmap, h, w)
        flow = unnormalise_and_convert_mapping_to_flow(flow)
        
        return flow, contr_loss

if __name__ == "__main__" :
    x = torch.rand((1, 3, 256, 256))
    y = torch.rand((1, 3, 256, 256))
    model = QueryMatching()
    model(x, y)