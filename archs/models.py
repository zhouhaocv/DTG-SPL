import torch
from torch import nn
import torch.nn.functional as F
from core.modules import *
from core.config import config
from core.transformer import build_transformer
from core.captionlayers import DecoderRNN,batch_semantic_sim_score
from core.eval import *

class DMRNet(nn.Module):
    def __init__(self,cfg):
        super(DMRNet, self).__init__()
        self.anchor_num = cfg.MATCHING.PARAMS.ANCHOR_NUM
        self.sampling_num = cfg.MATCHING.PARAMS.SAMPLING_NUM
        self.extract_split = cfg.MATCHING.PARAMS.split

        self.encoding_layer = VideoLanguageEncoding(cfg)

        self.transformer = build_transformer(cfg.TRANSFORMER.PARAMS)
        self.input_embed = nn.Embedding(cfg.TRANSFORMER.PARAMS.num_queries, cfg.TRANSFORMER.PARAMS.hidden_dim)
        self.regression_head = RegLayer(cfg.REGRESSION)

        self.semantic = DecoderRNN(cfg.SEMANTIC.PARAMS)  
        self.matching =  nn.Sequential(
                nn.Conv1d(cfg.MATCHING.PARAMS.input_size, cfg.MATCHING.PARAMS.hidden_dim,1, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv1d(cfg.MATCHING.PARAMS.hidden_dim, 1, 1, 1, bias=False),
                nn.Sigmoid()
                )
        self.anchors = gen_fixed_anchors(self.anchor_num,self.sampling_num)

    def forward(self, textual_input, txt_mask, visual_input,vis_mask,gt_time,epoch):
        
        txt_h,vis_h,vis_pre = self.encoding_layer(textual_input, txt_mask,visual_input,vis_mask)
        fused_h,att_w,encoder_txt,encoder_vis = self.transformer(vis_h,txt_h, vis_mask,txt_mask, self.input_embed.weight,None)
        prediction = self.regression_head(fused_h)

        features_anchors = extract_features(encoder_vis,self.anchors.repeat(visual_input.size(0),1,1),vis_mask,split=self.extract_split)
        match_scores = self.matching(features_anchors.transpose(1,2).contiguous()).squeeze(1)

        if self.training:
            extract_feat = extract_features(vis_pre,gt_time[:,None,:],vis_mask) 
            extract_gt = extract_features(vis_pre, gt_time[:, None, :],vis_mask,type='random',random_num=5)
            extract_feat = torch.cat((extract_gt,extract_feat),dim=1) 
        else:
            extract_feat = extract_features(vis_pre,gt_time[:,None,:],vis_mask)
            extract_pred = extract_features(vis_pre,prediction[0],vis_mask) 
            extract_feat = torch.cat((extract_pred,extract_feat),dim=1) 
        caption_out,_,ret_dict = self.semantic(extract_feat)
        caption_out = torch.stack(caption_out)

        return prediction,(att_w,match_scores,caption_out,ret_dict)

    def compute_similarity(self, textual_input, txt_mask, visual_input,vis_mask,gt_time,word_vectors):
        
        txt_h,vis_h,vis_pre = self.encoding_layer(textual_input, txt_mask,visual_input,vis_mask)
        _,_,_,encoder_vis = self.transformer(vis_h,txt_h, vis_mask,txt_mask, self.input_embed.weight,None)
        features_anchors = extract_features(encoder_vis,self.anchors.repeat(visual_input.size(0),1,1),vis_mask,split=self.extract_split)
        match_scores = self.matching(features_anchors.transpose(1,2).contiguous()).squeeze(1)
        
        anchor_num = self.anchors.size(1)
        extract_feat = extract_features(vis_pre,self.anchors.repeat(visual_input.size(0),1,1),vis_mask,split=self.extract_split)
        ret_dict = self.semantic.compute_similarity(extract_feat)
        caption_symbols = torch.stack(ret_dict['sequence']).squeeze(-1).permute(1, 2, 0)
        extract_gt = extract_features(vis_pre, gt_time[:, None, :], vis_mask,type='random',random_num=5)
        ret_dict_gt = self.semantic.compute_similarity(extract_gt)
        caption_gt_symbols = torch.stack(ret_dict_gt['sequence']).squeeze(-1).permute(1, 2, 0)
        semantic_scores = [batch_semantic_sim_score(caption_symbols,caption_gt_symbols[:,i][:,None,:].repeat(1,anchor_num,1),word_vectors) for i in range(caption_gt_symbols.size(1))]
        semantic_scores,_ = torch.max(torch.stack(semantic_scores),dim=0)

        return self.anchors,semantic_scores,match_scores