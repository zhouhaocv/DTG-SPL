import torch
from torch import nn
import torch.nn.functional as F


class VideoLanguageEncoding(nn.Module):

    def __init__(self, cfg):
        super(VideoLanguageEncoding, self).__init__()
        self.pre_fuse = cfg.QUERY_EMBEDDER.PARAMS.pre_fuse
        txt_input_size = cfg.QUERY_EMBEDDER.PARAMS.txt_input_size
        txt_hidden_size = cfg.QUERY_EMBEDDER.PARAMS.txt_hidden_dim
        vis_input_size = cfg.VIDEO_EMBEDDER.PARAMS.input_size
        vis_hidden_size = cfg.VIDEO_EMBEDDER.PARAMS.hidden_dim

        self.textual_encoder = nn.GRU(txt_input_size, txt_hidden_size//2 if cfg.QUERY_EMBEDDER.PARAMS.GRU.bidirectional else txt_hidden_size,
                                       num_layers=cfg.QUERY_EMBEDDER.PARAMS.GRU.num_layers, bidirectional=cfg.QUERY_EMBEDDER.PARAMS.GRU.bidirectional, batch_first=True)
        self.frame_layer =  nn.Sequential(
                nn.Conv1d(vis_input_size, vis_hidden_size,1, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv1d(vis_hidden_size, vis_hidden_size, 1, 1, bias=False),
                )

    def forward(self, textual_input,txt_mask,visual_input,vis_mask):
        '''

        '''
        self.textual_encoder.flatten_parameters()
        txt_h = self.textual_encoder(textual_input)[0]
        vis_h = self.frame_layer(visual_input.transpose(1, 2))*vis_mask[:,None,:]

        # better performance with pre_fuse, especially on charades-STA
        if self.pre_fuse:
            txt_h_1 = torch.stack([torch.sum(txt_h[i],dim=0)/torch.sum(mask) for i, mask in enumerate(txt_mask)])
            vis_h2 = F.normalize(txt_h_1[:,:,None] * vis_h)*vis_mask[:,None,:]
            return txt_h,vis_h2,vis_h
        else:
            return txt_h,vis_h,vis_h


class RegLayer(nn.Module):

    def __init__(self, cfg):
        super(RegLayer, self).__init__()
        self.cfg = cfg
        hidden_size = cfg.PARAMS.hidden_dim

        self.reg1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
            # nn.Sigmoid(),
            )
        self.reg2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
            # nn.Sigmoid(),
            )

    def forward(self, f_out):
        f_se = self.reg1(f_out)
        f_cw = self.reg2(f_out)
        return f_se,f_cw

def extract_features(feat,pred,vis_mask,split=False,type='avg',random_num=10):

    """
    Input:
        feat: (batch, hidden_dim, num_segment), e.g. 32 * 512 *128
        pred: (batch, num_moments,2)
        vis_mask: (batch, num_segment)
    Output:
        out_feat: (batch, num_moments, hidden_dim)
        masks: (batch, num_moments,num_segment)
    """


    num_segment = vis_mask.size(-1)
    batch_size = pred.size(0)
    num_moments = pred.size(1)
    vis_length = vis_mask.sum(-1)[:,None,None].repeat(1,num_moments,1)
    prediction = pred.clone().detach()
    prediction = torch.clamp(prediction,min=0.0,max=1.0)

    index = torch.round(vis_length * prediction).long()
    lower = index[:,:,0].unsqueeze(dim=-1)
    upper = index[:,:,1].unsqueeze(dim=-1)
    idx = torch.arange(num_segment).to(prediction.get_device())
    masks = idx.ge(lower) & idx.le(upper)
    masks = masks.long()
    masks_avg = masks / (masks.sum(-1)[:,:,None]+1e-10)

    if feat is not None:
        if type =="avg":
            if split:
                for i in range(batch_size):
                    if i == 0:
                        out_feat = (feat[i,None,:,:] * masks_avg[i,:,None,:]).sum(-1).unsqueeze(0)
                    else:
                        out_feat_p = (feat[i,None,:,:] * masks_avg[i,:,None,:]).sum(-1).unsqueeze(0)
                        out_feat = torch.cat((out_feat,out_feat_p),dim=0)
            else:
                out_feat = (feat[:,None,:,:] * masks_avg[:,:,None,:]).sum(-1)
        elif type =="random":
            mask_random = torch.rand(batch_size,num_moments*random_num,num_segment).to(prediction.get_device())
            masks2 = masks * mask_random
            masks_avg = masks2 / (masks2.sum(-1)[:,:,None]+1e-10)
            out_feat = (feat[:,None,:,:] * masks_avg[:,:,None,:]).sum(-1)
        elif type =="max":
            masks2 = (masks-1)*1e10
            out_feat = torch.max(feat[:,None,:,:] * masks2[:,:,None,:],dim=-1)[0]
    else:
        out_feat = None

    return out_feat

from matplotlib import pyplot as plt
def gen_fixed_anchors(anchor_num,sampling_num = [16]):

    """
    motivated by 2D-TAN
    """
    start_idx = torch.arange(0,anchor_num).float()/anchor_num
    end_idx = torch.arange(1,anchor_num+1).float()/anchor_num

    anchors = torch.stack([start_idx[:,None].expand(-1,anchor_num),end_idx[None,:].expand(anchor_num,-1)],dim=2).view(-1,2)
    # mask = torch.triu(torch.ones(anchor_num,anchor_num),diagonal=0)
    mask = torch.zeros(anchor_num,anchor_num)
    start_index = 0
    for i, num in enumerate(sampling_num):
        stride = 2**i
        for line in range(0,anchor_num,stride):
            scale_idxs = list(range(min(anchor_num,line+start_index), min(anchor_num,line+(stride*num)+start_index), stride))
            mask[line, scale_idxs] = 1
        start_index += stride * num + stride
    # plt.matshow(mask)
    # plt.show()
    mask = mask.view(-1)
    anchors = anchors[mask==1,:]
    if torch.cuda.is_available():
        anchors = anchors.cuda()
    return anchors.unsqueeze(0)


if __name__ == "__main__":
    anchors  = gen_fixed_anchors(16)
    print(anchors.size(),anchors)

