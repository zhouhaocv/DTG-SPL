""" Dataset loader for the Charades-STA dataset """
import os
import csv
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
import torchtext
import spacy
from core.config import config
import h5py
import pandas as pd
from .pre_process_data import description_translate_word
import json
from core.modules import gen_fixed_anchors
from core.eval import iou
class Charades(data.Dataset):

    vocab = torchtext.vocab.pretrained_aliases["glove.6B.300d"]()
    vocab.itos.extend(['<unk>'])
    vocab.stoi['<unk>'] = vocab.vectors.shape[0]
    vocab.vectors = torch.cat([vocab.vectors, torch.zeros(1, vocab.dim)], dim=0)
    word_embedding = nn.Embedding.from_pretrained(vocab.vectors)
    nlp = spacy.load('en_core_web_lg')

    def __init__(self, split):
        super(Charades, self).__init__()

        self.data_dir = config.DATA_DIR
        self.split = split
        self.vis_input_type = config.DATASET.VIS_INPUT_TYPE
        self.trans_type = config.DTG_SPL.SEMANTIC.PARAMS.trans_type
        self.intra = config.TRAIN.INTRA_CORRELATION

        self.durations = {}
        with open(os.path.join(self.data_dir, 'Charades_v1_{}.csv'.format(split))) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.durations[row['id']] = float(row['length'])

        if self.trans_type == 'word':
            ## vocab_size:678
            if not os.path.isfile(os.path.join(self.data_dir,'charades_vocab_translate.json')):    
                with open(self.data_dir+'/charades_vocab_translate.json', 'w') as f:
                    vocab_idx,idx_vocab,max_length  = description_translate_word(self.data_dir,'charades')
                    json.dump([vocab_idx, idx_vocab,max_length], f)
            with open(os.path.join(self.data_dir,'charades_vocab_translate.json'), 'r') as f:
                vocab_idx,idx_vocab,_ = json.load(f)
            self.max_length = config.DTG_SPL.SEMANTIC.PARAMS.max_word
        else:
            raise ValueError("Unsupported Trans_type: {0}".format(self.trans_type))

        anno_file = open(os.path.join(self.data_dir, "charades_sta_{}.txt".format(self.split)),'r')
        annotations = []
        annotations_video = {}
        for line in anno_file:
            anno, sent = line.split("##")
            sent = sent.split('.\n')[0]
            vid, s_time, e_time = anno.split(" ")
            s_time = float(s_time)
            e_time = min(float(e_time), self.durations[vid])
            if s_time < e_time:
                annotations.append({'video':vid, 'times':[s_time, e_time], 'description': sent, 'duration': self.durations[vid],'match_scores':[1.0]*136,'multi_moment':[[s_time, e_time]]*2})
                if vid in annotations_video.keys():
                    annotations_video[vid]['timestamps'].append([s_time, e_time])
                    annotations_video[vid]['sentences'].append(sent)
                else:
                    annotations_video[vid] = {}
                    annotations_video[vid]['timestamps'] = [[s_time, e_time]]
                    annotations_video[vid]['sentences'] = [sent]
        anno_file.close()
        self.annotations = annotations
        self.vocab_idx = vocab_idx
        self.idx_vocab = idx_vocab
        word_idxs = torch.tensor([self.vocab.stoi.get(idx_vocab[str(ii)], 400000) for ii in range(len(idx_vocab))], dtype=torch.long)
        self.word_vectors = self.word_embedding(word_idxs)
        self.anchors = gen_fixed_anchors(config.DTG_SPL.MATCHING.PARAMS.ANCHOR_NUM,config.DTG_SPL.MATCHING.PARAMS.SAMPLING_NUM).view(-1,2).cpu().tolist()
        self.annotations_video = annotations_video

    def __getitem__(self, index):
        video_id = self.annotations[index]['video']
        gt_s_time, gt_e_time = self.annotations[index]['times']
        description = self.annotations[index]['description']+'.'
        duration = self.durations[video_id]
        multi_moment = self.annotations[index]['multi_moment']

        sentence=self.nlp(description)
        word_idxs = torch.tensor([self.vocab.stoi.get(w.norm_, 400000) for w in sentence], dtype=torch.long)
        word_vectors = self.word_embedding(word_idxs)

        #caption gt for rnn
        gt_caption = torch.zeros(self.max_length+2)
        gt_caption[0] = self.vocab_idx['<sos>']
        step = 1
        word_nv_idxs = []
        if self.trans_type == 'word':
            for token in sentence:
                if token.pos_ in ["NOUN","VERB"]:
                    if step ==1 and token.lemma_ =='person':
                        continue
                    else:
                        gt_caption[step] = self.vocab_idx[token.lemma_]
                        word_nv_idxs.append(self.vocab.stoi.get(token.norm_, 400000))
                        step +=1
        gt_caption[step] = self.vocab_idx['<eos>']

        visual_input = get_video_features(self.data_dir,self.vis_input_type,video_id)
        vis_input,vis_mask,moment_masks,moment_anchors = get_fixed_length_feat(visual_input,self.split,duration,multi_moment)
        overlaps = iou(self.anchors,torch.tensor([gt_s_time/duration, gt_e_time/duration]).tolist())
        overlaps = torch.from_numpy(overlaps)
        overlaps = overlaps/max(overlaps)
        overlaps[overlaps < 1] = 0

        ##measure intra_video correlation
        if self.intra:
            sample = self.annotations_video[video_id]
            times = sample['timestamps']
            sents = sample['sentences']
            if len(word_nv_idxs)==0:
                word_nv_idxs = torch.tensor([400000])
            else:
                word_nv_idxs = torch.tensor(word_nv_idxs, dtype=torch.long)
            word_vectors_nv = self.word_embedding(word_nv_idxs).mean(0).unsqueeze(0)
            num = len(sents)
            for ii in range(num):
                s_time,e_time = times[ii]
                sent = self.nlp(sents[ii])
                sent_idxs = []
                step = 1
                for token in sent:
                    if token.pos_ in ["NOUN","VERB"]:
                        if step ==1 and token.lemma_ =='person':
                            continue
                        else:
                            sent_idxs.append(self.vocab.stoi.get(token.norm_, 400000))
                            step +=1
                if len(sent_idxs)==0:
                    sent_idxs = torch.tensor([400000])
                else:
                    sent_idxs = torch.tensor(sent_idxs, dtype=torch.long)
                w2v = self.word_embedding(sent_idxs).mean(0).unsqueeze(0)
                sim = F.cosine_similarity(word_vectors_nv,w2v)

                overlap = iou(self.anchors,torch.tensor([s_time/duration, e_time/duration]).tolist())
                overlap = torch.from_numpy(overlap)
                overlap = overlap/max(overlap)
                overlap[overlap < 1] = 0
                overlaps = torch.max(overlaps,sim*overlap)
            max_iou = 1.0
            min_iou = 0.5
            overlaps = (overlaps-min_iou)/(max_iou-min_iou)
            overlaps[overlaps > 1] = 1.0
            overlaps[overlaps < 0.0] = 0.0

        item = {
            'anno_idx': index,
            'visual_input': vis_input,
            'vis_mask': torch.from_numpy(vis_mask).float(),
            'word_vectors': word_vectors,
            'txt_mask': torch.ones(word_vectors.shape[0], 1),
            'duration': duration,
            'gt_time': torch.tensor([gt_s_time/duration, gt_e_time/duration]),
            'moment_masks': torch.tensor(moment_masks).float(),
            'moment_anchors': torch.tensor(moment_anchors).float(),
            'gt_caption': gt_caption,
            'map_gt': overlaps.float(),
        }

        return item

    def __len__(self):
        return len(self.annotations)

class Charades_multi(data.Dataset):

    vocab = torchtext.vocab.pretrained_aliases["glove.6B.300d"]()
    vocab.itos.extend(['<unk>'])
    vocab.stoi['<unk>'] = vocab.vectors.shape[0]
    vocab.vectors = torch.cat([vocab.vectors, torch.zeros(1, vocab.dim)], dim=0)
    word_embedding = nn.Embedding.from_pretrained(vocab.vectors)
    nlp = spacy.load('en_core_web_lg')

    def __init__(self, split):
        super(Charades_multi, self).__init__()

        self.vis_input_type = config.DATASET.VIS_INPUT_TYPE
        self.data_dir = config.DATA_DIR
        self.split = split
        self.trans_type = config.DTG_SPL.SEMANTIC.PARAMS.trans_type

        self.durations = {}
        with open(os.path.join(self.data_dir, 'Charades_v1_test.csv')) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.durations[row['id']] = float(row['length'])

        data = []
        self.data_dir_new = self.data_dir + '/re-annotated/'
        for data_file in os.listdir(self.data_dir_new):
            if data_file.split('.')[-1] =='csv':
                data.append(pd.read_csv(self.data_dir_new+data_file))
        data = pd.concat(data)

        if self.trans_type == 'word':
            with open(os.path.join(self.data_dir,'charades_vocab_translate.json'), 'r') as f:
                vocab_idx,idx_vocab,_ = json.load(f)
            self.max_length = config.DTG_SPL.SEMANTIC.PARAMS.max_word
        else:
            raise ValueError("Unsupported Trans_type: {0}".format(self.trans_type))

        charade_groundtruth = pd.read_csv(self.data_dir + '/charades_test.csv')
        annotations = []
        for data_line in data['HITId'].unique():
            line = data[data['HITId']==data_line]
            vid = line.iloc[0,[1]].values
            vid = vid[0].split('.')[0]
            sent = line.iloc[0,[2]].values
            sent = sent[0]
            duration = self.durations[vid]
            multi_times = []
            for i in range(5):
                multi_times.append([float(line.iloc[i,[3]])/100*duration,float(line.iloc[i,[4]])/100*duration])
            ground_line = charade_groundtruth.loc[(charade_groundtruth['id']==vid) & (charade_groundtruth['description']==sent)]
            s_time = float(ground_line.iloc[:,1])
            e_time = float(ground_line.iloc[:,2])
            annotations.append({'video':vid, 'multi_times':multi_times,'times':[s_time, e_time],'description': sent, 'duration': duration})

        self.annotations = annotations
        self.vocab_idx = vocab_idx
        self.idx_vocab = idx_vocab
        self.anchors = gen_fixed_anchors(config.DTG_SPL.MATCHING.PARAMS.ANCHOR_NUM,config.DTG_SPL.MATCHING.PARAMS.SAMPLING_NUM).view(-1,2).cpu().tolist()

    def __getitem__(self, index):
        video_id = self.annotations[index]['video']
        gt_s_time, gt_e_time = self.annotations[index]['times']
        description = self.annotations[index]['description']
        duration = self.durations[video_id]
        multi_times = self.annotations[index]['multi_times']

        sentence=self.nlp(description)
        word_idxs = torch.tensor([self.vocab.stoi.get(w.norm_, 400000) for w in sentence], dtype=torch.long)
        word_vectors = self.word_embedding(word_idxs)

        #caption gt for rnn
        gt_caption = torch.zeros(self.max_length+2)
        gt_caption[0] = self.vocab_idx['<sos>']
        step = 1
        word_nv_idxs = []
        if self.trans_type == 'word':
            for token in sentence:
                if token.pos_ in ["NOUN","VERB"]:
                    if step ==1 and token.lemma_ =='person':
                        continue
                    else:
                        gt_caption[step] = self.vocab_idx[token.lemma_]
                        word_nv_idxs.append(self.vocab.stoi.get(token.norm_, 400000))
                        step +=1
        gt_caption[step] = self.vocab_idx['<eos>']

        visual_input = get_video_features(self.data_dir,self.vis_input_type,video_id)
        vis_input,vis_mask,_,_ = get_fixed_length_feat(visual_input,self.split,duration)
        overlaps = iou(self.anchors,torch.tensor([gt_s_time/duration, gt_e_time/duration]).tolist())
        overlaps = torch.from_numpy(overlaps).float()
        item = {
            'anno_idx': index,
            'visual_input': vis_input,
            'vis_mask': torch.from_numpy(vis_mask).float(),
            'word_vectors': word_vectors,
            'txt_mask': torch.ones(word_vectors.shape[0], 1),
            'duration': duration,
            'gt_time': torch.tensor([gt_s_time/duration, gt_e_time/duration]),
            'multi_time': torch.tensor(multi_times),
            'gt_caption': gt_caption,
            'map_gt': overlaps,
        }
        return item

    def __len__(self):
        return len(self.annotations)

def get_fixed_length_feat(feat,split,duration,multi_moment=None):
    num_segment = config.DATASET.NUM_SAMPLE_CLIPS
    nfeats = feat.shape[0]
    if nfeats <= num_segment:
        stride = 1
    else:
        stride = nfeats * 1.0 / num_segment
    if split != "train":
        spos = 0
    else:
        random_end = -0.5 + stride
        if random_end == np.floor(random_end):
            random_end = random_end - 1.0
        spos = np.random.random_integers(0,random_end)
    s = np.round( np.arange(spos, nfeats-0.5, stride) ).astype(int)

    if not (nfeats < num_segment and len(s) == nfeats) \
            and not (nfeats >= num_segment and len(s) == num_segment):
        s = s[:num_segment] # ignore last one
    assert (nfeats < num_segment and len(s) == nfeats) \
           or (nfeats >= num_segment and len(s) == num_segment), \
        "{} != {} or {} != {}".format(len(s), nfeats, len(s), num_segment)

    cur_feat = feat[s, :]
    nfeats = min(nfeats, num_segment)
    out = np.zeros((num_segment, cur_feat.shape[1]))
    out [:nfeats,:] = cur_feat
    out = torch.from_numpy(out)

    vid_mask = np.zeros((num_segment, 1))
    vid_mask[:nfeats] = 1

    if multi_moment is None:
        moment_masks = None
        moment_anchors = None
    else:
        num_moments = len(multi_moment)
        moment_masks = np.zeros((num_moments,num_segment))
        moment_anchors = np.zeros((num_moments,2))
        for ii in range(num_moments):
            moment = multi_moment[ii]
            if type(moment).__name__ == 'str':
                s_time, e_time = moment.split(" ")
            else:
                s_time, e_time = moment
            s_time = float(s_time)/duration
            e_time = min(float(e_time), duration)/duration
            start_pos =  float(nfeats-1.0) * s_time
            end_pos = float(nfeats-1.0) * e_time

            start_index, end_index =  None, None
            for i in range(len(s)-1):
                if s[i] <= end_pos < s[i+1]:
                    end_index = i
                if s[i] <= start_pos < s[i+1]:
                    start_index = i
            if start_index is None:
                start_index = 0
            if end_index is None:
                # end_index = num_segment-1
                end_index = nfeats-1

            att_s = np.zeros(s.shape[0])
            att_s[start_index:end_index+1] =1
            moment_masks[ii,:nfeats] = att_s

            moment_anchors[ii,0] = s_time
            moment_anchors[ii,1] = e_time

    return out, vid_mask, moment_masks,moment_anchors

def get_video_features(data_dir,vis_input_type, vid):
    if vis_input_type == 'i3d_v2':
        hdf5_file = h5py.File(os.path.join(data_dir, 'charades_i3d_rgb.hdf5'), 'r')
        features = torch.from_numpy(np.array(hdf5_file[vid]['i3d_rgb_features'])).float()
    elif vis_input_type == 'i3d':
        hdf5_file = np.load(os.path.join(data_dir, 'features/{}.npy'.format(vid)), 'r')
        features = torch.from_numpy(np.array(hdf5_file[:])).float()
        features = features.squeeze(1)
        features = features.squeeze(1)
    elif vis_input_type == 'vgg_rgb':
        hdf5_file = h5py.File(os.path.join(data_dir, 'vgg_rgb_features.hdf5'), 'r')
        features = torch.from_numpy(hdf5_file[vid][:]).float()
    elif vis_input_type == 'c3d':
        hdf5_file = h5py.File(os.path.join(data_dir, 'charades_c3d_fc6_nonoverlap.hdf5'), 'r')
        features = torch.from_numpy(np.array(hdf5_file[vid]['c3d_fc6_features'])).float()
    else:
        print('there is not corresponding vis_input_type!')

    if config.DATASET.NORMALIZE:
        features = F.normalize(features,dim=1)
    return features

def collate_fn(batch):
    max_num_clips = config.DATASET.NUM_SAMPLE_CLIPS
    batchsize =  len(batch)
    batch_word_vectors = [b['word_vectors'] for b in batch]
    batch_txt_mask = [b['txt_mask'] for b in batch]
    batch_anno_idxs = [b['anno_idx'] for b in batch]
    batch_vis_feats = [b['visual_input'] for b in batch]
    batch_duration = [b['duration'] for b in batch]

    batch_gt_time = [b['gt_time'] for b in batch]
    batch_gt_time = torch.cat(batch_gt_time,dim=0).reshape(batchsize,2)
    batch_vis_mask = [b['vis_mask'] for b in batch]
    batch_vis_mask = torch.cat(batch_vis_mask,dim=0).reshape(batchsize,max_num_clips)
    batch_gt_caption = [b['gt_caption'] for b in batch]
    batch_gt_caption = torch.cat(batch_gt_caption,dim=0).reshape(batchsize,-1)

    batch_moment_masks = [b['moment_masks'][1:] for b in batch]
    batch_moment_masks = torch.cat(batch_moment_masks,dim=0)
    batch_moment_anchors = [b['moment_anchors'][1:] for b in batch]
    batch_moment_anchors = torch.cat(batch_moment_anchors,dim=0)
    batch_moment_num = [len(b['moment_masks'][1:]) for b in batch]
    batch_gt_masks = [b['moment_masks'][0] for b in batch]
    batch_gt_masks = torch.cat(batch_gt_masks,dim=0).reshape(batchsize,max_num_clips)
    batch_map_gt = [b['map_gt'] for b in batch]
    batch_map_gt = torch.cat(batch_map_gt,dim=0).reshape(batchsize,-1)


    batch_data = {
        'batch_anno_idxs': batch_anno_idxs,
        'batch_word_vectors': nn.utils.rnn.pad_sequence(batch_word_vectors, batch_first=True),
        'batch_txt_mask': nn.utils.rnn.pad_sequence(batch_txt_mask, batch_first=True),
        'batch_vis_input': nn.utils.rnn.pad_sequence(batch_vis_feats, batch_first=True).float(),
        'batch_duration': batch_duration,
        'batch_gt_time': batch_gt_time,
        'batch_vis_mask': batch_vis_mask,
        'batch_moment_masks': batch_moment_masks,
        'batch_moment_anchors': batch_moment_anchors,
        'batch_gt_caption': batch_gt_caption,   
        'batch_moment_num': batch_moment_num, 
        'batch_gt_masks': batch_gt_masks,
        'batch_map_gt':batch_map_gt,       
    }

    return batch_data

def collate_fn_multi(batch):
    max_num_clips = config.DATASET.NUM_SAMPLE_CLIPS
    batchsize =  len(batch)
    batch_word_vectors = [b['word_vectors'] for b in batch]
    batch_txt_mask = [b['txt_mask'] for b in batch]
    batch_anno_idxs = [b['anno_idx'] for b in batch]
    batch_vis_feats = [b['visual_input'] for b in batch]
    batch_duration = [b['duration'] for b in batch]

    batch_multi_time = [b['multi_time'] for b in batch]
    batch_multi_time = torch.cat(batch_multi_time,dim=0).reshape(-1,5,2)
    batch_gt_time = [b['gt_time'] for b in batch]
    batch_gt_time = torch.cat(batch_gt_time,dim=0).reshape(-1,2)
    batch_vis_mask = [b['vis_mask'] for b in batch]
    batch_vis_mask = torch.cat(batch_vis_mask,dim=0).reshape(-1,max_num_clips)
    batch_gt_caption = [b['gt_caption'] for b in batch]
    batch_gt_caption = torch.cat(batch_gt_caption,dim=0).reshape(batchsize,-1)
    batch_map_gt = [b['map_gt'] for b in batch]
    batch_map_gt = torch.cat(batch_map_gt,dim=0).reshape(batchsize,-1)

    batch_data = {
        'batch_anno_idxs': batch_anno_idxs,
        'batch_word_vectors': nn.utils.rnn.pad_sequence(batch_word_vectors, batch_first=True),
        'batch_txt_mask': nn.utils.rnn.pad_sequence(batch_txt_mask, batch_first=True),
        'batch_vis_input': nn.utils.rnn.pad_sequence(batch_vis_feats, batch_first=True).float(),
        'batch_duration': batch_duration,
        'batch_gt_time': batch_gt_time,
        'batch_vis_mask': batch_vis_mask,
        'batch_multi_time': batch_multi_time,
        'batch_gt_caption': batch_gt_caption,   
        'batch_map_gt':batch_map_gt, 
    }

    return batch_data