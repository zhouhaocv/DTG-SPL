from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import pprint
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
import time
import math

from core.config import config, update_config
from core.engine import Engine
from core.utils import AverageMeter,create_logger
from core.eval import *
from core.loss import *
from models import DMRNet
from datasets.charades import collate_fn, Charades,collate_fn_multi, Charades_multi
import random
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
def parse_args():
    parser = argparse.ArgumentParser(description='Train Diverse Temporal Grounding Network')

    # general
    parser.add_argument('--cfg',default="./../archs/cfg/charades/DTG-SPL-charades-i3d.yaml", help='experiment configure file name', type=str)
    args, rest = parser.parse_known_args()

    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--gpus', help='gpus', type=str)
    parser.add_argument('--workers', help='num of dataloader workers', type=int)
    parser.add_argument('--dataDir', help='data path', type=str)
    parser.add_argument('--modelDir', help='model path', type=str)
    parser.add_argument('--logDir', help='log path', type=str)
    parser.add_argument('--verbose', default=True, action="store_true", help='print progress bar')
    parser.add_argument('--gpu_id',default="0", help='gpu_id', type=str)
    parser.add_argument('--Lambda',help='lambda',type=float)
    parser.add_argument('--pre_fuse', help='pre_fuse', type=bool)
    args = parser.parse_args()

    return args

def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers
    if args.dataDir:
        config.DATA_DIR = args.dataDir
    if args.modelDir:
        config.MODEL_DIR = args.modelDir
    if args.logDir:
        config.LOG_DIR = args.logDir
    if args.verbose:
        config.VERBOSE = args.verbose
    if args.Lambda:
        config.LOSS.LAMBDA = args.Lambda
    if args.pre_fuse:
        config.DTG_SPL.QUERY_EMBEDDER.PARAMS.pre_fuse = args.pre_fuse
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

if __name__ == '__main__':

    #config update
    args = parse_args()
    reset_config(config, args)
    logger, final_output_dir = create_logger(config, args.cfg)
    logger.info('\n'+pprint.pformat(args))
    logger.info('\n'+pprint.pformat(config))
    model_save = os.path.basename(args.cfg).split('.yaml')[0]+ '_'+ time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    # model build
    model = DMRNet(config.DTG_SPL)
    if config.MODEL.CHECKPOINT and config.TRAIN.CONTINUE:
        model_checkpoint = torch.load(config.MODEL.CHECKPOINT)
        model.load_state_dict(model_checkpoint)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model)
    device = ("cuda" if torch.cuda.is_available() else "cpu" )
    model = model.to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(),lr=config.TRAIN.LR, betas=(0.9, 0.999), weight_decay=config.TRAIN.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.TRAIN.STEPS,gamma=0.5)

    # dataloader build
    dataset_name = config.DATASET.NAME
    train_dataset = Charades('train')
    if config.TEST.EVAL_TRAIN:
        eval_train_dataset = Charades('train')
    if not config.DATASET.NO_VAL:
        val_dataset = Charades('val')
    multi_dataset = Charades_multi('test')
    test_dataset = Charades('test')

    def iterator(split):
        if split == 'train':
            dataloader = DataLoader(train_dataset,
                                    batch_size=config.TRAIN.BATCH_SIZE,
                                    shuffle=config.TRAIN.SHUFFLE,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=collate_fn)
        elif split == 'val':
            dataloader = DataLoader(val_dataset,
                                    batch_size=config.TEST.BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=datasets.collate_fn)
        elif split == 'test':
            dataloader = DataLoader(test_dataset,
                                    batch_size=config.TEST.BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=collate_fn)
        elif split == 'train_no_shuffle':
            dataloader = DataLoader(eval_train_dataset,
                                    batch_size=config.TEST.BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=collate_fn)
        elif split == 'multi':
            dataloader = DataLoader(multi_dataset,
                                    batch_size=config.TEST.BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=collate_fn_multi)
        else:
            raise NotImplementedError

        return dataloader

    def network(sample,state):
        textual_input = sample['batch_word_vectors'].cuda()
        textual_mask = sample['batch_txt_mask'].cuda()
        visual_input = sample['batch_vis_input'].cuda()
        vis_mask = sample['batch_vis_mask'].cuda()
        duration = sample['batch_duration']
        batch_anno_idxs = sample['batch_anno_idxs']
        batch_gt_time = sample['batch_gt_time'].cuda()
        loss_value = None

        if not model.training and state['split'] == 'train':
            anchors,semantic_scores,match_scores = model.compute_similarity(textual_input, textual_mask, visual_input,vis_mask,batch_gt_time,test_dataset.word_vectors)
            positive_moment_estimation(anchors,duration,semantic_scores,match_scores,batch_anno_idxs,state['epoch'])
            return loss_value, None

        prediction, mid_out = model(textual_input, textual_mask, visual_input,vis_mask,batch_gt_time,state['epoch'])
        if state['split'] != 'multi':
            loss_value = reg_loss(prediction,mid_out,sample)

        sorted_times = None if model.training else get_diverse_results(prediction,duration)

        return loss_value, sorted_times

    def positive_moment_estimation(anchors,durations,semantic_scores,match_scores,anno_idxs,epoch):
        anchors =  torch.clamp(anchors,0.0,1.0).squeeze(0)
        t = config.TEST.T_THRESH
        early_stop = config.DTG_SPL.MATCHING.PARAMS.early_stop
        for duration,semantic_score,match_score,anno_idx in zip(durations,semantic_scores,match_scores,anno_idxs):
            if epoch <=early_stop:
                train_dataset.annotations[anno_idx]['match_scores'] = match_score.tolist()
            else:
                match_score = train_dataset.annotations[anno_idx]['match_scores']
                match_score = torch.tensor(match_score).float().cuda()
            anchor = anchors * duration
            sorted_index = torch.argsort(match_score,descending = True)
            anchor = torch.stack([anchor[i] for i in sorted_index])
            match_score = torch.stack([match_score[i] for i in sorted_index])
            semantic_score = torch.stack([semantic_score[i] for i in sorted_index])
            anchor = torch.cat((anchor,match_score.unsqueeze(1),semantic_score.unsqueeze(1)),dim=1)
            anchor = anchor.cpu().detach().numpy()

            gt_time = train_dataset.annotations[anno_idx]['times']
            multi_moment = anchor[(anchor[:,3]>t)|(anchor[:,2]>t)]
            multi_moment = np.insert(multi_moment,0,[gt_time[0],gt_time[1],1.0,1.0],axis = 0).astype(float)
            multi_moment = soft_nms(multi_moment, thresh=config.TEST.NMS_THRESH, top_k=5, method=config.TEST.NMS_TYPE)
            # multi_moment = np.insert(multi_moment,0,[gt_time[0],gt_time[1],1.0,1.0],axis = 0).astype(float)
            train_dataset.annotations[anno_idx]['multi_moment'] = multi_moment[:,:2].tolist()

    def get_diverse_results(prediction,durations):
        out_sorted_times = []
        prediction_se,_ = prediction
        for score, duration in zip(prediction_se, durations):
            score =  torch.clamp(score,0.0,1.0)
            score = score * duration
            score_reg = score.cpu().detach().numpy()
            score_reg = np.array([item for item in score_reg if item[0] < item[1]]).astype(float)
            out_sorted_times.append((score_reg.tolist()))

        return out_sorted_times

    def on_start(state):
        state['loss_meter'] = AverageMeter()
        state['test_interval'] = math.ceil(len(train_dataset)/config.TEST.BATCH_SIZE*config.TEST.INTERVAL)
        state['t'] = 1
        model.train()
        if config.VERBOSE:
            state['progress_bar'] = tqdm(total=state['test_interval'])

    def on_forward(state):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        state['loss_meter'].update(state['loss'].item(), 1)

    def on_update(state):# Save All
        if config.VERBOSE:
            state['progress_bar'].update(1)

        if state['t'] % state['test_interval'] == 0:
            model.eval()
            if config.VERBOSE:
                state['progress_bar'].close()

            state['scheduler'].step()
            table_message = ''
            loss_message = ''
            if config.TEST.EVAL_TRAIN:
                train_state = engine.test(network, iterator('train_no_shuffle'), 'train',state['epoch'])

            if not config.DATASET.NO_VAL:
                val_state = engine.test(network, iterator('val'), 'val',state['epoch'])
                loss_message += ' val loss {:.4f}'.format(val_state['loss_meter'].avg)
                val_state['loss_meter'].reset()
                val_table = display_results(val_state['Rank@N,mIoU@M'], val_state['miou'],
                                            'performance on validation set')
                table_message += '\n'+ val_table

            test_state = engine.test(network, iterator('test'), 'test',state['epoch'])
            test_table = display_results(test_state['Rank@N,mIoU@M'], test_state['miou'],
                                         'performance on testing set')
            table_message += '\n' + test_table

            multi_state = engine.test(network,iterator('multi'), 'multi',state['epoch'])
            multi_table = display_results_new(multi_state['MULTI_LABEL'],'performance on multi-testing set')
            table_message += '\n'+multi_table

            loss_message = '\nepoch: {} iter: {}, lr: {:.6f}, train loss {:.4f}, test loss {:.4f}'.format( \
                state['epoch'], state['t'], state['optimizer'].param_groups[0]["lr"], state['loss_meter'].avg, test_state['loss_meter'].avg)

            message = loss_message+table_message+'\n'
            logger.info(message)

            saved_model_filename = os.path.join(config.MODEL_DIR,'{}/{}/epoch{:03d}.pkl'.format(
                dataset_name, model_save,state['epoch']))

            rootfolder1 = os.path.dirname(saved_model_filename)
            rootfolder2 = os.path.dirname(rootfolder1)
            rootfolder3 = os.path.dirname(rootfolder2)
            if not os.path.exists(rootfolder3):
                print('Make directory %s ...' % rootfolder3)
                os.mkdir(rootfolder3)
            if not os.path.exists(rootfolder2):
                print('Make directory %s ...' % rootfolder2)
                os.mkdir(rootfolder2)
            if not os.path.exists(rootfolder1):
                print('Make directory %s ...' % rootfolder1)
                os.mkdir(rootfolder1)

            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), saved_model_filename)
            else:
                torch.save(model.state_dict(), saved_model_filename)

            if config.VERBOSE:
                state['progress_bar'] = tqdm(total=state['test_interval'])
            model.train()

            test_state['loss_meter'].reset()
            state['loss_meter'].reset()


    def on_end(state):
        if config.VERBOSE:
            state['progress_bar'].close()

    def on_test_start(state):
        state['loss_meter'] = AverageMeter()
        state['sorted_segments_list'] = []
        if config.VERBOSE:
            if state['split'] == 'train':
                state['progress_bar'] = tqdm(total=math.ceil(len(train_dataset)/config.TEST.BATCH_SIZE))
            elif state['split'] == 'val':
                state['progress_bar'] = tqdm(total=math.ceil(len(val_dataset)/config.TEST.BATCH_SIZE))
            elif state['split'] == 'test':
                state['progress_bar'] = tqdm(total=math.ceil(len(test_dataset)/config.TEST.BATCH_SIZE))
            elif state['split'] == 'multi':
                state['progress_bar'] = tqdm(total=math.ceil(len(multi_dataset)/config.TEST.BATCH_SIZE))
            else:
                raise NotImplementedError

    def on_test_forward(state):
        if config.VERBOSE:
            state['progress_bar'].update(1)
        if state['split'] != 'train':
            if state['split'] != 'multi':
                state['loss_meter'].update(state['loss'].item(), 1)
            min_idx = min(state['sample']['batch_anno_idxs'])
            batch_indexs = [idx - min_idx for idx in state['sample']['batch_anno_idxs']]
            sorted_segments = [state['output'][i] for i in batch_indexs]
            state['sorted_segments_list'].extend(sorted_segments)

    def on_test_end(state):
        annotations = state['iterator'].dataset.annotations
        if state['split'] == 'multi':
            state['MULTI_LABEL'],_ = evals_new(state['sorted_segments_list'],annotations)
        elif state['split'] == 'test':
            state['Rank@N,mIoU@M'], state['miou'] = evals(state['sorted_segments_list'], annotations)
        if config.VERBOSE:
            state['progress_bar'].close()

    engine = Engine()
    engine.hooks['on_start'] = on_start
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_update'] = on_update
    engine.hooks['on_end'] = on_end
    engine.hooks['on_test_start'] = on_test_start
    engine.hooks['on_test_forward'] = on_test_forward
    engine.hooks['on_test_end'] = on_test_end
    engine.train(network,
                 iterator('train'),
                 maxepoch=config.TRAIN.MAX_EPOCH,
                 optimizer=optimizer,
                 scheduler=scheduler)