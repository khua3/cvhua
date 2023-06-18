import argparse
import os
import math
import random
import uuid
import json
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import sys
from winner_spatial.data.dataset import ConsolidateDatasets, ReconstructDataset, make_batch_iterator
from winner_spatial.utils.path import package_path
from winner_spatial.logging.configuration import configure_experiment, get_logger
from winner_spatial.utils.flags import stringify_flags, init_with_flags_file, save_flags
from winner_spatial.utils.checkpoint import save_experiment

from winner_spatial.net.experiment_logger import ExperimentLogger
from winner_spatial.analysis.cky import ParsePredictor as CKY
from winner_spatial.analysis.diora_tree import TreesFromDiora
from winner_spatial.analysis.utils import *
import torch.nn.functional as F

data_types_choices = ('coco', 'flickr', 'vidstg')

def count_params(net):
    return sum([x.numel() for x in net.parameters() if x.requires_grad])

def build_net(options, embeddings):
    from cliora.net.trainer import build_net

    trainer = build_net(options, embeddings, random_seed=options.seed)
    logger = get_logger()
    logger.info('# of params = {}'.format(count_params(trainer.net)))
    return trainer

def generate_seeds(n, seed=11):
    random.seed(seed)
    seeds = [random.randint(0, 2**16) for _ in range(n)]
    return seeds

def run_train(options, train_iterator, trainer, validation_iterator):
    logger = get_logger()
    experiment_logger = ExperimentLogger()
    save_emb = options.emb == 'none'

    logger.info('Running train.')

    seeds = generate_seeds(options.max_epoch, options.seed)
    word2idx = train_iterator.word2idx
    idx2word = {v: k for k, v in word2idx.items()}

    matrix_for_cvpr = np.zeros([2,151])
    step = 0
    
    for epoch, seed in zip(range(options.max_epoch), seeds):
        logger.info('epoch={} seed={}'.format(epoch, seed))
        
        def myiterator():
            it = train_iterator.get_iterator(random_seed=seed)
            count = 0
            for batch_map in it:
                # TODO: Skip short examples (optionally).
                if batch_map['length'] <= 2:
                    continue
                yield count, batch_map
                count += 1

        for batch_idx, batch_map in myiterator():
            result = trainer.step(batch_map, idx2word)
            experiment_logger.record(result)
            if step % options.log_every_batch == 0:
                experiment_logger.log_batch(epoch, step, batch_idx, batch_size=options.batch_size)
            del result
            step += 1
  
        experiment_logger.log_epoch(epoch, step)
        

        if epoch%10==0:
            matrix_for_cvpr = run_eval(options, trainer, validation_iterator, matrix_for_cvpr, epoch)
        

        if options.max_step is not None and epoch == options.max_step-1:
            data = pd.DataFrame(matrix_for_cvpr)
            # writer = pd.ExcelWriter("score.xlsx")
            data.to_excel("score.xlsx", index=False)
            # writer.save()
            # writer.close()
            logger.info('Max-Step={} Quitting.'.format(options.max_step))
            sys.exit()



def get_candidate_score(gt_boxes, pred_boxes, reduction="sum"):

    x1 = np.maximum(gt_boxes[:, 0], pred_boxes[:, 0])
    y1 = np.maximum(gt_boxes[:, 1], pred_boxes[:, 1])
    x2 = np.minimum(gt_boxes[:, 2], pred_boxes[:, 2])
    y2 = np.minimum(gt_boxes[:, 3], pred_boxes[:, 3])

    inter = np.maximum(0, x2-x1+1.0) * np.maximum(0, y2-y1+1.0)
    S_a = (gt_boxes[:, 2]-gt_boxes[:, 0]+1.0)*(gt_boxes[:, 3]-gt_boxes[:, 1]+1.0)
    S_b = (pred_boxes[:, 2]-pred_boxes[:, 0]+1.0)*(pred_boxes[:, 3]-pred_boxes[:, 1]+1.0)
    union = S_a + S_b - inter
    IoU = inter/union

    IoU = IoU.reshape(-1)
    if reduction == 'sum':
        candidate_iou = np.sum(IoU, axis=0)
    elif reduction == 'mean':
        candidate_iou = np.average(IoU, axis=0)
    return candidate_iou


def run_eval(options, trainer, validation_iterator, matrix_for_cvpr, epoch):
  
    logger = get_logger()

    # Eval mode.
    trainer.net.eval()
    if options.multigpu:
        diora = trainer.net.module.diora
    else:
        diora = trainer.net.diora
        override_init_with_batch(diora)
        override_inside_hook(diora)
        parse_predictor = CKY(diora)

    # cliora.outside = False
    # cliora.outside = True # TODO
    diora.outside = options.obj_feats
    val_batches = validation_iterator.get_iterator(random_seed=options.seed)

    logger.info('####### Beginning Eval #######')

    total_num = 0;recall_num_05 = 0;recall_num_03 = 0;iou_sum = 0
    spa_num_05 = 0;spa_num_03 = 0;spa_iou_sum = 0
    
    subject_score = 0
    pair_score = 0
    count_number = 0
    
    with torch.no_grad():
        for i, batch_map in enumerate(val_batches):
            sentences = batch_map['sentences']
            length = sentences.shape[1]
            batch_size = sentences.shape[0]
            # Skip very short sentences.
            if length <= 2:
                continue

            _ = trainer.step(batch_map, train=False, compute_loss=False)

            trees = parse_predictor.parse_batch(batch_map)
            # for bid, tr in enumerate(trees):
            #     print(tr)
            #     pred_actions = get_actions(str(tr))
            #     print(pred_actions)
            #     pred_spans = set(get_spans(pred_actions)[:-1])
            #     exit()
            #       pr#int(pred_spans)
            # Grounding eval

            if diora.atten_score is not None:
                attenion_scores = diora.atten_score.cpu()

                pred_boxes = batch_map["boxes"]
                gt_boxes = batch_map['gt_boxes']
                video_ids = batch_map['video_ids']
                target_tokens = batch_map['target_tokens']
                # pair_tokens = batch_map["pair_tokens"]
                pred_proposals = np.array(batch_map['pred_proposal'])
                gt_proposals = np.array(batch_map['gt_proposal'])
                gt_tubes_index = batch_map['gt_tubes_index']
                # noun_indexs_for_cal = batch_map['noun_index_for_cal']
            
                starts = np.maximum(0, (pred_proposals[:,0]-gt_proposals[:,0])/(gt_proposals[:,1]-gt_proposals[:,0]))
                ends = np.minimum(1, (pred_proposals[:,1]-gt_proposals[:,0])/(gt_proposals[:,1]-gt_proposals[:,0]))

                for bid in range(batch_size):
                    
                    # noun_index_for_cal = noun_indexs_for_cal[bid]
                    gt_tube_index = gt_tubes_index[bid] 
                    pred_box = pred_boxes[bid]
                    pred_box_all = pred_boxes[bid]
                    gt_box = gt_boxes[bid]
                    gt_box_all = gt_boxes[bid]
                
                    attenion_scores_bid = attenion_scores[bid]
                    target_token_index = target_tokens[bid]
                    
                    # pair_token_index = pair_tokens[bid]
                    select_scores, select_box_ids = attenion_scores_bid.max(1)
                    
                    # if pair_token_index!=-1:
                    #     # cal_att_scores = attenion_scores_bid[noun_index_for_cal]
                    #     attenion_scores_bid = F.softmax(attenion_scores_bid, dim=0)
                    #     subject_score += attenion_scores_bid[target_token_index][gt_tube_index]
                    #     pair_score += attenion_scores_bid[pair_token_index][gt_tube_index]
                    #     count_number += 1

                    # pred_box = pred_box[select_box_ids]

                    # select_id = select_scores.max(0)[1]
                    select_id = select_box_ids[target_token_index]
                    
                    # if pair_token_index!=-1:
                    #     target_tube_score = select_scores[target_token_index]
                    #     pair_tube_score = select_scores[pair_token_index]
                    #     logger.info("{}: target_tube_score {.3f}, pair_tube_score {.3f}".format(video_ids,target_tube_score, pair_tube_score))
                    # print(select_id,end='')
                    # 
                    pred_box = pred_box[select_id]
                    
                    len = gt_box.shape[0]
                    assert gt_box.shape[0] == pred_box.shape[0]

                    score = get_candidate_score(gt_box, pred_box,'mean')
                    
                    if score > 0.5:
                        spa_num_05 += 1
                        spa_num_03 += 1
                    elif score > 0.3:
                        spa_num_03 += 1
                    spa_iou_sum += score

                    inter_start_frame = round(starts[bid]*len)
                    inter_end_frame = math.ceil(ends[bid]*len)
                    gt_box_cut = gt_box[inter_start_frame:inter_end_frame+1, :]
                    pred_box_cut = pred_box[inter_start_frame:inter_end_frame+1, :] 

                    union_start = min(gt_proposals[bid, 0], pred_proposals[bid, 0])
                    union_end = max(gt_proposals[bid, 1], pred_proposals[bid, 1])
                    
                    length = round((union_end-union_start)/(gt_proposals[bid,1]-gt_proposals[bid,0])*len)

                    assert gt_box_cut.shape[0] == pred_box_cut.shape[0]
                    score = get_candidate_score(gt_box_cut, pred_box_cut, 'sum') / length
                    
                    if score > 0.5:
                        recall_num_05 += 1
                        recall_num_03 += 1
                    elif score > 0.3:
                        recall_num_03 += 1
                    iou_sum += score
                    total_num += 1

                    # if score > 0.2 and score < 0.5:
                    #     tree_bid = trees[bid]
                    #     video_id = video_ids[bid]
                    #     logger.info(tree_bid)
                    #     logger.info(video_id)
                    #     logger.info("pred time: {:.4f}->{:.4f}".format(pred_proposals[bid,0],pred_proposals[bid,1]))
                    #     logger.info("real time: {:.4f}->{:.4f}".format(gt_proposals[bid,0],gt_proposals[bid,1]))
                    #     max_index = torch.argmax(attenion_scores_bid, dim=1)
                    #     logger.info(max_index)
                    #     logger.info(pred_box_all)
                    #     logger.info(gt_box_all)
                    #     logger.info("======================")
                    #     logger.info("======================")
    
    # comb_feat = diora.comb_feat_list
    # tube_feat = diora.target_feat_list

    ground_acc_05 = recall_num_05 / (total_num + 1e-8)
    ground_acc_03 = recall_num_03 / (total_num + 1e-8)
    avg_acc = iou_sum / (total_num+1e-8)
    logger.info('Both part: mIoU0.5 grounding acc:{:.3f}% || mIoU0.3 grounding acc:{:.3f}% || average iou acc: {:.3f}%'.format(
        ground_acc_05*100, ground_acc_03*100, avg_acc*100))
    
    ground_acc_05 = spa_num_05 / (total_num + 1e-8)
    ground_acc_03 = spa_num_03 / (total_num + 1e-8)
    avg_acc = spa_iou_sum / (total_num+1e-8)
    logger.info('Both part: mIoU0.5 grounding acc:{:.3f}% || mIoU0.3 grounding acc:{:.3f}% || average iou acc: {:.3f}%'.format(
        ground_acc_05*100, ground_acc_03*100, avg_acc*100))
    
    diora.outside = True
    trainer.net.train()
    # matrix_for_cvpr[0][epoch] = subject_score/(count_number+1e-5)
    # matrix_for_cvpr[1][epoch] = pair_score/(count_number+1e-5)
    return matrix_for_cvpr

def get_train_dataset(options):
    return ReconstructDataset().initialize(options, text_path=options.train_path,
        embeddings_path=options.embeddings_path, filter_length=options.train_filter_length,
        data_type=options.train_data_type)

def get_train_iterator(options, dataset):
    return make_batch_iterator(options, dataset, mode='train', shuffle=True,
            include_partial=False, filter_length=options.train_filter_length,
            batch_size=options.batch_size, length_to_size=options.length_to_size)

def get_validation_dataset(options):
    return ReconstructDataset().initialize(options, text_path=options.validation_path,
            embeddings_path=options.embeddings_path, filter_length=options.validation_filter_length,
            data_type=options.validation_data_type)


def get_validation_iterator(options, dataset):
    return make_batch_iterator(options, dataset, mode='test', shuffle=False,
            include_partial=True, filter_length=options.validation_filter_length,
            batch_size=options.validation_batch_size, length_to_size=options.length_to_size)

def get_train_and_validation(options):
    train_dataset = get_train_dataset(options)
    validation_dataset = get_validation_dataset(options)

    # Modifies datasets. Unifying word mappings, embeddings, etc.
    if options.data_type not in ['coco', 'flickr','vidstg']:
        ConsolidateDatasets([train_dataset, validation_dataset]).run()

    return train_dataset, validation_dataset


def run(options):
    logger = get_logger()

    train_dataset, validation_dataset = get_train_and_validation(options)
    if options.debug:
        train_iterator = get_validation_iterator(options, validation_dataset)
    else:
        train_iterator = get_train_iterator(options, train_dataset)
    validation_iterator = get_validation_iterator(options, validation_dataset)
    embeddings = train_dataset['embeddings']

    logger.info('Initializing model.')
    trainer = build_net(options, embeddings)
    logger.info('Model:')
    for name, p in trainer.net.named_parameters():
        logger.info('{} {} {}'.format(name, p.shape, p.requires_grad))

    run_train(options, train_iterator, trainer, validation_iterator)


def argument_parser():
    parser = argparse.ArgumentParser()

    # Debug.
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', default=12, type=int)
    parser.add_argument('--git_sha', default=None, type=str)
    parser.add_argument('--git_branch_name', default=None, type=str)
    parser.add_argument('--git_dirty', default=None, type=str)
    parser.add_argument('--uuid', default=None, type=str)
    parser.add_argument('--model_flags', default=None, type=str,
                        help='Load model settings from a flags file.')
    parser.add_argument('--flags', default=None, type=str,
                        help='Load any settings from a flags file.')

    parser.add_argument('--master_addr', default='127.0.0.1', type=str)
    parser.add_argument('--master_port', default='29500', type=str)
    parser.add_argument('--world_size', default=None, type=int)

    # Pytorch
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--multigpu', action='store_true')
    parser.add_argument("--local_rank", default=None, type=int) # for distributed-data-parallel

    # Logging.
    parser.add_argument('--default_experiment_directory', default=os.path.join(package_path(), '..', 'log'), type=str)
    parser.add_argument('--experiment_name', default=None, type=str)
    parser.add_argument('--experiment_path', default=None, type=str)
    parser.add_argument('--log_every_batch', default=10, type=int)
    parser.add_argument('--save_latest', default=1000, type=int)
    parser.add_argument('--save_distinct', default=5000, type=int)
    parser.add_argument('--save_after', default=1000, type=int)

    # Loading.
    parser.add_argument('--load_model_path', default="./outputs/flickr/flickr_diora_5e4_mlpshare_bs32_RandInit_seed1234/model.epoch_29.pt", type=str)

    # Data.
    parser.add_argument('--data_type', default='nli', choices=data_types_choices)
    parser.add_argument('--train_data_type', default=None, choices=data_types_choices)
    parser.add_argument('--validation_data_type', default=None, choices=data_types_choices)
    parser.add_argument('--train_path', default=os.path.expanduser('~/data/snli_1.0/snli_1.0_train.jsonl'), type=str)
    parser.add_argument('--validation_path', default=os.path.expanduser('~/data/snli_1.0/snli_1.0_dev.jsonl'), type=str)
    parser.add_argument('--embeddings_path', default=os.path.expanduser('~/data/glove/glove.6B.300d.txt'), type=str)

    # Data (synthetic).
    parser.add_argument('--synthetic-nexamples', default=1000, type=int)
    parser.add_argument('--synthetic-vocabsize', default=1000, type=int)
    parser.add_argument('--synthetic-embeddingsize', default=1024, type=int)
    parser.add_argument('--synthetic-minlen', default=20, type=int)
    parser.add_argument('--synthetic-maxlen', default=21, type=int)
    parser.add_argument('--synthetic-seed', default=11, type=int)
    parser.add_argument('--synthetic-length', default=None, type=int)
    parser.add_argument('--use-synthetic-embeddings', action='store_true')

    # Data (preprocessing).
    parser.add_argument('--uppercase', action='store_true')
    parser.add_argument('--train_filter_length', default=50, type=int)
    parser.add_argument('--validation_filter_length', default=0, type=int)

    # Model.
    parser.add_argument('--arch', default='mlp', choices=('mlp', 'hard'))
    parser.add_argument('--share', action='store_false')
    parser.add_argument('--hidden_dim', default=400, type=int)
    parser.add_argument('--normalize', default='unit', choices=('none', 'unit'))
    parser.add_argument('--compress', action='store_true',
                        help='If true, then copy root from inside chart for outside. ' + \
                             'Otherwise, learn outside root as bias.')

    # Model (Objective).
    parser.add_argument('--reconstruct_mode', default='softmax',
                        choices=('softmax'))

    # Model (Embeddings).
    parser.add_argument('--emb', default='w2v', choices=('w2v', 'skip', 'elmo', 'both', 'none'))

    # Model (Negative Sampler).
    parser.add_argument('--margin', default=1, type=float)
    parser.add_argument('--k_neg', default=100, type=int)
    parser.add_argument('--freq_dist_power', default=0.75, type=float)

    # ELMo
    parser.add_argument('--elmo_options_path', default='https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json', type=str)
    parser.add_argument('--elmo_weights_path', default='https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5', type=str)
    parser.add_argument('--elmo_cache_dir', default='./log/elmo', type=str,
                        help='If set, then context-insensitive word embeddings will be cached ' + \
                             '(identified by a hash of the vocabulary).')

    # Training.
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--length_to_size', default=None, type=str,
                        help='Easily specify a mapping of length to batch_size.' + \
                             'For instance, 10:32,20:16 means that all batches' + \
                             'of length 10-19 will have batch size 32, 20 or greater' + \
                             'will have batch size 16, and less than 10 will have batch size' + \
                             'equal to the batch_size arg. Only applies to training.')
    parser.add_argument('--train_dataset_size', default=None, type=int)
    parser.add_argument('--validation_dataset_size', default=None, type=int)
    parser.add_argument('--validation_batch_size', default=None, type=int)
    parser.add_argument('--max_epoch', default=5, type=int)
    parser.add_argument('--max_step', default=None, type=int)
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--finetune_after', default=0, type=int)

    # Parsing.
    parser.add_argument('--postprocess', action='store_true')
    parser.add_argument('--visualize', action='store_true')

    # Optimization.
    parser.add_argument('--lr', default=2e-3, type=float)

    # Vis feature
    parser.add_argument('--alpha_contr', type=float, default=1.0)
    parser.add_argument('--obj_feats', action='store_true')
    parser.add_argument('--vl_margin', default=0.2, type=float)
    parser.add_argument('--use_contr', action='store_true')
    parser.add_argument('--use_contr_ce', action='store_true')
    parser.add_argument('--vg_loss', action='store_true')
    parser.add_argument('--alpha_vg', type=float, default=1.0)
    parser.add_argument('--alpha_kl', type=float, default=1.0)

    # S-DIORA
    parser.add_argument('--hinge_margin', default=1, type=float)

    return parser


def parse_args(parser):
    options, other_args = parser.parse_known_args()

    # Set default flag values (data).
    options.train_data_type = options.data_type if options.train_data_type is None else options.train_data_type
    options.validation_data_type = options.data_type if options.validation_data_type is None else options.validation_data_type
    options.validation_batch_size = options.batch_size if options.validation_batch_size is None else options.validation_batch_size

    # Set default flag values (config).
    if not options.git_branch_name:
        # print("1,")
        options.git_branch_name = os.popen(
            'git rev-parse --abbrev-ref HEAD').read().strip()

    if not options.git_sha:
        # print("2,")
        options.git_sha = os.popen('git rev-parse HEAD').read().strip()

    if not options.git_dirty:
        # print("3,")
        options.git_dirty = os.popen("git diff --quiet && echo 'clean' || echo 'dirty'").read().strip()

    if not options.uuid:
        # print("4,")
        options.uuid = str(uuid.uuid4())

    # 这是什么东西
    if not options.experiment_name:
        # print("5,")
        options.experiment_name = '{}'.format(options.uuid[:8])

    if not options.experiment_path:
        # print("6,")
        options.experiment_path = os.path.join(options.default_experiment_directory, options.experiment_name)

    if options.length_to_size is not None:
        # print("7,")
        parts = [x.split(':') for x in options.length_to_size.split(',')]
        options.length_to_size = {int(x[0]): int(x[1]) for x in parts}

    options.lowercase = not options.uppercase

    for k, v in options.__dict__.items():
        if type(v) == str and v.startswith('~'):
            options.__dict__[k] = os.path.expanduser(v)

    # Load model settings from a flags file.
    if options.model_flags is not None:
        # print("8,")
        flags_to_use = []
        flags_to_use += ['arch']
        flags_to_use += ['compress']
        flags_to_use += ['emb']
        flags_to_use += ['hidden_dim']
        flags_to_use += ['normalize']
        flags_to_use += ['reconstruct_mode']

        options = init_with_flags_file(options, options.model_flags, flags_to_use)

    # Load any setting from a flags file.
    if options.flags is not None:
        # print("9,")
        options = init_with_flags_file(options, options.flags)
    
    return options


def configure(options):
    # Configure output paths for this experiment.
    configure_experiment(options.experiment_path, rank=options.local_rank)

    # Get logger.
    logger = get_logger()

    # Print flags.
    logger.info(stringify_flags(options))
    save_flags(options, options.experiment_path)


if __name__ == '__main__':
    parser = argument_parser()
    options = parse_args(parser)
    configure(options)
    run(options)
