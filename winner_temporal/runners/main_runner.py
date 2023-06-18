import os
import torch
import random
import logging
import collections
import numpy as np
from fairseq.utils import move_to_cuda
from utils import AverageMeter, TimeMeter

class MainRunner:
    def __init__(self, args):
        self.args = args
        # self.local_rank = local_rank
        self._build_dataset()
        self._build_model()
        if 'train' in args:
            self._build_optimizer()
            self.num_updates = 0
        # self.finetune = args["model"]["Scorer"]["finetune"]
        self.num_clips = args['dataset']['max_num_frames'] // args['dataset']['target_stride']
        # self.props = self.test_set.props

    def train(self):
        self._load_model(self.args["train"]["load_path"])
        for bias in [0.0]:
            # self._temporal_proposals(dataset=self.train_set, dataloader=self.train_loader, splits="train", bias=bias)
            # self._temporal_proposals(dataset=self.test_set, dataloader=self.test_loader, splits="test", bias=bias)
            # self._temporal_proposals(dataset=self.val_set, dataloader=self.val_loader, splits="val", bias=bias)
            # exit()

            logging.info('bias = {}.'.format(bias))
            for epoch in range(0, self.args['train']["max_num_epochs"]):
                logging.info('Start Epoch {}'.format(epoch))
                self.model_saved_path = self.args['train']['model_saved_path']
                os.makedirs(self.model_saved_path, mode=0o755, exist_ok=True)
                save_path = os.path.join(self.model_saved_path, 'model-{}.pt'.format(epoch))        
                
                self._train_one_epoch(epoch, bias=bias)
                # if epoch % 5 ==0 and epoch != 0:
                self._save_model(save_path) 
                self.eval(dataset=self.test_set, dataloader=self.test_loader, bias=bias)
                # self.eval(dataset=self.test_set, dataloader=self.test_loader, bias=bias, top_n=5, thresh=0.45)

                logging.info('=' * 60)

            print('-' * 120)
        logging.info('Done.')

    def _train_one_epoch(self, epoch, **kwargs):
        self.model.train()
        def print_log():
            msg = 'Epoch {}, Batch {}, lr = {:.7f}, '.format(epoch, bid, curr_lr)
            for k, v in loss_meter.items():
                msg += '{} = {:.4f}, '.format(k, v.avg)
                v.reset()
            msg += '{:.3f} seconds/batch'.format(1.0 / time_meter.avg)
            logging.info(msg)

        from models.weakly.loss import weakly_supervised_loss, supervised_loss
        display_n_batches, bid = 25, 0
        time_meter = TimeMeter()
        loss_meter = collections.defaultdict(lambda: AverageMeter())

        if self.args['dataset']['dataset'] == 'ActivityNet':
            num_cands = 153
            fp = open('append.log', encoding='utf8', mode='a')
        elif self.args['dataset']['dataset'] == 'CharadesSTA':
            num_cands = 200
            fp = open('append2.log', encoding='utf8', mode='a')
        elif self.args['dataset']['dataset'] == 'VidSTG':
            num_cands = 214
            fp = open('append3.log', encoding='utf8', mode='a')
        elif self.args['dataset']['dataset'] == 'HC_STVG':
            num_cands = 151
            fp = open('append4.log', encoding='utf8', mode='a')

        # 31self.args['train']['topK'] = 48
        # if self.num_updates < 50:
        #     self.args['train']['topK'] = 195
        # else:
        #     self.args['train']['topK'] = 48 

        torch.backends.cudnn.enabled = False
        for bid, batch in enumerate(self.train_loader, 1):
            self.optimizer.zero_grad()
            net_input = move_to_cuda(batch['net_input'])
            net_input['props'] = net_input['props'].expand(1, -1, -1)
            # net_input['props'] = net_input['props'].expand(len(self.device_ids), -1, -1)
            # net_input['props_graph'] = net_input['props_graph'].expand(len(self.device_ids), -1, -1)
            net_input['props_graph'] = net_input['props_graph'].expand(1, -1, -1)
            output = self.model(**net_input, get_negative=True, **kwargs)
            
            loss, _ = weakly_supervised_loss(pos_score=output['score'],
                                             neg_score1=output['inter_neg']['neg_score'],
                                             neg_score2=output['intra_neg']['neg_score'],
                                             neg_weight2=output['intra_neg']['weight'],
                                             weight_gt=net_input['frame_gt'],
                                             props=net_input['props'][0],
                                             log_fp=fp, num_cands=num_cands,
                                             loss_meter=loss_meter, 
                                             **self.args['train'])
            # if self.finetune:
            #     gt_box = torch.from_numpy(np.asarray([i[2] for i in batch['raw']])).cuda()
            #     durations = torch.from_numpy(np.asarray([i[1] for i in batch['raw']])).cuda()
            #     loss_supervise = supervised_loss(score=output['score'], map_gt = net_input['map_gt'],
            #                                     loss_meter = loss_meter, reduction="mean", delta=output['delta']
            #                                     ,probs=self.train_set.props_torch, gt_box=gt_box, durations=durations)
            #     loss = loss_supervise

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)

            self.optimizer.step()
            self.num_updates += 1
            curr_lr = self.lr_scheduler.step_update(self.num_updates)
            time_meter.update()
            loss_meter['loss'].update(loss.item())

            if bid % display_n_batches == 1:
                print_log()
           
        if bid % display_n_batches != 0:
            print_log()

        fp.write('=' * 60 + '\n')
        fp.flush()
        fp.close()

    def eval(self, dataset, dataloader, top_n=1, thresh=None, **kwargs):
        self.model.eval()
        metrics_logger = collections.defaultdict(lambda: AverageMeter())
        from models.weakly.loss import bce_rescale_loss
        logging.info('start eval')
        # if self.args['dataset']['dataset'] == 'ActivityNet':
        #     num_cands = 153
        # elif self.args['dataset']['dataset'] == 'CharadesSTA':
        #     num_cands = 200
        # elif self.args['dataset']['dataset'] == 'VidSTG':
        #     num_cands = 214
        # matrix_max = np.zeros([5])
       

        with torch.no_grad():
            for _, batch in enumerate(dataloader, 1):
                net_input = move_to_cuda(batch['net_input'])
                net_input['props'] = net_input['props'].expand(1, -1, -1)
                # net_input['props'] = net_input['props'].expand(len(self.device_ids), -1, -1)
                # net_input['props_graph'] = net_input['props_graph'].expand(len(self.device_ids), -1, -1)
                net_input['props_graph'] = net_input['props_graph'].expand(1, -1, -1)

                # forward
                output = self.model(**net_input, get_negative=True, **kwargs)


                durations = np.asarray([i[1] for i in batch['raw']])
                gt = np.asarray([i[2] for i in batch['raw']])

                loss, prob = bce_rescale_loss(output['score'], net_input['map_gt'])
                from models.weakly.loss import weakly_supervised_loss_fuck
                # neg, pos = weakly_supervised_loss_fuck(pos_score=output['score'],
                #                                        neg_score1=output['inter_neg']['neg_score'],
                #                                        neg_score2=output['intra_neg']['neg_score'],
                #                                        neg_weight2=output['intra_neg']['weight'],
                #                                        weight_gt=net_input['frame_gt'],
                #                                        props=net_input['props'][0],
                #                                        log_fp=None, num_cands=num_cands,
                #                                        loss_meter=None, **self.args['train'])
                metrics_logger['loss'].update(loss.item())
                bsz = prob.size(0)
                prob = np.reshape(prob.cpu().numpy(), [bsz, -1])
                
                idx = np.argmax(prob, -1)

                idx1 = np.argmax(np.reshape(output['intra_neg']['neg_score'].cpu().numpy(), [bsz, -1]), -1)
                
                
                selected_props = dataset.props[idx]  # [bsz, 2]
                # if self.finetune:
                
                #     delta = np.array(output['delta'].cpu().numpy())
                #     # print(delta.shape)

                #     selectes_delta = delta[range(bsz), idx]
                #     selected_props = selected_props + selectes_delta
                    
                # print(selected_props.shape)

                # neg_props = dataset.props[idx1]

                # weight = output['intra_neg']['weight'].cpu().numpy()[:, :, 0]
                if top_n > 1:
                    num_clips = self.num_clips
                    sort_idx = np.argsort(-prob, -1)

                    cand_props = list(dataset.props[sort_idx])  # [bsz, cand_props, 2]
                    
                    # 选出来的proposal
                    top_n_selected_props = [selected_props]

                    for it in range(1, top_n):
                        ptr_props = top_n_selected_props[-1]
                        selected_props = []
                        for i in range(bsz):
                            p2 = cand_props[i]
                            p1 = np.repeat(np.expand_dims(ptr_props[i], 0),
                                           p2.shape[0], 0)

                            iou = calculate_IoU_batch2((p1[:, 0], p1[:, 1]), (p2[:, 0], p2[:, 1]))
                            keep = iou <= thresh

                            cand_props[i] = cand_props[i][keep]

                            selected_props.append(cand_props[i][0])
                        top_n_selected_props[-1] = top_n_selected_props[-1] * durations[:, np.newaxis] / num_clips
                    
                        top_n_selected_props.append(np.asarray(selected_props))

                    top_n_selected_props[-1] = top_n_selected_props[-1] * durations[:, np.newaxis] / num_clips
                    top_n_selected_props = np.asarray(top_n_selected_props)
                   
                    res, iou_index = top_n_metric(top_n_selected_props, gt)
                    
                    # print(matrix_max, iou_index)
                    # for i in range(len(iou_index)):
                    #     matrix_max[iou_index[i]] += 1
                    # print(matrix_max)
                    # exit()
                else:
                    # ori_props = selected_props
             
                    selected_props = selected_props * durations[:, None] / self.num_clips

                    # neg_props = neg_props * durations[:, np.newaxis] / self.num_clips

                    res, iou = top_1_metric(selected_props, gt)
               
                    # neg_res, neg_iou = top_1_metric(neg_props, gt)

                for k, v in res.items():
                    metrics_logger[k].update(v, bsz)
        
        # print(matrix_max)
        outputs = ""
        for k, v in metrics_logger.items():
            outputs += '| {} {:.4f}'.format(k, v.avg)
        outputs += '|'
        # print('|')
        logging.info(outputs)
        self.model.train()
        return metrics_logger

    def _get_proposals_dataset(self, dataset, dataloader, splits="train", top_n=1, thresh=0.45, **kwargs):
        self.model.eval()
        metrics_logger = collections.defaultdict(lambda: AverageMeter())
        from models.weakly.loss import bce_rescale_loss
        
        proposal_start_top1 = []
        proposal_end_top1 = []
        gt_start_top1 = []
        gt_end_top1 = []
        video_hash_id_top1 = []
        sentence_top1 = []
        duration_top1 = []

        proposal_start_top5 = []
        proposal_end_top5 = []
        gt_start_top5 = []
        gt_end_top5 = []
        video_hash_id_top5 = []
        sentence_top5 = []
        duration_top5 = []

        with torch.no_grad():
            for _, batch in enumerate(dataloader, 1):
                if ( random.random() < 0.14 and splits == "train" ) or splits != 'train':
                    # 这里加一个10%的筛选
                    net_input = move_to_cuda(batch['net_input'])
                    net_input['props'] = net_input['props'].expand(1, -1, -1)
                    net_input['props_graph'] = net_input['props_graph'].expand(1, -1, -1)

                    # forward
                    output = self.model(**net_input, get_negative=True, **kwargs)
                    durations = np.asarray([i[1] for i in batch['raw']])
                    gt = np.asarray([i[2] for i in batch['raw']])
                    hash_id = list([i[0] for i in batch['raw']])
                    sentences = list(i[3] for i in batch['raw'])

                    loss, prob = bce_rescale_loss(output['score'], net_input['map_gt'])
                    metrics_logger['loss'].update(loss.item())

                    bsz = prob.size(0)
                    prob = np.reshape(prob.cpu().numpy(), [bsz, -1])
                    idx = np.argmax(prob, -1)
                    selected_props = dataset.props[idx]  # [bsz, 2]
                    top_n_selected_props = [selected_props]

                    if top_n > 1:
                        num_clips = self.num_clips
                        sort_idx = np.argsort(-prob, -1)
                        cand_props = list(dataset.props[sort_idx])  # [bsz, cand_props, 2]
                        
                        for _ in range(1, top_n):
                            ptr_props = top_n_selected_props[-1]
                            selected_props = []
                            for i in range(bsz):
                                p2 = cand_props[i]
                                p1 = np.repeat(np.expand_dims(ptr_props[i], 0),p2.shape[0], 0)
                                iou = calculate_IoU_batch2((p1[:, 0], p1[:, 1]), (p2[:, 0], p2[:, 1]))
                                keep = iou <= thresh
                                cand_props[i] = cand_props[i][keep]
                                selected_props.append(cand_props[i][0])
                            
                            top_n_selected_props[-1] = top_n_selected_props[-1] * durations[:, np.newaxis] / num_clips
                            top_n_selected_props.append(np.asarray(selected_props))

                        top_n_selected_props[-1] = top_n_selected_props[-1] * durations[:, np.newaxis] / num_clips
                        top_n_selected_props = np.asarray(top_n_selected_props)
                        res = top_n_metric(top_n_selected_props, gt)

                    for k, v in res.items():
                        metrics_logger[k].update(v, bsz)

                    
                    for j in range(bsz):
                        for i in range(top_n):
                            proposal_start_top5.append(top_n_selected_props[i][j][0])
                            proposal_end_top5.append(top_n_selected_props[i][j][1])
                            gt_start_top5.append(gt[j][0])
                            gt_end_top5.append(gt[j][1])
                            video_hash_id_top5.append(hash_id[j])
                            sentence_top5.append(sentences[j])
                            duration_top5.append(durations[j])
                    
                    for j in range(bsz):
                        proposal_start_top1.append(top_n_selected_props[0][j][0])
                        proposal_end_top1.append(top_n_selected_props[0][j][1])
                        gt_start_top1.append(gt[j][0])
                        gt_end_top1.append(gt[j][1])
                        video_hash_id_top1.append(hash_id[j])
                        sentence_top1.append(sentences[j])
                        duration_top1.append(durations[j])

        if splits == 'train':
            save_path = "./2D-TAN/{}/{}_rtbpn_top{}_10.npz".format(self.args["dataset"]["feature_type"],splits, top_n)
        else:
            save_path = "./2D-TAN/{}/{}_rtbpn_top{}.npz".format(self.args["dataset"]["feature_type"],splits, top_n)
        np.savez(save_path, 
            gt_start = gt_start_top5,
            gt_end = gt_end_top5,
            proposal_end = proposal_end_top5,
            proposal_start = proposal_start_top5,
            hash_id = video_hash_id_top5,
            sentence = sentence_top5,
            video_duration = duration_top5)
        print("save the proposal dataset {} successfully; len of dataset {}.".format(
            save_path.split("/")[-1], len(video_hash_id_top5)))

        if splits == 'train':
            save_path = "./2D-TAN/{}/{}_rtbpn_top{}_10.npz".format(self.args["dataset"]["feature_type"],splits, 1)
        else:
            save_path = "./2D-TAN/{}/{}_rtbpn_top{}.npz".format(self.args["dataset"]["feature_type"],splits, 1)
        np.savez(save_path, 
            gt_start = gt_start_top1,
            gt_end = gt_end_top1,
            proposal_end = proposal_end_top1,
            proposal_start = proposal_start_top1,
            hash_id = video_hash_id_top1,
            sentence = sentence_top1,
            video_duration = duration_top1)
        print("save the proposal dataset {} successfully; len of dataset {}.".format(
            save_path.split("/")[-1], len(video_hash_id_top1)))

        for k, v in metrics_logger.items():
            print('| {} {:.4f}'.format(k, v.avg), end=' ')
        print('|')
        self.model.train()
    
    def _temporal_proposals(self, dataset, dataloader, splits="test",**kwargs):
        print("start to precess {}".format(splits))
        self.model.eval()
        metrics_logger = collections.defaultdict(lambda: AverageMeter())
        from models.weakly.loss import bce_rescale_loss
        save_dir = "./VidSTG-Dataset/temporal_pro_question"
        video_index = 0

        with torch.no_grad():
            for _, batch in enumerate(dataloader, 1):
               
                net_input = move_to_cuda(batch['net_input'])
                net_input['props'] = net_input['props'].expand(1, -1, -1)
                net_input['props_graph'] = net_input['props_graph'].expand(1, -1, -1)

                # forward
                output = self.model(**net_input, get_negative=True, **kwargs)
                durations = np.asarray([i[1] for i in batch['raw']])
                gt = np.asarray([i[2] for i in batch['raw']])
                # hash_id = list([i[0] for i in batch['raw']])

                loss, prob = bce_rescale_loss(output['score'], net_input['map_gt'])
                metrics_logger['loss'].update(loss.item())

                bsz = prob.size(0)
                prob = np.reshape(prob.cpu().numpy(), [bsz, -1])
                idx = np.argsort(prob, axis=-1)[:, -1]
                
                selected_props = dataset.props[idx]  # [bsz, 2]
                
                selected_props = selected_props * durations[:, None] / self.num_clips
                res, iou = top_1_metric(selected_props, gt)

                for k, v in res.items():
                    metrics_logger[k].update(v, bsz)

                for j in range(bsz):
                    # print(gt[j], selected_props[j])
                    save_path = "{}/{}/{}.npz".format(save_dir, splits, video_index)
                    np.savez(save_path, gt_start = gt[j][0], gt_end = gt[j][1],
                        proposal_start = selected_props[j][0], proposal_end = selected_props[j][1],
                        duration = durations[j])
                    video_index += 1

        for k, v in metrics_logger.items():
            print('| {} {:.4f}'.format(k, v.avg), end=' ')
        print('|')
        # self.model.train()

    def _build_dataset(self):
        import datasets as da
        from gensim.models import KeyedVectors
        from torch.utils.data import DataLoader

        args = self.args['dataset']
        cls = getattr(da, args['dataset'], None)
        vocab = KeyedVectors.load_word2vec_format(args['vocab_path'], binary=True)

        # vidstg part
        self.train_set = cls(data_path=args['train_data'], vocab=vocab, args=args, split="train")
        self.test_set = cls(data_path=args['test_data'], vocab=vocab, args=args, split="test")
        self.val_set = cls(data_path=args['val_data'], vocab=vocab, args=args, split="val") if args['val_data']!='None' else None
        
        # self.train_set = cls(data_path=args['train_data'], vocab=vocab, args=args, is_training=True)
        # self.test_set = cls(data_path=args['test_data'], vocab=vocab, args=args)
        # self.val_set = cls(data_path=args['val_data'], vocab=vocab, args=args) if args['val_data'] else None
        # train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_set)
        # test_sampler = torch.utils.data.distributed.DistributedSampler(self.test_set)
        # val_sampler = torch.utils.data.distributed.DistributedSampler(self.val_set)

        # logging.info('train: {} samples, test: {} samples'.format(len(self.train_set), len(self.test_set)))
        batch_size = self.args['train']['batch_size']
        # print("this is training set:", self.train_set)

        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True,
                                       collate_fn=self.train_set.collate_data, num_workers=0,)
        self.test_loader = DataLoader(self.test_set, batch_size=batch_size, shuffle=False,
                                      collate_fn=self.test_set.collate_data,
                                      num_workers=1)
        self.val_loader = DataLoader(self.val_set, batch_size=batch_size, shuffle=False,
                                     collate_fn=self.val_set.collate_data,
                                     num_workers=1) if args['val_data']!='None' else None

    def _build_model(self):
        model_config = self.args['model']
        import models
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = torch.device('cuda:{}'.format(self.local_rank))
        self.model = getattr(models, model_config['name'], None)(model_config)
        self.model = self.model.to(device)

        # self.model = torch.nn.parallel.DistributedDataParallel(
        #     self.model, device_ids=[self.local_rank], 
        #     find_unused_parameters=False)
        # self.device_ids = self.local_rank

    def _build_optimizer(self):
        from optimizers import AdamOptimizer
        from optimizers.lr_schedulers import InverseSquareRootSchedule
        parameters = list(self.model.parameters())
        args = self.args['train']
        self.optimizer = AdamOptimizer(args, parameters)
        self.lr_scheduler = InverseSquareRootSchedule(args, self.optimizer)

    def _save_model(self, path):
        state_dict = {
            'num_updates': self.num_updates,
            'config': self.args,
            'model_parameters': self.model.state_dict(),
        }
        torch.save(state_dict, path)
        logging.info('save model to {}, num_updates {}.'.format(path, self.num_updates))

    def _load_model(self, path):
        # path = os.path.join(self.args.model_saved_path, name)
        # map_location = {'cuda:%d' % 0: 'cuda:%d' % self.local_rank}
        state_dict = torch.load(path, map_location="cuda:0")
        self.num_updates = state_dict['num_updates']
        self.lr_scheduler.step_update(self.num_updates)
        parameters = state_dict['model_parameters']
        self.model.load_state_dict(parameters)
        logging.info('load model from {}, num_updates {}.'.format(path, self.num_updates))


def calculate_IoU(i0, i1):
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))

    if union[1] - union[0] < -1e-5:
        return 0
    iou = 1.0 * (inter[1] - inter[0] + 1e-10) / (union[1] - union[0] + 1e-10)
    return iou if iou >= 0.0 else 0.0


def calculate_IoU_batch1(i0, i1):
    union = (np.min(np.stack([i0[0], i1[0]], 0), 0), np.max(np.stack([i0[1], i1[1]], 0), 0))
    inter = (np.max(np.stack([i0[0], i1[0]], 0), 0), np.min(np.stack([i0[1], i1[1]], 0), 0))
    # iou = 1.0 * (inter[1] - inter[0] + 1e-10) / (union[1] - union[0] + 1e-10)
    iou = 1.0 * (inter[1] - inter[0] + 1) / (union[1] - union[0] + 1)
    iou[union[1] - union[0] < -1e-5] = 0
    iou[iou < 0] = 0.0
    return iou


def calculate_IoU_batch2(i0, i1):
    union = (np.min(np.stack([i0[0], i1[0]], 0), 0), np.max(np.stack([i0[1], i1[1]], 0), 0))
    inter = (np.max(np.stack([i0[0], i1[0]], 0), 0), np.min(np.stack([i0[1], i1[1]], 0), 0))
    # iou = 1.0 * (inter[1] - inter[0] + 1e-10) / (union[1] - union[0] + 1e-10)
    iou = 1.0 * (inter[1] - inter[0] + 1e-10) / (union[1] - union[0] + 1e-10)
    iou[union[1] - union[0] < -1e-5] = 0
    iou[iou < 0] = 0.0
    return iou

def calculate_IoU_batch3(i0, i1):
    union_1 = torch.min(i0[:, 0], i1[:, 0])
    union_2 = torch.max(i0[:, 1], i1[:, 1])
    
    inter_1 = torch.max(i0[:, 0], i1[:, 0])
    inter_2 = torch.min(i0[:, 1], i1[:, 1])
    
    iou = 1.0 * (inter_2 - inter_1 + 1e-10) / (union_2 - union_1 + 1e-10)
    iou[union_2-union_1 < -1e-5] = 0
    iou[iou < 0] = 0
    # union = torch.tensor([torch.min(torch.stack([i0[0], i1[0]], 0), 0), torch.max(torch.stack([i0[1], i1[1]], 0), 0)])
    # inter = torch.tensor([torch.max(torch.stack([i0[0], i1[0]], 0), 0), torch.min(torch.stack([i0[1], i1[1]], 0), 0)])
    # # iou = 1.0 * (inter[1] - inter[0] + 1e-10) / (union[1] - union[0] + 1e-10)
    # iou = 1.0 * (inter[1] - inter[0] + 1e-10) / (union[1] - union[0] + 1e-10)
    # iou[union[1] - union[0] < -1e-5] = 0
    # iou[iou < 0] = 0.0
    return iou

# [nb, 2], [nb, 2]
# matrix_max = np.zeros([5])
def top_n_metric(preds, label):
    result = {}
    bsz = preds[0].shape[0]
    top_iou = []
    for pred in preds:
        iou = calculate_IoU_batch1((pred[:, 0], pred[:, 1]), (label[:, 0], label[:, 1]))
        top_iou.append(iou)
    
    # print(top_iou)
    iou_index = np.argmax(np.stack(top_iou, 1), 1)
    # matrix_max[iou_index] += 1
    # print(iou_index)

    iou = np.max(np.stack(top_iou, 1), 1)
    result['mIoU'] = np.mean(iou)
    for i in range(1, 10, 2):
        result['IoU@0.{}'.format(i)] = 1.0 * np.sum(iou >= i / 10) / bsz
    return result, iou_index


def top_1_metric(pred, label):
    result = {}
    bsz = pred.shape[0]
    iou = calculate_IoU_batch2((pred[:, 0], pred[:, 1]), (label[:, 0], label[:, 1]))
  
    result['mIoU'] = np.mean(iou)
    for i in range(1, 10, 2):
        result['IoU@0.{}'.format(i)] = 1.0 * np.sum(iou >= i / 10) / bsz
    return result, iou
