import json
import h5py
import torch
import numpy as np
from utils import iou
from gensim.utils import tokenize
from torch.utils.data import Dataset
from runners.main_runner import calculate_IoU_batch3
from datasets.base_dataset import build_collate_data, calculate_IoU_batch


class VidSTG(Dataset):
    def __init__(self, data_path, vocab, args, split):
        self.vocab = vocab
        self.args = args
        self.data = json.load(open(data_path,'r'))
        self.max_num_frames = args['max_num_frames']
        self.max_num_words = args['max_num_words']
        self.target_stride = args['target_stride']

        self.split = split
        self._get_proposal()
        self._get_annotation()

        self.collate_fn = build_collate_data(
                        self.max_num_frames, self.max_num_words,
                        args['frame_dim'], args['word_dim'],
                        self.props_torch, self.props_graph_torch)
        
    def _load_frame_features(self, hash_id=None, index=-1, split=None):
        if self.args["feature_type"] == 'rcnn':
            if split != "train":
                with h5py.File("{}/{}.hdf5".format(self.args['feature_path'], split), 'r') as fr:
                    return np.array(fr[hash_id]).astype(np.float32)
            else:
                with np.load("{}/train/{}.npz".format(self.args['feature_path'], index), allow_pickle=True) as fr:
                    return np.array(fr["vidstg_feature"]).astype(np.float32)
        elif self.args["feature_type"] == 'c3d':
            if split != "train":
                with h5py.File("{}/{}_slide8.hdf5".format(self.args['feature_path'], split), 'r') as fr:
                    return np.array(fr[hash_id]).astype(np.float32)
            else:
                with np.load("{}/train_features/{}.npz".format(self.args['feature_path'], index), allow_pickle=True) as fr:
                    return np.array(fr["c3d_feature"]).astype(np.float32)

    def __len__(self):
        return len(self.data["videos"])

    def __getitem__(self, index):
        
        fps = self.annotations[index]['fps']
        hash_id = self.annotations[index]["hash_id"]
        sentence = self.annotations[index]["sentence"]
        video_start = self.annotations[index]["video_start"]
        video_end = self.annotations[index]["video_end"]
        gt_start = (self.annotations[index]["gt_start"] - video_start) / fps
        gt_end = (self.annotations[index]["gt_end"] - video_start) / fps

        video_duration = (video_end - video_start) / fps
        gt_end = min(gt_end, video_duration)
        
        words = [w.lower() for w in tokenize(sentence)]
        words = [w for w in words if w in self.vocab]
        words_feat = [self.vocab[w].astype(np.float32) for w in words]
        
        num_clips = self.max_num_frames // self.target_stride
        props = self.props_torch.float() * video_duration / num_clips
        gts = torch.tensor([gt_start, gt_end]).unsqueeze(0).expand(props.size(0), -1).float()
        map_gt = calculate_IoU_batch3(props, gts)

        reg_gt = self.props_torch[torch.argmax(map_gt)].int()

        frame_gt = torch.zeros(num_clips).float()
        frame_gt[reg_gt[0]:reg_gt[1] - 1] = 1.0

        frames_feat = self._load_frame_features(hash_id, self.annotations[index]["hash_id"].split('_')[-1], self.split)
        return {
            'frames_feat': frames_feat,
            'words_feat': words_feat,
            'map_gt': map_gt,
            'reg_gt': reg_gt,
            'frame_gt': frame_gt,
            'raw': [hash_id, video_duration, [gt_start, gt_end], sentence]
        }


    def _get_proposal(self):

        self.num_clips = self.max_num_frames // self.target_stride
        self.props = []
        tmp = [[1], [2], [2]]
        tmp[0].extend([1] * 15)
        tmp[1].extend([1] * 7)
        tmp[2].extend([1] * 7)
        acum_layers = 0
        stride = 1
        for scale_idx, strides in enumerate(tmp):
            for i, stride_i in enumerate(strides):

                stride = stride * stride_i
                keep = False

                if scale_idx == 0 and i in [3, 7, 15]:
                    keep = True
                elif scale_idx == 1 and (i in [3, 7]):
                    keep = True
                elif scale_idx == 2 and (i in [3, 5, 7]):
                    keep = True

                if keep==False:
                    continue
                    
                ori_s_idxs = list(range(0, self.num_clips - acum_layers - i * stride, stride))
                ori_e_idxs = [s_idx + acum_layers + i * stride for s_idx in ori_s_idxs]
                self.props.append(np.stack([ori_s_idxs, ori_e_idxs], -1))

            acum_layers += stride * (len(strides) + 1)
        self.props = np.concatenate(self.props, 0)
        self.props[:, 1] += 1

        # predefined proposals graph
        props_iou = iou(self.props.tolist(), self.props.tolist())
        self.props_graph = np.zeros_like(props_iou).astype(np.int32)
        
        sort_idx = np.argsort(-props_iou, -1)
        for i in range(self.props.shape[0]):
            self.props_graph[i, sort_idx[i]] = 1
            low_idx = props_iou[i] < 0.6
            self.props_graph[i, low_idx] = 0

        self.props_torch = torch.from_numpy(self.props)
        self.props_graph_torch = torch.from_numpy(self.props_graph)
        
        # print("Let's look at self.pros_graph_torch: ", self.props_graph_torch, "\n", self.props_graph_torch.shape)
        # print("Let's look at self.pros_torch: ", self.props_torch, "\n", self.props_torch.shape)
        # exit(1)


    def _get_annotation(self):
        anno_pairs = []
        annotations = self.data["videos"]
        # dis = np.zeros([101])

        for video_anno in annotations:
            original_video_id = video_anno["original_video_id"]
            sentence = video_anno["caption"]
            video_index = video_anno["video_id"]
            hash_id = "{}_{}".format(original_video_id, video_index)
            video_start = video_anno["start_frame"]
            video_end = video_anno["end_frame"]
            fps = video_anno['fps']

            gt_start = video_anno["tube_start_frame"]
            gt_end = video_anno["tube_end_frame"]
            gt_end = min(gt_end, video_end)
            # tmp = round((gt_end-gt_start)/(video_end-video_start)*100)
            # dis[tmp] += 1

            anno_pairs.append({
                "hash_id": hash_id,
                "video_start": video_start,
                "video_end":video_end,
                "gt_start": gt_start,
                "gt_end": gt_end,
                "sentence": sentence,
                'fps':fps
            })

        
        # print(np.sum(dis[10:]) / np.sum(dis))
        # print(np.sum(dis[20:])/ np.sum(dis))
        # print(np.sum(dis[30:])/ np.sum(dis))
        # print(np.sum(dis[50:])/ np.sum(dis))
        
        # print(dis)
        # exit()
        self.annotations = anno_pairs

    def collate_data(self, samples):
        return self.collate_fn(samples)
