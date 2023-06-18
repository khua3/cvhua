from ctypes.wintypes import tagRECT
import os
import h5py
import torch
from torch.utils.data import Sampler
import spacy
import numpy as np
import pickle as pkl
import json
from winner_spatial.logging.configuration import get_logger

class FixedLengthBatchSampler(Sampler):

    def __init__(self, data_source, batch_size, include_partial=False, rng=None, maxlen=None,
                 length_to_size=None):
        self.data_source = data_source
        self.active = False
        if rng is None:
            rng = np.random.RandomState(seed=11)
        self.rng = rng
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.include_partial = include_partial
        self.length_to_size = length_to_size
        self._batch_size_cache = { 0: self.batch_size }
        self.logger = get_logger()

    def get_batch_size(self, length):
        if self.length_to_size is None:
            return self.batch_size
        if length in self._batch_size_cache:
            return self._batch_size_cache[length]
        start = max(self._batch_size_cache.keys())
        batch_size = self._batch_size_cache[start]
        for n in range(start+1, length+1):
            if n in self.length_to_size:
                batch_size = self.length_to_size[n]
            self._batch_size_cache[n] = batch_size
        return batch_size

    def reset(self):
        """
        Create a map of {length: List[example_id]} and maintain how much of
        each list has been seen.

        If include_partial is False, then do not provide batches that are below
        the batch_size.

        If length_to_size is set, then batch size is determined by length.

        """

        # Record the lengths of each example.
        length_map = {}
        for i in range(len(self.data_source)):
            x = self.data_source.dataset[i]
            length = len(x)

            if self.maxlen is not None and self.maxlen > 0 and length > self.maxlen:
                continue

            length_map.setdefault(length, []).append(i)

        # Shuffle the order.
        for length in length_map.keys():
            self.rng.shuffle(length_map[length])

        # Initialize state.
        state = {}
        for length, arr in length_map.items():
            batch_size = self.get_batch_size(length)
            nbatches = len(arr) // batch_size
            surplus = nbatches * batch_size < len(arr)
            state[length] = dict(nbatches=nbatches, surplus=surplus, position=-1)

        # Batch order, in terms of length.
        order = []
        for length, v in state.items():
            order += [length] * v['nbatches']

        ## Optionally, add partial batches.
        if self.include_partial:
            for length, v in state.items():
                if v['surplus']:
                    order += [length]

        self.rng.shuffle(order)

        self.length_map = length_map
        self.state = state
        self.order = order
        self.index = -1

    def get_next_batch(self):
        index = self.index + 1

        length = self.order[index]
        batch_size = self.get_batch_size(length)
        position = self.state[length]['position'] + 1
        start = position * batch_size
        batch_index = self.length_map[length][start:start+batch_size]

        self.state[length]['position'] = position
        self.index = index

        return batch_index

    def __iter__(self):
        self.reset()
        for _ in range(len(self)):
            yield self.get_next_batch()

    def __len__(self):
        return len(self.order)


class SimpleDataset(torch.utils.data.Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        item = self.dataset[index]
        return index, item, np.zeros(1), np.zeros(1), np.zeros(1)

    def __len__(self):
        return len(self.dataset)


class COCODataset(torch.utils.data.Dataset):

    def __init__(self, dataset, img_ids=None):
        self.dataset = dataset
        self.img_ids = img_ids
        self.data_path = './coco_data'

    def __getitem__(self, index):
        item = self.dataset[index]

        obj_feats = np.zeros(1).astype(np.int32) - 1
        boxes = np.zeros(1).astype(np.int32) - 1
        obj_cates = np.zeros(1).astype(np.int32) - 1
        return index, item, obj_feats, boxes, obj_cates

    def __len__(self):
        return len(self.dataset)


class FlickrDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, img_ids=None, mode='train'):
        self.dataset = dataset
        self.img_ids = img_ids
        self.mode = mode
        self.data_path = './flickr_data/flickr_feat_maf/'
        self.imgid2idx = pkl.load(open(self.data_path+f"{mode}_imgid2idx.pkl", "rb"))
        self.detection_dict = json.load(open(self.data_path+f"{mode}_detection_dict.json"))
        obj_vocab = open(self.data_path+"objects_vocab.txt").readlines()
        self.obj2ind = {obj.strip():idx for idx,obj in enumerate(obj_vocab)}
        
        with h5py.File(self.data_path+f"{mode}_features_compress.hdf5", "r") as hdf5_file:
            self.features = np.array(hdf5_file.get("features"))
            self.predicted_boxes = np.array(hdf5_file.get("bboxes"))
            self.indexes = np.array(hdf5_file.get("pos_bboxes"))

    def __getitem__(self, index):
        item = self.dataset[index]

        img_id = self.img_ids[index]
        feat_index = self.imgid2idx[int(img_id)]
        start_end_index = self.indexes[feat_index]
        num_box = min(start_end_index[1] - start_end_index[0], 36)
        # Get boxes
        boxes = np.zeros([36, 4]).astype(np.float32) - 1
        boxes[:num_box] = self.predicted_boxes[start_end_index[0] : start_end_index[1]][:num_box]
        # Get features
        obj_feats = np.zeros([36, 2048]).astype(np.float32)
        obj_feats[:num_box] = self.features[start_end_index[0] : start_end_index[1]][:num_box]
        # Get classes
        obj_cates = np.zeros([36]).astype(np.int32) - 1
        obj_cates[:num_box] = np.array([self.obj2ind.get(i) for i in
                              self.detection_dict[img_id]["classes"]]).astype(np.int32)[:num_box]
        

        return index, item, obj_feats, boxes, obj_cates

    def __len__(self):
        return len(self.dataset)


def create_1d_absolute_sin_cos_embedding(pos_len, dim):
    assert dim % 2 == 0, "wrong dimension!"
    position_emb = torch.zeros(pos_len, dim, dtype=torch.float)
    i_matrix = torch.arange(dim//2, dtype=torch.float)
    i_matrix /= dim / 2
    i_matrix = torch.pow(10000, i_matrix)
    i_matrix = 1 / i_matrix
    i_matrix = i_matrix.to(torch.long)
    pos_vec = torch.arange(pos_len).to(torch.long)
    out = pos_vec[:, None] @ i_matrix[None, :]
    emb_cos = torch.cos(out)
    emb_sin = torch.sin(out)
    position_emb[:, 0::2] = emb_sin
    position_emb[:, 1::2] = emb_cos
    return position_emb



def cal_ground_truth(pred_box, gt_box):
    
    x1 = np.maximum(gt_box[None, :, 0], pred_box[:, :, 0])
    y1 = np.maximum(gt_box[None, :, 1], pred_box[:, :, 1])
    x2 = np.minimum(gt_box[None, :, 2], pred_box[:, :, 2])
    y2 = np.minimum(gt_box[None, :, 3], pred_box[:, :, 3])

    inter = np.maximum(0, x2-x1+1.0) * np.maximum(0, y2-y1+1.0)
    S_a = (gt_box[:, 2]-gt_box[:, 0]+1.0)*(gt_box[:, 3]-gt_box[:, 1]+1.0)
    S_b = (pred_box[:, :, 2]-pred_box[:, :, 0]+1.0)*(pred_box[:, :, 3]-pred_box[:, :, 1]+1.0)
    union = S_a + S_b - inter
    IoU = inter/union

    IoU = IoU.reshape(5, -1)  
    candidate_iou = np.average(IoU, axis=1)
    max_index = np.argmax(candidate_iou).reshape(-1)

    return max_index



class VidstgDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, mode='train'):
        self.dataset = dataset
        self.mode = mode

        self.pos_embed = create_1d_absolute_sin_cos_embedding(5, 1024)
        vocab_path = './dataset_scope_gt/'
        obj_vocab = open(vocab_path+"objects_vocab.txt").readlines()
        self.obj2ind = {obj.strip():idx for idx,obj in enumerate(obj_vocab)}
        
        features = []
        predicted_boxes = []
        classes = []
        gt_boxes = []
        pro_start = []
        pro_end = []
        gt_start = []
        gt_end = []
        video_ids = []
        target_tokens_index = []
        pair_token_index = []
        gt_tubes_index = []
        # noun_index_for_cal = []

        parse_tool = spacy.load("en_core_web_sm")
        data_path = "./spa_proposal_best"
        for bid, file in enumerate(os.listdir(os.path.join(data_path, mode))):
            if bid>=3610*1 and mode=='train':
                break
            tmp = np.load("{}/{}/{}.npz".format(data_path, mode, bid), allow_pickle=True)
            features.append(tmp["feats"][0:5])
            predicted_boxes.append(tmp["box"][0:5])
            classes.append(np.zeros([20]))
            gt_boxes.append(tmp['gt_box'])
            max_index = cal_ground_truth(tmp['box'][0:5], tmp['gt_box'])
            gt_tubes_index.append(max_index)



        tem_path = "./pretrain_temporal"
        anno_path = "./VidSTG-Dataset/VidSTG"
        annos = json.load(open("{}/{}.json".format(anno_path, mode), 'r'))
        annos = annos['videos']
        for bid, anno in enumerate(annos):
            if bid>=3610*1 and mode=='train':
                break
            
            original_video_id = anno["original_video_id"]
            video_index = anno["video_id"]
            
            caption = anno['caption']
            doc = parse_tool(caption)
            noun_index = []
            for bid, token in enumerate(doc):
                if token.pos_ == "NOUN":
                    noun_index.append(bid)
            assert len(noun_index)>=1
            target_index = noun_index[0]
            target_tokens_index.append(target_index)
            noun_index = np.array(noun_index)

            pair_token = -1
            pair_token_index.append(pair_token)
            
            hash_id = "{}_{}".format(original_video_id, video_index)
            tem_file = "{}/{}/{}.npz".format(tem_path, mode, hash_id)
            tem_data = np.load(tem_file)
            gt_start.append(tem_data['gt_start'])
            gt_end.append(tem_data['gt_end'])
            pro_start.append(tem_data['proposal_start'])
            pro_end.append(tem_data['proposal_end'])
            video_ids.append(hash_id)
            

        self.features = features
        self.boxes = predicted_boxes
        self.classes = classes
        self.gt_boxes = gt_boxes
        self.pro_start = pro_start
        self.pro_end = pro_end
        self.gt_start = gt_start
        self.gt_end = gt_end
        self.video_ids = video_ids 
        self.target_tokens = target_tokens_index
        self.pair_tokens = pair_token_index
        self.gt_tubes_index = gt_tubes_index
        
    def __getitem__(self, index):

        item = self.dataset[index]
        obj_feats = np.array(self.features[index]) + np.array(self.pos_embed)
        obj_cates = np.array(self.classes[index])
        boxes = np.array(self.boxes[index])
        if self.mode=='test' or self.mode=='val':
            gt_boxes = np.array(self.gt_boxes[index])
            gt_proposal = [self.gt_start[index], self.gt_end[index]]
            pred_proposal = [self.pro_start[index], self.pro_end[index]]
            video_id = self.video_ids[index]
        else:
            gt_boxes = []
            gt_proposal = []
            pred_proposal = []
            video_id = []

        return  index, item, obj_feats, boxes, obj_cates, \
                gt_boxes, gt_proposal, pred_proposal, video_id, \
                self.target_tokens[index], self.pair_tokens[index], \
                self.gt_tubes_index[index]

    def __len__(self):
        return len(self.dataset)
