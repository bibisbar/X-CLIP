from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np
import json
import math
import csv
from dataloaders.rawvideo_util import RawVideoExtractor

class MOVIEGraph_Dataloader(Dataset):
    """MOVIEGraph dataset loader."""
    def __init__(
            self,
            subset,
            data_path,
            features_path,
            tokenizer,
            train_file,
            val_file,
            test_file,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
    ):
        self.data_path = data_path
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]
        self.subset = subset
        assert self.subset in ["train", "val", "test"]
        #TODO: change the path
        video_json_path_dict = {}
        video_json_path_dict["train"] = os.path.join(self.data_path, train_file) 
        video_json_path_dict["val"] = os.path.join(self.data_path, val_file)
        video_json_path_dict["test"] = os.path.join(self.data_path, test_file)
        # <CLIP_ID>\t<SENTENCE>
        # <CLIP_ID> is not a unique identifier, i.e. the same <CLIP_ID> can be associated with multiple sentences.
        

        #get the (clip_id, sentence) from the video_json_path_dict
        video_id_list = []
        caption_dict = {}
        with open(video_json_path_dict[self.subset], 'r') as fp:
            for line in fp:
                line = line.strip()
                line_split = line.split(".mp4")
                #print(line_split)
                assert len(line_split) == 2
                clip_id, sentence = line_split
                #clip_id = ".".join(clip_id.split(".")[:-1])
                caption_dict[len(caption_dict)] = (clip_id, sentence)
                if clip_id not in video_id_list: video_id_list.append(clip_id)
            #  for line in fp:
            #     line = line.strip()
            #     line =line.replace('"','')
            #     line_split = line.split("	")
            #     assert len(line_split) == 2
            #     clip_id, sentence = line_split
            #     clip_id = ".".join(clip_id.split(".")[:-1])
            #     caption_dict[len(caption_dict)] = (clip_id, sentence)
            #     if clip_id not in video_id_list: video_id_list.append(clip_id)   # remove the suffix
        
        #TODO: change the join method regarding path
        video_dict = {}
        for root, dub_dir, video_files in os.walk(self.features_path):
            for video_file in video_files:
                video_id_ = ".".join(video_file.split(".")[:-1])      # remove the suffix
                if video_id_ not in video_id_list:
                    continue
                file_path_ = os.path.join(root, video_file)
                video_dict[video_id_] = file_path_

        self.video_dict = video_dict

        # Get all captions
        self.iter2video_pairs_dict = {}
        for clip_id, sentence in caption_dict.values():
            if clip_id not in self.video_dict:
                continue
            self.iter2video_pairs_dict[len(self.iter2video_pairs_dict)] = (clip_id, sentence)

        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
    def __len__(self):
        return len(self.iter2video_pairs_dict)


    def _get_text(self, video_id, caption):
        k = 1
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

        for i, video_id in enumerate(choice_video_ids):
            words = self.tokenizer.tokenize(caption)

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _get_rawvideo(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float)

        try:
            for i, video_id in enumerate(choice_video_ids):
                video_path = self.video_dict[video_id]

                raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
                raw_video_data = raw_video_data['video']

                if len(raw_video_data.shape) > 3:
                    raw_video_data_clip = raw_video_data
                    # L x T x 3 x H x W
                    raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
                    if self.max_frames < raw_video_slice.shape[0]:
                        if self.slice_framepos == 0:
                            video_slice = raw_video_slice[:self.max_frames, ...]
                        elif self.slice_framepos == 1:
                            video_slice = raw_video_slice[-self.max_frames:, ...]
                        else:
                            sample_indx = np.linspace(0, raw_video_slice.shape[0]-1, num=self.max_frames, dtype=int)
                            video_slice = raw_video_slice[sample_indx, ...]
                    else:
                        video_slice = raw_video_slice

                    video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)

                    slice_len = video_slice.shape[0]
                    max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                    if slice_len < 1:
                        pass
                    else:
                        video[i][:slice_len, ...] = video_slice
                else:
                    print("video path: {} error. video id: {}".format(video_path, video_id))
        except Exception as excep:
            print("Video ids: {}".format(choice_video_ids))
            raise excep

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length
        return video, video_mask

    def __getitem__(self, feature_idx):
        clip_id, sentence = self.iter2video_pairs_dict[feature_idx]
        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(clip_id, sentence)
        video, video_mask = self._get_rawvideo(choice_video_ids)
        return pairs_text, pairs_mask, pairs_segment, video, video_mask

