# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""PyTorch Dataset implementation for Iterative CLEVR dataset"""
from collections import defaultdict
import json

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob
import os
import json

from geneva.utils.config import keys


class ICLEVERDataset(Dataset):
    def __init__(self, path, cfg, img_size=128):
        super().__init__()
        self.dataset = None
        self.dataset_path = path

        self.glove = _parse_glove(keys['glove_path'])
        with h5py.File(path, 'r') as f:
            self.keys = list(f.keys())
            self.background = f['background'].value
            self.background = cv2.resize(self.background, (128, 128))
            self.background = self.background.transpose(2, 0, 1)
            self.background = self.background / 128. - 1
            self.background += np.random.uniform(size=self.background.shape,
                                                 low=0, high=1. / 128)
            self.entities = np.array(json.loads(f['entities'].value))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.dataset_path, 'r')

        example = self.dataset[self.keys[idx]]

        scene_id = example['scene_id'].value
        images = example['images'].value
        text = example['text'].value
        objects = example['objects'].value

        images = images[..., ::-1]
        images = images / 128. - 1
        images += np.random.uniform(size=images.shape, low=0, high=1. / 128)
        images = images.transpose(0, 3, 1, 2)

        text = json.loads(text)

        turns_tokenized = [t.split() for t in text]
        lengths = [len(t) for t in turns_tokenized]
        turn_word_embeddings = np.zeros((len(text), max(lengths), 300))

        for i, turn in enumerate(turns_tokenized):
            for j, w in enumerate(turn):
                turn_word_embeddings[i, j] = self.glove[w]

        sample = {
            'scene_id': scene_id,
            'image': images,
            'turn': text,
            'objects': objects,
            'turn_word_embedding': turn_word_embeddings,
            'turn_lengths': lengths,
            'background': self.background,
            'entities': self.entities,
        }

        return sample


def _parse_glove(glove_path):
    glove = {}
    with open(glove_path, 'r') as f:
        for line in f:
            splitline = line.split()
            word = splitline[0]
            embedding = np.array([float(val) for val in splitline[1:]])
            glove[word] = embedding

    return glove


def collate_data(batch):
    batch = sorted(batch, key=lambda x: len(x['image']), reverse=True)
    dialog_lengths = list(map(lambda x: len(x['image']), batch))
    max_len = max(dialog_lengths)

    batch_size = len(batch)
    _, c, h, w = batch[0]['image'].shape

    batch_longest_turns = [max(b['turn_lengths']) for b in batch]
    longest_turn = max(batch_longest_turns)

    stacked_images = np.zeros((batch_size, max_len, c, h, w))
    stacked_turns = np.zeros((batch_size, max_len, longest_turn, 300))
    stacked_turn_lengths = np.zeros((batch_size, max_len))
    stacked_objects = np.zeros((batch_size, max_len, 24))
    turns_text = []
    scene_ids = []

    background = None
    for i, b in enumerate(batch):
        img = b['image']
        turns = b['turn']
        background = b['background']
        entities = b['entities']
        turns_word_embedding = b['turn_word_embedding']
        turns_lengths = b['turn_lengths']

        dialog_length = img.shape[0]
        stacked_images[i, :dialog_length] = img
        stacked_turn_lengths[i, :dialog_length] = np.array(turns_lengths)
        stacked_objects[i, :dialog_length] = b['objects']
        turns_text.append(turns)
        scene_ids.append(b['scene_id'])

        for j, turn in enumerate(turns_word_embedding):
            turn_len = turns_lengths[j]
            stacked_turns[i, j, :turn_len] = turn[:turn_len]

    sample = {
        'scene_id': np.array(scene_ids),
        'image': torch.FloatTensor(stacked_images),
        'turn': np.array(turns_text),
        'turn_word_embedding': torch.FloatTensor(stacked_turns),
        'turn_lengths': torch.LongTensor(stacked_turn_lengths),
        'dialog_length': torch.LongTensor(np.array(dialog_lengths)),
        'background': torch.FloatTensor(background),
        'entities': entities,
        'objects': torch.FloatTensor(stacked_objects),
    }

    return sample






# class CRIMDataset(Dataset):
#     def __init__(self, data_path, cfg, img_size=128):
#         super().__init__()
#         self.dataset = None
#         self.dataset_path = data_path

#         self.glove = _parse_glove(keys['glove_path'])
#         # with h5py.File(path, 'r') as f:
#         #     self.keys = list(f.keys())
#         #     self.background = f['background'].value
#         #     self.background = cv2.resize(self.background, (128, 128))
#         #     self.background = self.background.transpose(2, 0, 1)
#         #     self.background = self.background / 128. - 1
#         #     self.background += np.random.uniform(size=self.background.shape,
#         #                                          low=0, high=1. / 128)
#         #     self.entities = np.array(json.loads(f['entities'].value))

#         # make imagenames2command dictionary, images come from images_c1 folder
#         im2comm = dict()
#         with open(os.path.join(self.dataset_path, 'CLEVR_questions'), 'r') as f:
#             command_dicts = json.load(f)['questions']
#             im2comm = {command_dict['image_output_filename']: command_dict for  command_dict in command_dicts}


#         # use the data_path to make a list of images/, images_c1/
#         images = sorted(glob(os.path.join(self.dataset_path, 'images', 'CLEVR_new*'))) # the images before change command
#         images_c1 = sorted(glob(os.path.join(self.dataset_path, 'images_c1', 'CLEVR_new*'))) # the images after change command
#         # combine the above images and make tuples
#         self.command_data_tuples = [] # of the form [(img1_name, image1_changed_name, command), ....]
#         images_c1_counter = 0
#         for img in images:
#             image_name = img.split('.')[0] # eg CLEVR_new_000041
#             while image_name in images_c1[images_c1_counter]:
#                 changed_image_name = images_c1[images_c1_counter]
#                 self.command_data_tuples.append((img, changed_image_name, im2comm[changed_image_name]))
#                 images_c1_counter+=1
#         assert len(self.command_data_tuples) == images_c1

#     def __len__(self):
#         return len(self.command_data_tuples)

#     def __getitem__(self, idx):
#         # image_source = cv2.resize(cv2.imread(self.command_data_tuples[idx][0]), (128, 128))
#         # image_target = cv2.resize(cv2.imread(self.command_data_tuples[idx][1]), (128, 128))
#         # text = self.command_data_tuples[idx][2]['question']
        

#         # sample = {
#         #     'scene_id': scene_id,
#         #     'image_source': image_source,
#         #     'image_target': image_target,
#         #     'turn': text,
#         #     'objects': objects,
#         #     'turn_word_embedding': turn_word_embeddings,
#         #     'turn_lengths': lengths,
#         #     'background': self.background,
#         #     'entities': self.entities,
#         # }

#         if self.dataset is None:
#             self.dataset = h5py.File(self.dataset_path, 'r')

#         example = self.dataset[self.keys[idx]]

#         scene_id = example['scene_id'].value
#         images = example['images'].value
#         text = example['text'].value
#         objects = example['objects'].value

#         images = images[..., ::-1]
#         images = images / 128. - 1
#         images += np.random.uniform(size=images.shape, low=0, high=1. / 128)
#         images = images.transpose(0, 3, 1, 2)

#         text = json.loads(text)

#         turns_tokenized = [t.split() for t in text]
#         lengths = [len(t) for t in turns_tokenized]
#         turn_word_embeddings = np.zeros((len(text), max(lengths), 300))

#         for i, turn in enumerate(turns_tokenized):
#             for j, w in enumerate(turn):
#                 turn_word_embeddings[i, j] = self.glove[w]

#         sample = {
#             'scene_id': scene_id,
#             'image': images,
#             'turn': text,
#             'objects': objects,
#             'turn_word_embedding': turn_word_embeddings,
#             'turn_lengths': lengths,
#             'background': self.background,
#             'entities': self.entities,
#         }

#         return sample


# def _parse_glove(glove_path):
#     glove = {}
#     with open(glove_path, 'r') as f:
#         for line in f:
#             splitline = line.split()
#             word = splitline[0]
#             embedding = np.array([float(val) for val in splitline[1:]])
#             glove[word] = embedding

#     return glove


# def collate_data(batch):
#     batch = sorted(batch, key=lambda x: len(x['image']), reverse=True)
#     dialog_lengths = list(map(lambda x: len(x['image']), batch))
#     max_len = max(dialog_lengths)

#     batch_size = len(batch)
#     _, c, h, w = batch[0]['image'].shape

#     batch_longest_turns = [max(b['turn_lengths']) for b in batch]
#     longest_turn = max(batch_longest_turns)

#     stacked_images = np.zeros((batch_size, max_len, c, h, w))
#     stacked_turns = np.zeros((batch_size, max_len, longest_turn, 300))
#     stacked_turn_lengths = np.zeros((batch_size, max_len))
#     stacked_objects = np.zeros((batch_size, max_len, 24))
#     turns_text = []
#     scene_ids = []

#     background = None
#     for i, b in enumerate(batch):
#         img = b['image']
#         turns = b['turn']
#         background = b['background']
#         entities = b['entities']
#         turns_word_embedding = b['turn_word_embedding']
#         turns_lengths = b['turn_lengths']

#         dialog_length = img.shape[0]
#         stacked_images[i, :dialog_length] = img
#         stacked_turn_lengths[i, :dialog_length] = np.array(turns_lengths)
#         stacked_objects[i, :dialog_length] = b['objects']
#         turns_text.append(turns)
#         scene_ids.append(b['scene_id'])

#         for j, turn in enumerate(turns_word_embedding):
#             turn_len = turns_lengths[j]
#             stacked_turns[i, j, :turn_len] = turn[:turn_len]

#     sample = {
#         'scene_id': np.array(scene_ids),
#         'image': torch.FloatTensor(stacked_images),
#         'turn': np.array(turns_text),
#         'turn_word_embedding': torch.FloatTensor(stacked_turns),
#         'turn_lengths': torch.LongTensor(stacked_turn_lengths),
#         'dialog_length': torch.LongTensor(np.array(dialog_lengths)),
#         'background': torch.FloatTensor(background),
#         'entities': entities,
#         'objects': torch.FloatTensor(stacked_objects),
#     }

#     return sample
