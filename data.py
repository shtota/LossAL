# General
import numpy as np
import importlib
import os
# Torch
import torch
import torchvision.transforms as T
from torchtext.data.functional import to_map_style_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import AG_NEWS, IMDB, YahooAnswers, YelpReviewPolarity, YelpReviewFull
from torchtext.vocab import build_vocab_from_iterator, GloVe
from torchvision.datasets import CIFAR100, CIFAR10, SVHN

# Custom


def _yield_tokens(data_iter):
    tokenizer = get_tokenizer('basic_english')
    for _, text in data_iter:
        yield tokenizer(text)


def _prepare_image_data(ds, work_dir):
    config = importlib.import_module('config.' + ds)
    config.DATASET = ds
    config.TASK = 'image'
    data_dir = os.path.join(work_dir, 'data', ds)
    if ds == 'cifar10' or ds == 'cifar100':
        train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(size=32, padding=4),
            T.ToTensor(),
            T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
        if ds == 'cifar10':
            data_train = CIFAR10(data_dir, train=True, download=True, transform=train_transform)
            data_unlabeled = CIFAR10(data_dir, train=True, download=True, transform=test_transform)
            data_test = CIFAR10(data_dir, train=False, download=True, transform=test_transform)
        else:
            data_train = CIFAR100(data_dir, train=True, download=True, transform=train_transform)
            data_unlabeled = CIFAR100(data_dir, train=True, download=True, transform=test_transform)
            data_test = CIFAR100(data_dir, train=False, download=True, transform=test_transform)

    elif ds == 'svhn':
        train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(size=32, padding=4),
            T.ToTensor(),
            T.Normalize([0.4310, 0.4302, 0.4463], [0.1965, 0.1984, 0.1992])
        ])
        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.4310, 0.4302, 0.4463], [0.1965, 0.1984, 0.1992])
        ])
        target_transform = None

        data_train = SVHN(root=data_dir, split='train', transform=train_transform, download=True,
                          target_transform=target_transform)
        data_unlabeled = SVHN(root=data_dir, split='train', transform=test_transform, download=True,
                              target_transform=target_transform)
        data_test = SVHN(root=data_dir, split='test', transform=test_transform, download=True,
                         target_transform=target_transform)
    else:
        raise ValueError
    return config, data_train, data_unlabeled, data_test, None, None


def _prepare_text_data(ds, work_dir):
    ds_to_class = {'ag': AG_NEWS, 'imdb': IMDB, 'yahoo': YahooAnswers, 'yelp': YelpReviewPolarity,
                   'yelp5': YelpReviewFull}
    config = importlib.import_module('config.text')
    config.DATASET = ds
    config.TASK = 'text'
    data_dir = os.path.join(work_dir, 'data', ds)
    if ds == 'imdb' or ds == 'yelp':
        config.CLASS = 2
    elif ds == 'yahoo':
        config.CLASS = 10
        config.BATCH = 16
    elif ds == 'yelp5':
        config.CLASS = 5
    train_iter, test_iter = ds_to_class[ds](data_dir)

    train_iter = list(train_iter)
    test_iter = list(test_iter)
    np.random.RandomState(0).shuffle(train_iter)
    train_iter = train_iter[:config.NUM_TRAIN]
    data_train = to_map_style_dataset(train_iter)
    data_unlabeled = to_map_style_dataset(train_iter)
    data_test = to_map_style_dataset(test_iter)

    vecs = GloVe()
    tokenizer = get_tokenizer('basic_english')
    my_vocab = build_vocab_from_iterator(_yield_tokens(train_iter), specials=["<unk>"])
    my_vocab.set_default_index(my_vocab["<unk>"])
    pretrained_embedding = vecs.get_vecs_by_tokens(my_vocab.get_itos())

    def text_pipeline(x):
        return my_vocab(tokenizer(x))

    def label_pipeline(x):
        if ds == 'imdb':
            return x == 'pos'
        return int(x) - 1

    def collate_fn(batch):
        label_list, text_list, offsets = [], [], [0]
        for (_label, _text) in batch:
            label_list.append(label_pipeline(_label))
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return (text_list.cuda(), offsets.cuda()), label_list.cuda()

    return config, data_train, data_unlabeled, data_test, collate_fn, pretrained_embedding


def prepare(ds, work_dir):
    if ds in ['cifar10', 'cifar100', 'svhn']:
        return _prepare_image_data(ds, work_dir)
    return _prepare_text_data(ds, work_dir)
