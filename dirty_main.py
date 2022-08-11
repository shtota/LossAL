# General
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product

# Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler, Sampler

# Custom
from models.resnet import ResNet18
from models.vgg import VGG16
from models.text import TextClassificationModel
from data import prepare
from query_strategies import query

torch.backends.cudnn.benchmark = True
AUXILIARY = 'NONE'
TEST_BATCH = 1024


class SubsetSequentialSampler(Sampler):
    def __init__(self, indices):
        super().__init__(None)
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class Runner:
    def __init__(self, datasets, models, samplings, trials, start_trial, cycles, work_dir):
        self.datasets = datasets
        self.model_names = models
        self.samplings = samplings
        self.start_trial = start_trial
        self.trials = trials
        self.cycles = cycles
        self.work_dir = work_dir

        os.makedirs(os.path.join(work_dir, 'results'), exist_ok=True)
        os.makedirs(os.path.join(work_dir, 'weights'), exist_ok=True)
        os.makedirs(os.path.join(work_dir, 'data'), exist_ok=True)
        self.dataset, self.model_name, self.sampling, self.config, self.data_train, self.data_unlabeled, self.data_test, self.collate_fn, self.pretrained_embedding = [None]*9

    def results_path(self, trial):
        return os.path.join(self.work_dir, 'results', f'{self.dataset}_{self.sampling}_{trial}_{self.model_name}_{AUXILIARY}.csv')

    def initial_model_path(self, trial):
        return os.path.join(self.work_dir, 'weights',
                            f'{self.dataset}_auxiliary_{AUXILIARY}_trial{trial+1}_start{self.config.START}_model_{self.model_name}.pth')

    def checkpoint_path(self, trial, cycle):
        return os.path.join(self.work_dir, 'weights',
                            f'{self.dataset}_auxiliary_{AUXILIARY}_sampling_{self.sampling}_trial{trial+1}_cycle{cycle+1}_start{self.config.START}_model_{self.model_name}.pth')

    def run(self):
        for dataset in self.datasets:
            self.dataset = dataset
            self.config, self.data_train, self.data_unlabeled, self.data_test, self.collate_fn, self.pretrained_embedding = prepare(dataset, self.work_dir)
            for model_name, sampling, trial in product(self.model_names, self.samplings, range(self.start_trial, self.trials)):
                self.model_name = model_name
                self.sampling = sampling
                self.run_trial(trial)

    def train(self, model, dataloaders, cycle):
        # Loss, criterion and scheduler (re)initialization
        criterion = nn.CrossEntropyLoss(reduction='none') if self.config.CLASS > 2 else nn.BCEWithLogitsLoss(
            reduction='none')
        optimizer = optim.SGD(model.parameters(), lr=self.config.LR, momentum=self.config.MOMENTUM,
                              weight_decay=self.config.WDECAY)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=self.config.MILESTONES)
        best_acc = 0.

        for epoch in tqdm(range(self.config.EPOCH)):
            model.train()

            for (inputs, labels) in dataloaders['train']:
                if self.config.TASK == 'image':
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                optimizer.zero_grad()
                scores, features = model(inputs)
                if self.config.CLASS == 2:
                    target_loss = criterion(scores, labels.float().reshape(-1, 1))
                else:
                    target_loss = criterion(scores, labels)
                loss = torch.sum(target_loss) / target_loss.size(0)
                loss.backward()
                optimizer.step()

            scheduler.step()

            if epoch and epoch % 25 == 0:
                acc = self.test(model, dataloaders['test'])
                if best_acc < acc:
                    best_acc = acc
                print(self.dataset, 'Cycle:', cycle + 1, 'Epoch:', epoch, '---',
                      'Val Acc: {:.2f} \t Best Acc: {:.2f}'.format(acc, best_acc))

    def test(self, model, dataloader):
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for (inputs, labels) in dataloader:
                if self.config.TASK == 'image':
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                scores, _ = model(inputs)
                if self.config.CLASS == 2:
                    preds = (scores.reshape(-1) > 0)
                else:
                    _, preds = torch.max(scores.data, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        return 100 * correct / total

    def run_trial(self, trial):
        pin_memory = self.config.TASK == 'image'
        if os.path.exists(self.results_path(trial)):
            print(f'Skipping {self.dataset} {self.model_name} {self.sampling} {trial}')
            return
        results = []
        indices = list(range(self.config.NUM_TRAIN))
        np.random.RandomState(trial).shuffle(indices)
        labeled_set = indices[:self.config.START]
        unlabeled_set = indices[self.config.START:]

        torch.manual_seed(trial)
        train_loader = DataLoader(self.data_train, batch_size=self.config.BATCH, sampler=SubsetRandomSampler(labeled_set),
                                  pin_memory=pin_memory, collate_fn=self.collate_fn)
        test_loader = DataLoader(self.data_test, batch_size=TEST_BATCH, collate_fn=self.collate_fn)
        unlabeled_loader = DataLoader(self.data_unlabeled, batch_size=self.config.BATCH,
                                      sampler=SubsetSequentialSampler(unlabeled_set),
                                      pin_memory=pin_memory, collate_fn=self.collate_fn)
        pool_loader = DataLoader(self.data_unlabeled, batch_size=self.config.BATCH,
                                 sampler=SubsetSequentialSampler(sorted(indices)),
                                 pin_memory=pin_memory, collate_fn=self.collate_fn)
        dataloaders = {'train': train_loader, 'test': test_loader, 'unlabeled': unlabeled_loader, 'all': pool_loader}
        if self.config.TASK == 'image':
            model = ResNet18(num_classes=self.config.CLASS).cuda() if self.model_name == 'resnet' else VGG16(
                num_classes=self.config.CLASS).cuda()

        for cycle in range(self.cycles):
            if self.config.TASK == 'text':
                model = TextClassificationModel(self.pretrained_embedding.shape[0], num_class=self.config.CLASS,
                                                embedding=self.pretrained_embedding if 'Emb' in self.model_name else None,
                                                freeze='Freeze' in self.model_name).cuda()

            # Training and test
            if cycle == 0:
                initial_model_path = self.initial_model_path(trial)
                if os.path.exists(initial_model_path):
                    print('loading initial model')
                    checkpoint = torch.load(initial_model_path)
                    model.load_state_dict(checkpoint['state_dict_backbone'])
                else:
                    self.train(model, dataloaders, cycle)
                    torch.save({'state_dict_backbone': model.state_dict(), 'labeled_set': labeled_set}, initial_model_path)
            else:
                self.train(model, dataloaders, cycle)
                to_save = {'cycle': cycle + 1, 'labeled_set': labeled_set}
                if self.config.TASK == 'image':
                    to_save['state_dict_backbone'] = model.state_dict()
                torch.save(to_save, self.checkpoint_path(trial, cycle))

            acc = self.test(model, dataloaders['test'])
            if cycle == 0:
                min_acc = acc
            print(f'{self.dataset} {self.model_name}. sampling:{self.sampling} Trial:{trial+1}/{self.trials} || Cycle:{cycle+1}/{self.cycles} || Label set size:{len(labeled_set)} ||  Test acc:{acc:.2f}', flush=True)
            results.append([self.dataset, AUXILIARY, self.sampling, self.model_name, trial + 1, cycle + 1, len(labeled_set), acc])

            # Active sampling
            np.random.RandomState(cycle + trial * 1000).shuffle(unlabeled_set)
            if cycle == self.cycles - 1:
                pd.DataFrame(results).to_csv(self.results_path(trial))
                continue

            subset = query(model, dataloaders, self.sampling, labeled_set, unlabeled_set, self.config, acc, min_acc)
            labeled_set += subset
            unlabeled_set = [x for x in unlabeled_set if x not in set(subset)]

            # Create a new dataloader for the updated labeled dataset
            dataloaders['train'].sampler.indices = labeled_set
            dataloaders['unlabeled'].sampler.indices = unlabeled_set

def _to_list(x):
    if isinstance(x, list):
        return x
    return [x]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Active Learning')
    parser.add_argument('--dataset', default='cifar10', type=str, nargs='+',
                        choices=['cifar10', 'cifar100', 'svhn', 'yelp', 'yelp5', 'yahoo', 'imdb', 'ag'])
    parser.add_argument('--qs', default='RANDOM', type=str, help='query strategy', nargs='+')
    parser.add_argument('--model', default='resnet', type=str, choices=['resnet', 'vgg', 'textEmb', 'text'], nargs='+')
    parser.add_argument('--cycles', default=10, type=int)
    parser.add_argument('--trials', default=1, type=int, help='number of restarts')
    parser.add_argument('--start', default=0, type=int, help='start trial')
    parser.add_argument('--work-dir', default='.', type=str)
    args = parser.parse_args()
    runner = Runner(_to_list(args.dataset), _to_list(args.model), _to_list(args.qs), args.trials, args.start, args.cycles, args.work_dir)
    runner.run()

