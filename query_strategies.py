# General
import numpy as np
# Torch
import torch
from scipy import stats
from sklearn.metrics import pairwise_distances, pairwise_distances_chunked
from sklearn.cluster import AgglomerativeClustering
from collections import Counter, defaultdict
MAX_ACCURACIES = {'cifar10': 90, 'svhn': 96, 'imdb': 87, 'yelp': 91.5, 'yelp5': 53, 'yahoo': 65.5, 'ag': 91}


def furthest_first(X, labeled_set, n, temperature='inf', use_labeled=True, dist_ctr=None):
    if dist_ctr is None:
        if X.shape[0] > 30000:
            X = X.astype('float32')
            iterator = pairwise_distances_chunked(X, working_memory=2048)
            dist_ctr = np.zeros((X.shape[0], X.shape[0]), dtype='float32')
            start = 0
            for chunk in iterator:
                l = chunk.shape[0]
                finish = start + l
                dist_ctr[start:finish] = chunk
                start = finish
                del chunk
        else:
            dist_ctr = pairwise_distances(X)

    if use_labeled:
        min_dist = np.amin(dist_ctr[:, labeled_set], axis=1).astype('float64')
    else:
        min_dist = np.ones(X.shape[0]) * 1e10
        min_dist[labeled_set] = 0

    chosen_indices = []
    for i in range(n):
        if temperature != 'inf':
            Ddist = (min_dist ** temperature) / sum(min_dist ** temperature)
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(Ddist)), Ddist))
            idx = customDist.rvs(size=1)[0]
        else:
            idx = min_dist.argmax()

        chosen_indices.append(idx)
        dist_new_ctr = dist_ctr[:, idx].astype('float64')
        min_dist = np.minimum(min_dist, dist_new_ctr)
    del dist_ctr
    return chosen_indices


def get_probs_embeddings(model, loader):
    model.eval()
    X = []
    Y = []
    activation = torch.nn.Softmax(dim=1) if model.linear.out_features > 1 else torch.nn.Sigmoid()
    with torch.no_grad():
        for (inputs, labels) in loader:
            inputs = inputs.cuda()
            scores, embeddings = model(inputs)
            X.append(embeddings.cpu().numpy())

            Y.append(activation(scores).cpu().numpy())
    if model.linear.out_features == 1:
        Y = np.vstack(Y)
        return np.hstack([1 - Y, Y]), np.vstack(X)
    return np.vstack(Y), np.vstack(X)


def get_grad_embeddings(model, loader):
    probs, embs = get_probs_embeddings(model, loader)
    dimension = embs.shape[1]
    classes = probs.shape[1]
    if classes == 2:
        grad_embeddings = embs * (1 - np.max(probs, axis=1))[:, np.newaxis]
        grad_embeddings = grad_embeddings * (2 * np.argmax(probs, axis=1) - 1)[:, np.newaxis]
        return grad_embeddings, probs

    max_idx = np.argmax(probs, 1)
    grad_embeddings = np.zeros((embs.shape[0], classes * dimension))

    for j in range(embs.shape[0]):
        for c in range(classes):
            grad_embeddings[j, dimension * c: dimension * (c + 1)] = embs[j] * (1 * (c == max_idx[j]) - probs[j][c])

    return grad_embeddings, probs


def get_uncertainty(model, loader, sampling):
    probs, _ = get_probs_embeddings(model, loader)
    if sampling == 'MARGIN_1':
        uncertainty = (1 - np.max(probs, axis=1))
    elif sampling == 'MARGIN_2':
        p = np.sort(probs, axis=1)
        uncertainty = 1 - (p[:, -1] - p[:, -2])
    else:
        uncertainty = stats.entropy(probs, axis=1)
    return uncertainty


class Selector:
    K_M_MULTIPLIER = 1.25

    def __init__(self, sampling, config):
        self.sampling = sampling
        self.config = config
        self.cluster_mapping = None
        self.batch_size = self.config.STEP_SIZE
        self.k_m = int(np.ceil(self.K_M_MULTIPLIER*self.config.STEP_SIZE))

    def HAC(self, model, dataloaders, unlabeled_set):
        probs, X = get_probs_embeddings(model, dataloaders['all'])
        if self.cluster_mapping is None:
            alg = AgglomerativeClustering(n_clusters=X.shape[0]//self.K_M_MULTIPLIER, linkage='average').fit(X)
            self.cluster_mapping = alg.labels_
        unlabeled_probs = probs[np.array(unlabeled_set)]
        unlabeled_probs = np.sort(unlabeled_probs, axis=1)
        uncertainty = 1 - (unlabeled_probs[:, -1] - unlabeled_probs[:, -2])
        unc_indices = [unlabeled_set[i] for i in np.argsort(uncertainty)[-self.k_m:]]  # k_m examples with the highest unc
        original_indices = [unlabeled_set[i] for i in unc_indices]

        clusters = {i: self.cluster_mapping[i] for i in original_indices}
        inverse_mapping = defaultdict(list)
        for index, cluster in clusters.items():
            inverse_mapping[cluster].append(index)
        cluster_sizes = Counter(clusters)
        cluster_order = sorted(cluster_sizes.keys(), key=lambda x: cluster_sizes[x])
        selected = []
        cluster_index = 0
        while len(selected) < self.batch_size:
            if cluster_index == len(cluster_order):
                cluster_index = 0
            cluster_name = cluster_order[cluster_index]
            cluster_examples = inverse_mapping[cluster_name]
            selected.append(cluster_examples.pop(0))
            if len(cluster_examples) == 0:
                cluster_order.pop(cluster_index)
            else:
                cluster_index += 1
        return selected

    def query(self, model, dataloaders, labeled_set, unlabeled_set, acc, min_acc):
        sampling = self.sampling
        if sampling == 'RANDOM':
            subset = unlabeled_set[:self.batch_size]
        elif sampling == 'HAC':
            subset = self.HAC(model, dataloaders, unlabeled_set)
        elif sampling == 'ENTROPY' or sampling.startswith('MARGIN'):
            uncertainty = get_uncertainty(model, dataloaders['unlabeled'], sampling)
            arg = np.argsort(uncertainty)
            subset = [unlabeled_set[i] for i in arg[-self.batch_size:]]
        else:
            if sampling.startswith('BADGE'):
                X, probs = get_grad_embeddings(model, dataloaders['all'])
                temperature = 'inf' if 'NO_SAMPLING' in sampling else 2
                use_labeled = 'USE_LABELED' in sampling
            else:
                probs, X = get_probs_embeddings(model, dataloaders['all'])
                temperature = 'inf'
                use_labeled = True

            if sampling.startswith('CORESET_SCALED'):
                X = X * (1 - probs.max(axis=1).reshape(-1, 1))
                temperature = 2 if 'SAMPLING' in sampling else 'inf'
                use_labeled = not 'NO_LABELED' in sampling

            if 'TEMP=' in sampling:
                if 'linear' in sampling or 'squared' in sampling or 'log' in sampling:
                    max = 100 if '100' in sampling else MAX_ACCURACIES[self.config.DATASET]
                    min = 100 / self.config.CLASS if 'class' in sampling else min_acc
                    temperature = (max - min) / (max - acc)
                    if 'squared' in sampling:
                        temperature = (max ** 2 - min ** 2) / (max ** 2 - acc ** 2)
                    if 'log' in sampling:
                        temperature = np.log(max / min) / np.log(max / acc)

                else:
                    temperature = float(sampling[sampling.find('TEMP=') + 5:])

                if 'double' in sampling:
                    temperature = temperature * 2

                if temperature > 20 or temperature < 0:
                    temperature = 'inf'

            print('temperature', temperature, 'use_labeled', use_labeled)
            subset = furthest_first(X, labeled_set, self.batch_size, temperature=temperature,
                                    use_labeled=use_labeled)
        return subset

