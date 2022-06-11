from typing import Optional

import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, structured_negative_sampling
from sklearn.metrics import roc_auc_score

from pGRACE.model import LogReg


def get_idx_split(data, split, preload_split):
    if split[:4] == 'rand':
        train_ratio = float(split.split(':')[1])
        num_nodes = data.x.size(0)
        train_size = int(num_nodes * train_ratio)
        indices = torch.randperm(num_nodes)
        return {
            'train': indices[:train_size],
            'val': indices[train_size:2 * train_size],
            'test': indices[2 * train_size:]
        }
    elif split.startswith('cora') or split.startswith('citeseer'):
        return {
            'train': data.train_mask,
            'test': data.test_mask,
            'val': data.val_mask
        }
    elif split == 'preloaded':
        assert preload_split is not None, 'use preloaded split, but preloaded_split is None'
        train_mask, test_mask, val_mask = preload_split
        return {
            'train': train_mask,
            'test': test_mask,
            'val': val_mask
        }
    else:
        raise RuntimeError(f'Unknown split type {split}')


def log_regression(z,
                   data,
                   evaluator,
                   num_epochs: int = 5000,
                   test_device: Optional[str] = None,
                   split: str = 'rand:0.1',
                   verbose: bool = False,
                   preload_split=None,
                   ):
    test_device = z.device if test_device is None else test_device
    z = z.detach().to(test_device)
    num_hidden = z.size(1)
    y = data.y.view(-1).to(test_device)
    num_classes = data.y.max().item() + 1
    classifier = LogReg(num_hidden, num_classes).to(test_device)
    optimizer = Adam(classifier.parameters(), lr=0.01, weight_decay=0.0)

    split = get_idx_split(data, split, preload_split)
    split = {k: v.to(test_device) for k, v in split.items()}
    f = nn.LogSoftmax(dim=-1)
    nll_loss = nn.NLLLoss()

    best_test_acc = 0
    best_val_acc = 0
    best_epoch = 0

    for epoch in range(num_epochs):
        classifier.train()
        optimizer.zero_grad()

        output = classifier(z[split['train']])
        loss = nll_loss(f(output), y[split['train']])

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            if 'val' in split:
                # val split is available
                test_acc = evaluator.eval({
                    'y_true': y[split['test']].view(-1, 1),
                    'y_pred': classifier(z[split['test']]).argmax(-1).view(-1, 1)
                })['acc']
                val_acc = evaluator.eval({
                    'y_true': y[split['val']].view(-1, 1),
                    'y_pred': classifier(z[split['val']]).argmax(-1).view(-1, 1)
                })['acc']
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    best_epoch = epoch
            else:
                acc = evaluator.eval({
                    'y_true': y[split['test']].view(-1, 1),
                    'y_pred': classifier(z[split['test']]).argmax(-1).view(-1, 1)
                })['acc']
                if best_test_acc < acc:
                    best_test_acc = acc
                    best_epoch = epoch
            if verbose:
                print(f'logreg epoch {epoch}: best test acc {best_test_acc}')

    return {'acc': best_test_acc, 'model': classifier, 'split': split}


class MulticlassEvaluator:
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def _eval(y_true, y_pred):
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)
        total = y_true.size(0)
        correct = (y_true == y_pred).to(torch.float32).sum()
        return (correct / total).item()

    def eval(self, res):
        return {'acc': self._eval(**res)}


class LPEvaluator:
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def _eval(scores, negative_edge_index, target_edge_index):
        edge_index = torch.cat([negative_edge_index, target_edge_index], -1)
        ranking_scores = scores[edge_index[0], edge_index[1]]
        ranking_labels = torch.cat([torch.zeros(negative_edge_index.shape[1]), torch.ones(target_edge_index.shape[1])]).to(scores.device)
        auc = roc_auc_score(ranking_labels.detach().cpu().numpy(), ranking_scores.detach().cpu().numpy())
        return auc

    def eval(self, res):
        return self._eval(**res)


def link_prediction(z,
                    edge_index,
                    train_edge_index,
                    val_edge_index,
                    test_edge_index,
                    num_nodes,
                    evaluator,
                    num_epochs: int = 5000,
                    test_device: Optional[str] = None,
                    verbose: bool = False,
                    ):
    test_device = z.device if test_device is None else test_device
    z = z.detach().to(test_device)
    num_hidden = z.size(1)
    observed_edge_sp_adj = torch.sparse.FloatTensor(edge_index,
                                                    torch.ones(edge_index.shape[1]).to(test_device),
                                                    [num_nodes, num_nodes])
    observed_edge_adj = observed_edge_sp_adj.to_dense().to(test_device)
    negative_edges = 1 - observed_edge_adj - torch.eye(num_nodes).to(test_device)
    negative_edge_index = torch.nonzero(negative_edges).t()

    projecter = LogReg(num_hidden, num_hidden).to(test_device)
    optimizer = Adam(projecter.parameters(), lr=0.01, weight_decay=0.0)

    best_test_auc = 0
    best_val_auc = 0
    for epoch in range(num_epochs):
        projecter.train()
        optimizer.zero_grad()

        output = projecter(z)
        output = F.normalize(output)
        scores = torch.mm(output, output.t())

        edge_index_with_self_loops = add_self_loops(train_edge_index)[0]
        train_u, train_i, train_j = structured_negative_sampling(edge_index_with_self_loops, num_nodes)
        train_u = train_u[:train_edge_index.shape[1]]
        train_i = train_i[:train_edge_index.shape[1]]
        train_j = train_j[:train_edge_index.shape[1]]
        loss = -torch.log(torch.sigmoid(scores[train_u, train_i] - scores[train_u, train_j])).sum()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            test_auc = evaluator.eval({
                'scores': scores,
                'negative_edge_index': negative_edge_index,
                'target_edge_index': test_edge_index
            })
            val_auc = evaluator.eval({
                'scores': scores,
                'negative_edge_index': negative_edge_index,
                'target_edge_index': val_edge_index
            })
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_test_auc = test_auc
            if verbose:
                print(f'logreg epoch {epoch}: best test acc {best_test_auc}')

    return {'auc': best_test_auc, 'model': projecter}
