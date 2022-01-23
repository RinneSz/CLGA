from torch.nn.modules.module import Module
import argparse
import os.path as osp
import time
import pickle as pkl
import torch
from torch_geometric.utils import dropout_adj, degree, to_undirected, dense_to_sparse, is_undirected, to_networkx, contains_isolated_nodes
from simple_param.sp import SimpleParam
from pGRACE.functional import drop_feature, drop_edge_weighted, \
    degree_drop_weights, evc_drop_weights, pr_drop_weights, \
    feature_drop_weights, drop_feature_weighted, feature_drop_weights_dense
from pGRACE.utils import get_activation, compute_pr, eigenvector_centrality
from pGRACE.dataset import get_dataset
from differentiable_models.gcn import GCN
from differentiable_models.model import GRACE


class Metacl(Module):
    def __init__(self, args, dataset, param, device):
        super(Metacl, self).__init__()
        self.model = None
        self.optimizer = None
        self.param = param
        self.args = args
        self.device = device
        self.dataset = dataset
        self.data = dataset.data.to(device)
        self.drop_weights = None
        self.feature_weights = None

    def drop_edge(self, p):
        if self.param['drop_scheme'] == 'uniform':
            return dropout_adj(self.data.edge_index, p=p)[0]
        elif self.param['drop_scheme'] in ['degree', 'evc', 'pr']:
            return drop_edge_weighted(self.data.edge_index, self.drop_weights, p=p,
                                      threshold=0.7)
        else:
            raise Exception(f'undefined drop scheme: {param["drop_scheme"]}')

    def train_gcn(self):
        self.model.train()
        self.optimizer.zero_grad()
        edge_index_1 = self.drop_edge(self.param['drop_edge_rate_1'])
        edge_index_2 = self.drop_edge(self.param['drop_edge_rate_2'])
        x_1 = drop_feature(self.data.x, self.param['drop_feature_rate_1'])
        x_2 = drop_feature(self.data.x, self.param['drop_feature_rate_2'])
        edge_sp_adj_1 = torch.sparse.FloatTensor(edge_index_1,
                                                 torch.ones(edge_index_1.shape[1]).to(self.device),
                                                 [self.data.num_nodes, self.data.num_nodes])
        edge_sp_adj_2 = torch.sparse.FloatTensor(edge_index_2,
                                                 torch.ones(edge_index_2.shape[1]).to(self.device),
                                                 [self.data.num_nodes, self.data.num_nodes])
        if self.param['drop_scheme'] in ['pr', 'degree', 'evc']:
            x_1 = drop_feature_weighted(self.data.x, self.feature_weights, self.param['drop_feature_rate_1'])
            x_2 = drop_feature_weighted(self.data.x, self.feature_weights, self.param['drop_feature_rate_2'])
        z1 = self.model(x_1, edge_sp_adj_1, sparse=True)
        z2 = self.model(x_2, edge_sp_adj_2, sparse=True)
        loss = self.model.loss(z1, z2, batch_size=None)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def compute_drop_weights(self):
        if self.param['drop_scheme'] == 'degree':
            self.drop_weights = degree_drop_weights(self.data.edge_index).to(self.device)
        elif self.param['drop_scheme'] == 'pr':
            self.drop_weights = pr_drop_weights(self.data.edge_index, aggr='sink', k=200).to(self.device)
        elif self.param['drop_scheme'] == 'evc':
            self.drop_weights = evc_drop_weights(self.data).to(self.device)
        else:
            self.drop_weights = None

        if self.param['drop_scheme'] == 'degree':
            edge_index_ = to_undirected(self.data.edge_index)
            node_deg = degree(edge_index_[1])
            if self.args.dataset == 'WikiCS':
                self.feature_weights = feature_drop_weights_dense(self.data.x, node_c=node_deg).to(self.device)
            else:
                self.feature_weights = feature_drop_weights(self.data.x, node_c=node_deg).to(self.device)
        elif self.param['drop_scheme'] == 'pr':
            node_pr = compute_pr(self.data.edge_index)
            if self.args.dataset == 'WikiCS':
                self.feature_weights = feature_drop_weights_dense(self.data.x, node_c=node_pr).to(self.device)
            else:
                self.feature_weights = feature_drop_weights(self.data.x, node_c=node_pr).to(self.device)
        elif self.param['drop_scheme'] == 'evc':
            node_evc = eigenvector_centrality(self.data)
            if self.args.dataset == 'WikiCS':
                self.feature_weights = feature_drop_weights_dense(self.data.x, node_c=node_evc).to(self.device)
            else:
                self.feature_weights = feature_drop_weights(self.data.x, node_c=node_evc).to(self.device)
        else:
            self.feature_weights = torch.ones((self.data.x.size(1),)).to(self.device)

    def inner_train(self):
        encoder = GCN(self.dataset.num_features, self.param['num_hidden'], get_activation(self.param['activation']))
        self.model = GRACE(encoder, self.param['num_hidden'], self.param['num_proj_hidden'], self.param['tau']).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.param['learning_rate'],
            weight_decay=self.param['weight_decay']
        )
        self.compute_drop_weights()
        for epoch in range(1, self.param['num_epochs'] + 1):
            loss = self.train_gcn()

    def compute_gradient(self, pe1, pe2, pf1, pf2):
        self.model.eval()
        self.compute_drop_weights()
        edge_index_1 = self.drop_edge(pe1)
        edge_index_2 = self.drop_edge(pe2)
        x_1 = drop_feature(self.data.x, pf1)
        x_2 = drop_feature(self.data.x, pf2)
        edge_sp_adj_1 = torch.sparse.FloatTensor(edge_index_1,
                                                 torch.ones(edge_index_1.shape[1]).to(self.device),
                                                 [self.data.num_nodes, self.data.num_nodes])
        edge_sp_adj_2 = torch.sparse.FloatTensor(edge_index_2,
                                                 torch.ones(edge_index_2.shape[1]).to(self.device),
                                                 [self.data.num_nodes, self.data.num_nodes])
        if self.param['drop_scheme'] in ['pr', 'degree', 'evc']:
            x_1 = drop_feature_weighted(self.data.x, self.feature_weights, pf1)
            x_2 = drop_feature_weighted(self.data.x, self.feature_weights, pf2)
        edge_adj_1 = edge_sp_adj_1.to_dense()
        edge_adj_2 = edge_sp_adj_2.to_dense()
        edge_adj_1.requires_grad = True
        edge_adj_2.requires_grad = True
        z1 = self.model(x_1, edge_adj_1, sparse=False)
        z2 = self.model(x_2, edge_adj_2, sparse=False)
        loss = self.model.loss(z1, z2, batch_size=None)
        loss.backward()
        return edge_adj_1.grad, edge_adj_2.grad

    def attack(self):
        perturbed_edges = []
        num_total_edges = self.data.num_edges
        adj_sp = torch.sparse.FloatTensor(self.data.edge_index, torch.ones(self.data.edge_index.shape[1]).to(self.device),
                                          [self.data.num_nodes, self.data.num_nodes])
        adj = adj_sp.to_dense()

        print('Begin perturbing.....')
        # save three poisoned adj when the perturbation rate reaches 1%, 5%, 10%
        while len(perturbed_edges) < int(0.10 * num_total_edges):
            if len(perturbed_edges) == int(0.01 * num_total_edges) or \
                    len(perturbed_edges) == int(0.01 * num_total_edges)-1 or \
                    len(perturbed_edges) == int(0.01 * num_total_edges)-2:
                output_adj = adj.to(device)
                pkl.dump(output_adj.to(torch.device('cpu')), open('poisoned_adj/%s_CLGA_0.010000_adj.pkl' % args.dataset, 'wb'))
                print('---1% poisoned adjacency matrix saved---')
            if len(perturbed_edges) == int(0.05 * num_total_edges) or \
                    len(perturbed_edges) == int(0.05 * num_total_edges)-1 or \
                    len(perturbed_edges) == int(0.05 * num_total_edges)-2:
                output_adj = adj.to(device)
                pkl.dump(output_adj.to(torch.device('cpu')), open('poisoned_adj/%s_CLGA_0.050000_adj.pkl' % args.dataset, 'wb'))
                print('---5% perturbed adjacency matrix saved---')
            start = time.time()
            self.inner_train()
            adj_1_grad, adj_2_grad = self.compute_gradient(self.param['drop_edge_rate_1'], self.param['drop_edge_rate_2'], self.param['drop_feature_rate_1'], self.param['drop_feature_rate_2'])
            grad_sum = adj_1_grad + adj_2_grad
            grad_sum_1d = grad_sum.view(-1)
            grad_sum_1d_abs = torch.abs(grad_sum_1d)
            values, indices = grad_sum_1d_abs.sort(descending=True)
            i = -1
            while True:
                i += 1
                index = int(indices[i])
                row = int(index / self.data.num_nodes)
                column = index % self.data.num_nodes
                if [row, column] in perturbed_edges:
                    continue
                if grad_sum_1d[index] < 0 and adj[row, column] == 1:
                    adj[row, column] = 0
                    adj[column, row] = 0
                    perturbed_edges.append([row, column])
                    perturbed_edges.append([column, row])
                    break
                elif grad_sum_1d[index] > 0 and adj[row, column] == 0:
                    adj[row, column] = 1
                    adj[column, row] = 1
                    perturbed_edges.append([row, column])
                    perturbed_edges.append([column, row])
                    break
            self.data.edge_index = dense_to_sparse(adj)[0]
            end = time.time()
            print('Perturbing edges: %d/%d. Finished in %.2fs' % (len(perturbed_edges)/2, int(0.10 * num_total_edges)/2, end-start))
        print('Number of perturbed edges: %d' % (len(perturbed_edges)/2))
        output_adj = adj.to(device)
        return output_adj


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--param', type=str, default='local:general.json')
    default_param = {
        'learning_rate': 0.01,
        'num_hidden': 256,
        'num_proj_hidden': 32,
        'activation': 'prelu',
        'drop_edge_rate_1': 0.3,
        'drop_edge_rate_2': 0.4,
        'drop_feature_rate_1': 0.1,
        'drop_feature_rate_2': 0.0,
        'tau': 0.4,
        'num_epochs': 3000,
        'weight_decay': 1e-5,
        'drop_scheme': 'degree',
    }

    # add hyper-parameters into parser
    param_keys = default_param.keys()
    for key in param_keys:
        parser.add_argument(f'--{key}', type=type(default_param[key]), nargs='?')
    args = parser.parse_args()

    # parse param
    sp = SimpleParam(default=default_param)
    param = sp(source=args.param, preprocess='nni')

    # merge cli arguments and parsed param
    for key in param_keys:
        if getattr(args, key) is not None:
            param[key] = getattr(args, key)

    use_nni = args.param == 'nni'
    if use_nni and args.device != 'cpu':
        args.device = 'cuda'

    # torch_seed = args.seed
    # torch.manual_seed(torch_seed)
    # random.seed(12345)

    device = torch.device(args.device)

    path = osp.expanduser('dataset')
    path = osp.join(path, args.dataset)
    dataset = get_dataset(path, args.dataset)

    data = dataset[0]

    model = Metacl(args, dataset, param, device).to(device)
    poisoned_adj = model.attack()
    pkl.dump(poisoned_adj.to(torch.device('cpu')), open('poisoned_adj/%s_CLGA_0.100000_adj.pkl' % args.dataset, 'wb'))
    print('---10% perturbed adjacency matrix saved---')
