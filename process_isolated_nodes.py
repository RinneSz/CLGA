from torch_geometric.utils import contains_isolated_nodes, remove_isolated_nodes


def process_isolated_nodes(edge_index):
    if contains_isolated_nodes(edge_index):
        new_edge_index, _, mask = remove_isolated_nodes(edge_index)
        mapping = {}
        for i in range(edge_index.shape[1]):
            if edge_index[0, i] != new_edge_index[0, i]:
                mapping[new_edge_index[0, i].item()] = edge_index[0, i].item()
        return new_edge_index, mapping, mask
    else:
        return edge_index, None, None


def restore_isolated_ndoes(new_edge_index, mapping):
    for i in range(new_edge_index.shape[1]):
        if new_edge_index[0, i].item() in mapping:
            new_edge_index[0, i] = mapping[new_edge_index[0, i].item()]
        if new_edge_index[1, i].item() in mapping:
            new_edge_index[1, i] = mapping[new_edge_index[1, i].item()]
    return new_edge_index
