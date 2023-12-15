import scipy.sparse as sp
import torch
import numpy as np
from torch import Tensor, nn

from networks.predictor import Predictor
from networks.graph_learner import GraphLearner
from utils import load_graph_data


def create_model(dataset: str, model_config: dict, adaptor_config: dict, device):
    supports = load_graph_data(dataset, 'doubletransition')
    # supports = torch.tensor(list(map(sp.coo_matrix.toarray, supports)), dtype=torch.float32, device=device)
    supports_dense = [s.toarray() for s in supports]
    supports_array = np.stack(supports_dense, axis=0)  # 首先将列表转换为单一的 numpy 数组
    supports = torch.tensor(supports_array, dtype=torch.float32, device=device)  # 然后将 numpy 数组转换为张量

    edge_dim = supports.size(0)

    graph_learner = GraphLearner(supports, **adaptor_config)
    predictor = Predictor(edge_dim=edge_dim, **model_config)

    return Model(predictor, graph_learner)


class Model(nn.Module):
    def __init__(self, predictor: Predictor, graph_learner: GraphLearner):
        super(Model, self).__init__()
        self.predictor = predictor
        self.graph_learner = graph_learner

    def forward(self, inputs: Tensor) -> Tensor:
        supports = self.graph_learner(inputs)
        outputs = self.predictor(inputs, supports)
        return outputs
