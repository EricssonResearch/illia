# Libraries
from typing import Tuple, Union

import torch
import torch.nn.functional as F  # type: ignore
from torch import Tensor
from torch_geometric.nn import MessagePassing  # type: ignore
from torch_geometric.typing import Adj, OptTensor, PairTensor  # type: ignore

from illia.torch.nn.linear import Linear


class CGConv(MessagePassing):
    """
    Definition of the Crystal Graph Convolutional operator.
    """

    def __init__(
        self,
        channels: Union[int, Tuple[int, int]],
        dim: int = 0,
        aggr: str = "add",
        **kwargs,
    ):
        r"""
        "Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties"
        (https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301).

        The operation is defined as:

        $$
        \mathbf{x}^{\prime}_i = \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)}\sigma ( \mathbf{z}_{i,j} \mathbf{W}_f + \mathbf{b}_f )
        \odot g ( \mathbf{z}_{i,j} \mathbf{W}_s + \mathbf{b}_s )
        $$

        where \(\mathbf{z}_{i,j} = [ \mathbf{x}_i, \mathbf{x}_j, \mathbf{e}_{i,j} ]\)
        denotes the concatenation of central node features, neighboring node features,
        and edge features. In addition, \(\sigma\) and \(g\) denote the sigmoid
        and softplus functions, respectively.

        Args:
            channels (int or tuple): The size of each input sample. A tuple corresponds to the sizes of source and target dimensionalities.
            dim (int, optional): The edge feature dimensionality. Defaults to 0.
            aggr (str, optional): The aggregation operator to use ("add", "mean", "max"). Defaults to "add".
            **kwargs (optional): Additional arguments for :class:`torch_geometric.nn.conv.MessagePassing`.

        Shapes:
            - **input:**
            node features \((|\mathcal{V}|, F)\) or \(((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))\) if bipartite,
            edge indices \((2, |\mathcal{E}|)\),
            edge features \((|\mathcal{E}|, D)\) *(optional)*
            - **output:** node features \((|\mathcal{V}|, F)\) or \((|\mathcal{V_t}|, F_{t})\) if bipartite
        """

        # Call super class constructor
        super(CGConv, self).__init__(aggr=aggr, **kwargs)

        self.channels = channels
        self.dim = dim

        if isinstance(channels, int):
            channels = (channels, channels)

        self.lin_f = Linear(sum(channels) + dim, channels[1])
        self.lin_s = Linear(sum(channels) + dim, channels[1])
        # self.reset_parameters()

    def reset_parameters(self):
        self.lin_f.reset_parameters()
        self.lin_s.reset_parameters()
        if self.bn is not None:
            self.bn.reset_parameters()

    def forward(
        self, x: Union[Tensor, PairTensor], edge_index: Adj, edge_attr: OptTensor = None
    ) -> Tensor:
        if isinstance(x, Tensor):
            x = (x, x)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        out = out + x[1]
        return out

    def message(self, x_i, x_j, edge_attr: OptTensor) -> Tensor:
        if edge_attr is None:
            z = torch.cat([x_i, x_j], dim=-1)
        else:
            z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.lin_f(z).sigmoid() * F.softplus(self.lin_s(z))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.channels}, dim={self.dim})"