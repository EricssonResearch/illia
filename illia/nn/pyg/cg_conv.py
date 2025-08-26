"""
This module contains the code for the Crystal Graph Convolutional
operator.
"""

# Standard libraries
from typing import Any

# 3pps
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor

# Own modules
from illia.nn.torch.linear import Linear


class CGConv(MessagePassing):
    """
    Definition of the Crystal Graph Convolutional operator.
    """

    def __init__(
        self,
        channels: int | tuple[int, int],
        dim: int = 0,
        aggr: str = "add",
        **kwargs: Any,
    ) -> None:
        r"""
        "Crystal Graph Convolutional Neural Networks for an Accurate
        and Interpretable Prediction of Material Properties"
        (https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301).

        The operation is defined as:

        $$
        \mathbf{x}^{\prime}_i = \mathbf{x}_i + \sum_{j \in
        \mathcal{N}(i)}\sigma ( \mathbf{z}_{i,j} \mathbf{W}_f +
        \mathbf{b}_f )
        \odot g ( \mathbf{z}_{i,j} \mathbf{W}_s + \mathbf{b}_s )
        $$

        where \(\mathbf{z}_{i,j} = [ \mathbf{x}_i, \mathbf{x}_j,
        \mathbf{e}_{i,j} ]\)
        denotes the concatenation of central node features, neighboring
        node features, and edge features. In addition, \(\sigma\) and
        \(g\) denote the sigmoid and softplus functions, respectively.

        Args:
            channels (int or tuple): The size of each input sample. A
                tuple corresponds to the sizes of source and target
                dimensionalities.
            dim (int, optional): The edge feature dimensionality.
            aggr (str, optional): The aggregation operator to use
                ("add", "mean", "max").
            **kwargs (optional): Additional arguments for
                :class:`torch_geometric.nn.conv.MessagePassing`.

        Shapes:
            - **input:**
            node features \((|\mathcal{V}|, F)\) or \(((|\mathcal{V_s}|,
            F_{s}), (|\mathcal{V_t}|, F_{t}))\) if bipartite,
            edge indices \((2, |\mathcal{E}|)\),
            edge features \((|\mathcal{E}|, D)\) *(optional)*
            - **output:** node features \((|\mathcal{V}|, F)\) or
                \((|\mathcal{V_t}|, F_{t})\) if bipartite
        """

        # Call super class constructor
        super().__init__(aggr=aggr, **kwargs)

        self.channels = channels
        self.dim = dim

        if isinstance(channels, int):
            channels = (channels, channels)

        self.lin_f = Linear(sum(channels) + dim, channels[1])
        self.lin_s = Linear(sum(channels) + dim, channels[1])

    def reset_parameters(self) -> None:
        """
        Resets the parameters of the linear layers.
        """

        self.lin_f.reset_parameters()
        self.lin_s.reset_parameters()
        if self.bn is not None:
            self.bn.reset_parameters()

    def forward(
        self, x: Tensor | PairTensor, edge_index: Adj, edge_attr: OptTensor = None
    ) -> Tensor:
        """
        Performs a forward pass of the convolutional layer.

        Args:
            x: Input node features, either as a single tensor or a pair
                of tensors if bipartite.
            edge_index: Edge indices.
            edge_attr: Optional edge features.

        Returns:
            The output node features.
        """

        if isinstance(x, Tensor):
            x = (x, x)

        # Propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        out = out + x[1]
        return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: OptTensor) -> Tensor:
        """
        Constructs messages to be passed to neighboring nodes.

        Args:
            x_i: Central node features.
            x_j: Neighboring node features.
            edge_attr: Optional edge features.

        Returns:
            The messages to be aggregated.
        """

        if edge_attr is None:
            z = torch.cat([x_i, x_j], dim=-1)
        else:
            z = torch.cat([x_i, x_j, edge_attr], dim=-1)

        # pylint: disable=E1102
        return self.lin_f(z).sigmoid() * F.softplus(input=self.lin_s(z))

    def __repr__(self) -> str:
        """
        Returns a string representation of the module.
        """

        return f"{self.__class__.__name__}({self.channels}, dim={self.dim})"
