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
    r"""
    Crystal Graph Convolutional operator for material property prediction.

    Updates node features using neighboring nodes and edge features as:

        x'_i = x_i + sum_{j in N(i)} sigmoid(z_ij W_f + b_f) *
               softplus(z_ij W_s + b_s)

    where z_ij is the concatenation of central node features, neighbor
    features, and edge features. Applies element-wise sigmoid and
    softplus functions.

    Args:
        channels: Size of input features. If tuple, represents source and
            target feature dimensions.
        dim: Dimensionality of edge features.
        aggr: Aggregation method ("add", "mean", "max").
        **kwargs: Additional arguments for MessagePassing.

    Returns:
        None.

    Shapes:
        - input: node features (|V|, F) or ((|Vs|, Fs), (|Vt|, Ft)) if
          bipartite, edge indices (2, |E|), edge features (|E|, D) optional.
        - output: node features (|V|, F) or (|Vt|, Ft) if bipartite.

    Notes:
        Based on "Crystal Graph Convolutional Neural Networks for an
        Accurate and Interpretable Prediction of Material Properties"
        (https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301)
    """

    def __init__(
        self,
        channels: int | tuple[int, int],
        dim: int = 0,
        aggr: str = "add",
        **kwargs: Any,
    ) -> None:
        """
        Initializes the CGConv layer with linear transformations.

        Args:
            channels: Size of input features. Tuple for source and target.
            dim: Dimensionality of edge features.
            aggr: Aggregation operator ("add", "mean", "max").
            **kwargs: Extra arguments passed to the base class.

        Returns:
            None.
        """

        # Call super class constructor
        super().__init__(aggr=aggr, **kwargs)

        # Set attributes
        self.channels = channels
        self.dim = dim

        if isinstance(channels, int):
            channels = (channels, channels)

        # Define linear layers
        self.lin_f = Linear(sum(channels) + dim, channels[1])
        self.lin_s = Linear(sum(channels) + dim, channels[1])

    def reset_parameters(self) -> None:
        """
        Resets parameters of the linear layers and optional batch norm.

        Returns:
            None.
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
            x: Input node features, as a single tensor or a pair if bipartite.
            edge_index: Edge indices.
            edge_attr: Optional edge features.

        Returns:
            Node features after applying the convolution.
        """

        if isinstance(x, Tensor):
            x = (x, x)

        # Propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        out = out + x[1]
        return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: OptTensor) -> Tensor:
        """
        Constructs messages passed to neighboring nodes.

        Args:
            x_i: Central node features.
            x_j: Neighboring node features.
            edge_attr: Optional edge features.

        Returns:
            Aggregated messages for neighbors.
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

        Returns:
            String with class name, channels, and edge feature dimension.
        """

        return f"{self.__class__.__name__}({self.channels}, dim={self.dim})"
