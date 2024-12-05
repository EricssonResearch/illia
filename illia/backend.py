# Available backends for illia
_AVAILABLE_DNN_BACKENDS = ["torch", "tf", "jax"]
_AVAILABLE_GNN_BACKENDS = ["torch_geometric", "dgl", "spektral"]


def show_available_backends():
    """
    This function prints the available deep neural network (DNN) and graph neural network (GNN) backends.
    """

    print("Available backends for DNN: ")
    for i, dnn_backend in enumerate(_AVAILABLE_DNN_BACKENDS):
        print(f"\t{i+1}. {dnn_backend}.")

    print("\n")

    print("Available backends for GNN: ")
    for i, gnn_backend in enumerate(_AVAILABLE_GNN_BACKENDS):
        print(f"\t{i+1}. {gnn_backend}.")
