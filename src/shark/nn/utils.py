# pylint: disable=invalid-name
__all__ = ["mixture_bernoulli_loss", "gnn_block"]

import typing as ty
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


def gnn_block(
    dim_in: int,
    dim_out: int,
    start: str = None,
    n_layers: int = 1,
    dropout: float = 0.5,
) -> ty.Sequence[ty.Tuple[torch.nn.Module, str]]:
    """GNN block."""
    blocks: ty.List[ty.Tuple[torch.nn.Module, str]] = []
    for idx in range(1, n_layers + 1):
        blocks += _gnn_block(dim_in, dim_out, start, idx, dropout)
    return blocks


def _gnn_block(
    dim_in: int,
    dim_out: int,
    start: str = None,
    idx: int = 1,
    dropout: float = 0.5,
) -> ty.Sequence[ty.Tuple[torch.nn.Module, str]]:
    """GNN block."""
    assert idx > 0
    if start is None:
        start = "x" if idx < 2 else f"x{idx-1}d"
    block = [
        (GCNConv(dim_in, dim_out), f"{start}, edge_index -> x{idx}"),
        (torch.nn.ReLU(), f"x{idx} -> x{idx}a"),
        (torch.nn.Dropout(p=dropout), f"x{idx}a -> x{idx}d"),
    ]
    return block


def mixture_bernoulli_loss(
    label: torch.Tensor,
    log_theta: torch.Tensor,
    log_alpha: torch.Tensor,
    adj_loss_func: ty.Callable,
    subgraph_idx: torch.Tensor,
    subgraph_idx_base: torch.Tensor,
    num_canonical_order: int,
    sum_order_log_prob: bool = False,
    reduction: str = "mean",
) -> ty.Sequence[torch.Tensor]:
    """
    Compute likelihood for mixture of Bernoulli model
    Args:
        label: (E,1), see comments above
        log_theta: (E,D), see comments above
        log_alpha: (E,D), see comments above
        adj_loss_func: BCE loss
        subgraph_idx: (E,1), see comments above
        subgraph_idx_base: (B+1), cumulative # of edges in the subgraphs associated with each batch
        num_canonical_order: int, number of node orderings considered
        sum_order_log_prob: boolean, if True sum the log prob of orderings instead of taking logsumexp
            i.e. log p(G, pi_1) + log p(G, pi_2) instead of log [p(G, pi_1) + p(G, pi_2)]
            This is equivalent to the original GRAN loss.
        return_neg_log_prob: boolean, if True also return neg log prob
        reduction: string, type of reduction on batch dimension ("mean", "sum", "none")
    Returns:
        loss (and potentially neg log prob)
    """
    num_subgraph: int = int(subgraph_idx_base[-1])  # == subgraph_idx.max() + 1
    B = subgraph_idx_base.size(0) - 1
    C = num_canonical_order
    E = log_theta.size(0)
    K = log_theta.size(1)
    assert E % C == 0
    adj_loss = torch.stack([adj_loss_func(log_theta[:, kk].unsqueeze(-1), label) for kk in range(K)], dim=0)
    const = torch.zeros(num_subgraph).to(label.device).view(-1, 1)  # S
    const = const.scatter_add(0, subgraph_idx, torch.ones_like(subgraph_idx).float())
    reduce_adj_loss = torch.zeros(num_subgraph, K).to(label.device)
    reduce_adj_loss = reduce_adj_loss.scatter_add(0, subgraph_idx.unsqueeze(1).expand(-1, K), adj_loss)
    reduce_log_alpha = torch.zeros(num_subgraph, K).to(label.device)
    reduce_log_alpha = reduce_log_alpha.scatter_add(0, subgraph_idx.unsqueeze(1).expand(-1, K), log_alpha)
    reduce_log_alpha = reduce_log_alpha / const.view(-1, 1)
    reduce_log_alpha = F.log_softmax(reduce_log_alpha, -1)
    log_prob = -reduce_adj_loss + reduce_log_alpha
    log_prob = torch.logsumexp(log_prob, dim=1)  # S, K
    bc_log_prob = torch.zeros([B * C]).to(label.device)  # B*C
    bc_idx = torch.arange(B * C).to(label.device)  # B*C
    bc_const = torch.zeros(B * C).to(label.device)
    bc_size = (subgraph_idx_base[1:] - subgraph_idx_base[:-1]) // C  # B
    bc_size = torch.repeat_interleave(bc_size, C)  # B*C
    bc_idx = torch.repeat_interleave(bc_idx, bc_size)  # S
    bc_log_prob = bc_log_prob.scatter_add(0, bc_idx, log_prob)
    # loss must be normalized for numerical stability
    bc_const = bc_const.scatter_add(0, bc_idx, const)
    bc_loss = bc_log_prob / bc_const
    bc_log_prob = bc_log_prob.reshape(B, C)
    bc_loss = bc_loss.reshape(B, C)
    if sum_order_log_prob:
        b_log_prob = torch.sum(bc_log_prob, dim=1)
        b_loss = torch.sum(bc_loss, dim=1)
    else:
        b_log_prob = torch.logsumexp(bc_log_prob, dim=1)
        b_loss = torch.logsumexp(bc_loss, dim=1)
    # probability calculation was for lower-triangular edges
    # must be squared to get probability for entire graph
    b_neg_log_prob = -2 * b_log_prob
    b_loss = -b_loss
    if reduction.lower() == "mean":
        neg_log_prob = b_neg_log_prob.mean()
        loss = b_loss.mean()
    elif reduction.lower() == "sum":
        neg_log_prob = b_neg_log_prob.sum()
        loss = b_loss.sum()
    elif reduction.lower() == "none":
        neg_log_prob = b_neg_log_prob
        loss = b_loss
    else:
        raise ValueError("Unsupported reduction method. Supported methods are: 'mean', 'sum' or 'none'.")
    return loss, neg_log_prob
