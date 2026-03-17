import torch
import torch.nn as nn

class LowRankCRFHead(nn.Module):
    def __init__(self, vocab_size: int, d: int = 32):
        """
        Initializes the Low-Rank CRF Head.
        
        Args:
            vocab_size (int): The size of the vocabulary |V|.
            d (int): The dimension of the low-rank embeddings. Default is 32.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d = d
        
        # Initialize two embedding matrices E1 (outgoing) and E2 (incoming)
        self.E1 = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d)
        self.E2 = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d)

    def forward(self, prev_tokens: torch.Tensor, curr_tokens: torch.Tensor, pairwise: bool = True) -> torch.Tensor:
        """
        Calculates the low-rank transition scores between y_{t-1} and y_t.
        
        Args:
            prev_tokens (torch.Tensor): Tensor of previous tokens y_{t-1}. Shape: (batch_size, ..., N)
            curr_tokens (torch.Tensor): Tensor of current tokens y_t. Shape: (batch_size, ..., M)
            pairwise (bool): If True, computes an N x M matrix of scores for all pairs between 
                             the candidate sets in prev_tokens and curr_tokens.
                             If False, computes element-wise scores (assumes prev/curr shapes match).
                             
        Returns:
            torch.Tensor: Transition scores.
                          If pairwise=True: Shape is (batch_size, ..., N, M)
                          If pairwise=False: Shape is (batch_size, ..., N)
        """
        # Embed the tokens
        # e1 shape: (batch_size, ..., N, d) 
        e1 = self.E1(prev_tokens)
        
        # e2 shape: (batch_size, ..., M, d)
        e2 = self.E2(curr_tokens)
        
        if pairwise:
            # We want to compute transitions from all N candidates to all M candidates.
            # Highly vectorized Batched Matrix Multiplication:
            # (batch_size, ..., N, d) @ (batch_size, ..., d, M) -> (batch_size, ..., N, M)
            scores = torch.matmul(e1, e2.transpose(-1, -2))
        else:
            # Element-wise dot product for exact path scoring (e.g., ground truth sequence)
            # Shapes of prev_tokens and curr_tokens must match.
            # (batch_size, ..., d) * (batch_size, ..., d) -> sum over d -> (batch_size, ...)
            scores = (e1 * e2).sum(dim=-1)
            
        return scores
