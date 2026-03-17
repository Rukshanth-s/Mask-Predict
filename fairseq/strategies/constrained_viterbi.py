import torch
import torch.nn as nn

def constrained_viterbi_search(
    emissions: torch.Tensor,
    crf_head: nn.Module,
    left_context_token: torch.Tensor,
    right_context_token: torch.Tensor,
    k: int = 64
) -> torch.Tensor:
    """
    Constrained Viterbi Search with Beam Approximation for replacing greedy argmax
    during Mask-Predict decoding.

    Args:
        emissions: Emission logits from the transformer [batch_size, segment_len, vocab_size]
        crf_head: LowRankCRFHead module measuring transition matrix E1(prev) @ E2(curr)
        left_context_token: Unmasked token to the left of the segment [batch_size, 1]
        right_context_token: Unmasked token to the right of the segment [batch_size, 1]
        k: The beam size (truncation threshold) for Viterbi decoding

    Returns:
        decoded_tokens: The optimal token IDs for the masked segment [batch_size, segment_len]
    """
    batch_size, segment_len, vocab_size = emissions.shape
    
    # 1. Beam Truncation
    # Shape of topk_emissions/topk_indices: [batch_size, segment_len, k]
    topk_emissions, topk_indices = torch.topk(emissions, k, dim=-1)
    
    backpointers = []
    
    # --- Step 0: Viterbi initialization from Left Clamped Node ---
    # Transition from left_context_token (k=1) to candidates at t=0 (k=k)
    # Output shape: [batch_size, 1, k] -> squeezed to [batch_size, k]
    transitions_0 = crf_head(left_context_token, topk_indices[:, 0, :], pairwise=True).squeeze(1)
    
    # Alpha (cumulative path score) initialization
    alpha = transitions_0 + topk_emissions[:, 0, :] # [batch_size, k]
    
    # --- Step 1 to segment_len-1: Viterbi Forward Pass ---
    for t in range(1, segment_len):
        prev_candidates = topk_indices[:, t-1, :]
        curr_candidates = topk_indices[:, t, :]
        
        # Calculate full k x k transition block
        transitions = crf_head(prev_candidates, curr_candidates, pairwise=True) # [batch_size, k, k]
        
        # Add transition scores to the accumulated alpha history
        # alpha is broadcast over Dim=2 (current candidates direction)
        scores = alpha.unsqueeze(2) + transitions # [batch_size, k, k]
        
        # Marginalize (maximize) over Dim=1 (previous candidates direction)
        max_scores, best_prev_indices = torch.max(scores, dim=1) # [batch_size, k], [batch_size, k]
        
        # Add current node emissions
        alpha = max_scores + topk_emissions[:, t, :] # [batch_size, k]
        
        # Save historical indices to recover the path
        backpointers.append(best_prev_indices)
        
    # --- Final Step: Viterbi termination to Right Clamped Node ---
    # Transition from candidates at t=segment_len-1 (k=k) to right_context_token (k=1)
    # Output shape: [batch_size, k, 1] -> squeezed to [batch_size, k]
    final_transitions = crf_head(topk_indices[:, segment_len - 1, :], right_context_token, pairwise=True).squeeze(-1)
    
    # Complete Final Score calculation
    final_scores = alpha + final_transitions # [batch_size, k]
    
    # Find the top overall state at the end of the chain
    _, best_final_indices = torch.max(final_scores, dim=1) # [batch_size]
    
    # --- Viterbi Backward Pass ---
    # Backtrack starting from the best terminating candidate
    curr_best_idx = best_final_indices
    best_path_k_indices = [curr_best_idx]
    
    # Traverse backwards from segment_len-1 to step 1
    for bp in reversed(backpointers):
        # Extract the node from step t-1 that best led to step t
        curr_best_idx = torch.gather(bp, 1, curr_best_idx.unsqueeze(1)).squeeze(1) # [batch_size]
        best_path_k_indices.append(curr_best_idx)
        
    # Re-order to chronological array (Step 0 to segment_len-1)
    best_path_k_indices.reverse()
    
    # Reconstruct token sequence using the top k lookup map
    decoded_tokens = []
    for t in range(segment_len):
        idx = best_path_k_indices[t]
        tokens_t = torch.gather(topk_indices[:, t, :], 1, idx.unsqueeze(1)).squeeze(1) # [batch_size]
        decoded_tokens.append(tokens_t)
        
    return torch.stack(decoded_tokens, dim=1)
