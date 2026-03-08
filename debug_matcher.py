import torch
import torch.nn.functional as F

def test_matcher_shapes():
    # Simulate setup
    bs = 2
    num_queries = 10
    num_classes = 1
    
    # 1 class + background = 2 channels
    pred_logits = torch.randn(bs, num_queries, num_classes + 1)
    
    # Case 1: Normal 2D behavior
    out_prob = pred_logits.flatten(0, 1).softmax(-1)
    print(f"Case 1 out_prob shape: {out_prob.shape}")
    
    # Targets
    tgt_ids = torch.zeros(5, dtype=torch.long) # 5 targets total
    
    try:
        cost_class = -out_prob[:, tgt_ids]
        print(f"Case 1 cost_class shape: {cost_class.shape}")
    except IndexError as e:
        print(f"Case 1 failed: {e}")

    # Case 2: If pred_logits was [B, Q] (1D)
    pred_logits_1d = torch.randn(bs, num_queries)
    out_prob_1d = pred_logits_1d.flatten(0, 1).softmax(-1)
    print(f"Case 2 out_prob_1d shape: {out_prob_1d.shape}") # Should be [20]
    
    try:
        cost_class = -out_prob_1d[:, tgt_ids]
        print(f"Case 2 cost_class shape: {cost_class.shape}")
    except IndexError as e:
        print(f"Case 2 failed: {e}")

if __name__ == "__main__":
    test_matcher_shapes()
