import torch

class KDLoss(torch.nn.Module):
    def __init__(self, temperature=2.0, alpha=0.5):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
    def forward(self, t_logits, s_logits, student_loss):
        kd_loss = F.kl_div(
            F.log_softmax(s_logits / T, dim=-1),
            F.softmax(t_logits / T, dim=-1),
            reduction="batchmean"
        ) * (T ** 2)

        return alpha * kd_loss + (1 - alpha) * student_loss