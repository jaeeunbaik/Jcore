import torch
import torch.nn as nn
import torch.nn.functional as F

class KLDivergenceLoss(nn.Module):
    """
    KL Divergence 손실 함수
    로짓 레벨에서의 지식 증류에 유용
    """
    def __init__(self, temperature=2.0):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, teacher_logits, student_logits):
        """
        Args:
            teacher_logits: 교사 모델의 로짓
            student_logits: 학생 모델의 로짓
        """
        # 온도 스케일링 적용
        t_logits = teacher_logits / self.temperature
        s_logits = student_logits / self.temperature
        
        # KL Divergence 손실 계산
        kd_loss = F.kl_div(
            F.log_softmax(s_logits, dim=-1),
            F.softmax(t_logits, dim=-1),
            reduction="batchmean"
        ) * (self.temperature ** 2)
        
        return kd_loss