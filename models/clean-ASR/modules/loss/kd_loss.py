import torch
import torch.nn as nn
import torch.nn.functional as F

from .kl_divergence_loss import KLDivergenceLoss
from .soft_l1_loss import SoftL1Loss
from .mse_loss import MSELoss

class KDLoss(nn.Module):
    """
    지식 증류를 위한 손실 함수 래퍼 클래스
    다양한 손실 함수를 설정에 따라 선택할 수 있습니다.
    """
    def __init__(self, cfg):
        super().__init__()
        
        self.loss_type = cfg.loss_type
        self.temperature = cfg.temperature
        self.alpha = cfg.alpha
        self.target = cfg.target
        
        if self.loss_type == 'kl_div':
            self.loss_fn = KLDivergenceLoss(temperature=self.temperature)
        elif self.loss_type == 'soft_l1':
            self.loss_fn = SoftL1Loss(temperature=self.temperature)
        elif self.loss_type == 'mse':
            self.loss_fn = MSELoss(temperature=self.temperature)
        else:
            raise ValueError(f"지원하지 않는 손실 함수 유형: {self.loss_type}")
        
        print(f"KD Loss 초기화: 타입={self.loss_type}, 온도={self.temperature}, 알파={self.alpha}, 타겟={self.target}")
    
    def forward(self, teacher_outputs, student_outputs, student_loss=None):
        kd_loss = self.loss_fn(teacher_outputs, student_outputs)

        if student_loss is not None and self.alpha < 1.0:
            return self.alpha * kd_loss + (1 - self.alpha) * student_loss
        else:
            return kd_loss