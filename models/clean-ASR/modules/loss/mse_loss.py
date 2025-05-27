import torch
import torch.nn as nn
import torch.nn.functional as F

class MSELoss(nn.Module):
    """
    Mean Squared Error 손실 함수
    특성 또는 로짓 레벨에서의 지식 증류에 유용
    """
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, teacher_outputs, student_outputs):
        """
        Args:
            teacher_outputs: 교사 모델의 출력
            student_outputs: 학생 모델의 출력
        """
        # 온도 스케일링 적용
        t_outputs = teacher_outputs / self.temperature
        s_outputs = student_outputs / self.temperature
        
        # MSE 손실 계산
        loss = F.mse_loss(s_outputs, t_outputs, reduction='mean')
        
        return loss