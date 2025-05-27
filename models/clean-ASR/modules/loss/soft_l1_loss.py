import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftL1Loss(nn.Module):
    '''
    data2vec에서 사용한 soft l1 loss
    '''
    def __init__(self, temperature=2.0):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, teacher_features, student_features):
        t_features = teacher_features / self.temperature
        s_features = student_features / self.temperature
        
        loss = F.smooth_l1_loss(s_features, t_features, reduction='mean')
        
        return loss