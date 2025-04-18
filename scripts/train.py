import os
import yaml
import pytorch_lightning as pl
from src.data.datamodule import ASRDataModule
from src.models.teacher_model import TeacherModel
from src.models.student_model import StudentModel
from src.training.trainer import Trainer
from src.training.losses import DistillationLoss

def main():
    # Load configurations
    with open(os.path.join('configs', 'model_config.yaml'), 'r') as f:
        model_config = yaml.safe_load(f)
    
    with open(os.path.join('configs', 'data_config.yaml'), 'r') as f:
        data_config = yaml.safe_load(f)
    
    with open(os.path.join('configs', 'training_config.yaml'), 'r') as f:
        training_config = yaml.safe_load(f)

    # Initialize data module
    data_module = ASRDataModule(data_config)

    # Initialize models
    teacher_model = TeacherModel(model_config['teacher'], trainer_config['optimizer'])
    student_model = StudentModel(model_config['student'])

    # Initialize loss function
    distillation_loss = DistillationLoss()

    # Initialize trainer
    trainer = Trainer(
        max_epochs=training_config['epochs'],
        gpus=training_config['gpus'],
        logger=pl.loggers.TensorBoardLogger('logs/')
    )

    # Start training
    trainer.fit(student_model, data_module)

if __name__ == '__main__':
    main()