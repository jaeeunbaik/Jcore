import argparse
import torch
from src.models.teacher_model import TeacherModel
from src.models.student_model import StudentModel
from src.models.distillation import Distillation
from src.data.datamodule import DataModule
from src.utils.metrics import calculate_metrics

def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_metrics = {}

    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            metrics = calculate_metrics(outputs, targets)
            for key, value in metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = 0
                total_metrics[key] += value

    avg_loss = total_loss / len(dataloader)
    for key in total_metrics:
        total_metrics[key] /= len(dataloader)

    return avg_loss, total_metrics

def main():
    parser = argparse.ArgumentParser(description='Evaluate the ASR model.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint.')
    parser.add_argument('--data_config', type=str, required=True, help='Path to the data configuration file.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for evaluation.')

    args = parser.parse_args()

    device = args.device
    data_module = DataModule(args.data_config)
    dataloader = data_module.test_dataloader()

    model = StudentModel()  # or TeacherModel() based on your evaluation needs
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)

    avg_loss, metrics = evaluate_model(model, dataloader, device)

    print(f'Average Loss: {avg_loss}')
    for key, value in metrics.items():
        print(f'{key}: {value}')

if __name__ == '__main__':
    main()