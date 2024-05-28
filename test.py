import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import os
from dataset import MaterialDataset, get_dataloader
from model import MLP, Siren
import matplotlib.pyplot as plt


# 加载模型
def load_model(model_path, config):
    if config["model_type"] == "siren":
        model = Siren(in_features=config['input_feature'], out_features=config['output_feature'],
                      hidden_features=config['hidden_ch'], hidden_layers=config['hidden_layers'])
    else:
        model = MLP(in_features=config['input_feature'], out_features=config['output_feature'],
                    hidden_features=config['hidden_ch'], hidden_layers=config['hidden_layers'])
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()
    return model


# 计算 PSNR
def calculate_psnr(target, prediction, max_pixel_value=1.0):
    mse = np.mean((target - prediction) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_pixel_value / np.sqrt(mse))


# 模型评估函数
def evaluate_model(model, dataloader, model_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs[:, 4:9].to(device)  # 取输入的第4到第8列
            targets = targets.to(device)

            predictions = model(inputs)

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    mse = mean_squared_error(all_targets, all_predictions)
    mae = mean_absolute_error(all_targets, all_predictions)
    psnr = calculate_psnr(all_targets, all_predictions)

    with open(f"{model_dir}/evaluation_metrics.txt", "w") as f:
        f.write(f'Mean Squared Error: {mse}\n')
        f.write(f'Mean Absolute Error: {mae}\n')
        f.write(f'PSNR: {psnr}\n')

    plt.figure(figsize=(10, 5))
    plt.plot(all_predictions[:100, 0], label='Predicted')
    plt.plot(all_targets[:100, 0], label='Actual')
    plt.legend()
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title('Predicted vs Actual Values')
    plt.savefig(f"{model_dir}/predicted_vs_actual.png")
    plt.close()

    return mse, mae, psnr


# 主函数
if __name__ == '__main__':
    config = {
        "model_type": "mlp",  # "mlp" 或 "siren"
        "batch_size": 32,
        'input_feature': 5,
        "hidden_layers": 2,
        "hidden_ch": 64,  # 与训练时相同的隐藏层宽度
        'output_feature': 3,
    }

    data_path = 'data/new_materials_dataset.pth'

    dataset = MaterialDataset(data_path, num_materials=2)
    material_index = [0, 1]

    for i in material_index:
        dataloader = get_dataloader(dataset, i, batch_size=config['batch_size'], shuffle=False,
                                    num_samples=100000)
        model_suffix = "_siren" if config["model_type"] == "siren" else ""
        model_dir = f'models/model_{i}{model_suffix}'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = f'{model_dir}/final_model.pth'
        model = load_model(model_path, config)

        mse, mae, psnr = evaluate_model(model, dataloader, model_dir)

        print(f'Model {i} - Mean Squared Error: {mse}')
        print(f'Model {i} - Mean Absolute Error: {mae}')
        print(f'Model {i} - PSNR: {psnr}')
