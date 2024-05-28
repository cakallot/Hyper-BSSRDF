import numpy as np
import torch
import os
from dataset import MaterialDataset, get_dataloader
from tqdm import tqdm
from model import MLP, Siren
import matplotlib.pyplot as plt

config = {
    'skip': True,
    "model_type": "mlp",  # "mlp" 或 "siren"
    "total_steps": 64,
    "batch_size": 32,
    'input_feature': 5,
    "hidden_layers": 2,
    "hidden_ch": 64,  # 增加隐藏层宽度
    'output_feature': 3,
    "steps_till_summary": 1024,
    "num_samples_per_epoch": 100000,  # 每个 epoch 中的样本数
    "learning_rate": 1e-4,  # 初始学习率
    "lr_scheduler_step_size": 10,  # 每10个epoch调整一次学习率
    "lr_scheduler_gamma": 0.1  # 学习率调整比例
}


def save_model(model, path):
    torch.save(model.state_dict(), path)


def init_model(config):
    if config["model_type"] == "siren":
        return Siren(in_features=config['input_feature'], out_features=config['output_feature'],
                     hidden_features=config['hidden_ch'], hidden_layers=config['hidden_layers']).cuda()
    else:
        return MLP(in_features=config['input_feature'], out_features=config['output_feature'],
                   hidden_features=config['hidden_ch'], hidden_layers=config['hidden_layers']).cuda()


def train(config, dataset, idx, model=None):
    if model is None:
        model = init_model(config)
    loss_history = []
    num_epochs = config["total_steps"]
    optimizer = torch.optim.Adam(lr=config["learning_rate"], params=model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["lr_scheduler_step_size"],
                                                gamma=config["lr_scheduler_gamma"])

    model_suffix = "_siren" if config["model_type"] == "siren" else ""

    model_dir = f"./models/model_{idx}{model_suffix}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    for epoch in tqdm(range(num_epochs)):
        dataloader = get_dataloader(dataset, idx, config["num_samples_per_epoch"], batch_size=config["batch_size"])
        batch_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        epoch_losses = []
        for inputs, outputs in batch_bar:
            optimizer.zero_grad()
            inputs = inputs[:, 4:9].cuda()
            outputs = outputs.cuda()

            predictions = model(inputs)
            l1_l = torch.nn.functional.l1_loss(predictions, outputs)
            mse_l = torch.nn.functional.mse_loss(predictions, outputs)

            total_loss = l1_l + 10. * mse_l
            epoch_losses.append(total_loss.item())

            total_loss.backward()
            optimizer.step()

            batch_bar.set_postfix(loss=total_loss.item())

        scheduler.step()
        average_loss = sum(epoch_losses) / len(epoch_losses)
        loss_history.append(average_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {average_loss}')

    save_model(model, path=f"{model_dir}/final_model.pth")
    torch.save(loss_history, f"{model_dir}/loss_history.pth")
    plt.figure()
    plt.plot(range(len(loss_history)), loss_history)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss for Material {idx}")
    plt.savefig(f"{model_dir}/training_loss.png")

    plt.close()
    return model, np.array(loss_history)


if __name__ == '__main__':
    ## 训练的材质的index列表
    train_idx = [0, 1]
    data_path = 'data/new_materials_dataset.pth'
    dataset = MaterialDataset(data_path, num_materials=2)

    for i in train_idx:
        model, history = train(config, dataset, i)
