import torch
import torch.optim as optim
import numpy as np
import os
from dataset import MaterialDataset, get_dataloader
from tqdm import tqdm
from models import HyperBSSRDF
import matplotlib.pyplot as plt

config = {
    "total_steps_phase1": 10,  # number of epochs for phase 1
    "total_steps_phase2": 10,  # number of epochs for phase 2
    "batch_size": 32,
    "input_feature": 4,  # hypernetwork input feature size
    "hidden_layers": 1,
    "hidden_ch": 128,
    "output_feature": 3,
    "steps_till_summary": 1024,
    "num_samples_per_epoch": 100000,  # number of samples per epoch
    "learning_rate": 1e-4,
    "lr_scheduler_step_size": 10,
    "lr_scheduler_gamma": 0.1,
    "k1_steps": 5,  # steps for updating F in phase 2
    "k2_steps": 5,  # steps for updating M and F in phase 2
}


def save_model(model, path):
    torch.save(model.state_dict(), path)


def train_phase1(config, dataset, idx, model=None):
    if model is None:
        model = HyperBSSRDF(in_features=config["input_feature"], out_features=config["output_feature"]).cuda()
    loss_history = []
    optimizer = torch.optim.Adam(lr=config["learning_rate"], params=model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["lr_scheduler_step_size"],
                                                gamma=config["lr_scheduler_gamma"])

    model_dir = f"./models/model_{idx}_hyperbssrdf_phase1"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    for epoch in tqdm(range(config["total_steps_phase1"])):
        dataloader = get_dataloader(dataset, idx, config["num_samples_per_epoch"], batch_size=config["batch_size"])
        batch_bar = tqdm(dataloader, desc=f"Phase 1 - Epoch {epoch + 1}/{config['total_steps_phase1']}")
        epoch_losses = []
        for material_params, (inputs, outputs) in batch_bar:
            optimizer.zero_grad()
            material_params, inputs, outputs = material_params.cuda(), inputs[:, 4:9].cuda(), outputs.cuda()

            predictions = model(material_params, {'coords': inputs})['model_out']
            l1_loss = torch.nn.functional.l1_loss(predictions, outputs)
            mse_loss = torch.nn.functional.mse_loss(predictions, outputs)

            total_loss = l1_loss + 10. * mse_loss
            epoch_losses.append(total_loss.item())

            total_loss.backward()
            optimizer.step()

            batch_bar.set_postfix(loss=total_loss.item())

        scheduler.step()
        average_loss = sum(epoch_losses) / len(epoch_losses)
        loss_history.append(average_loss)
        print(f'Epoch [{epoch + 1}/{config["total_steps_phase1"]}], Average Loss: {average_loss}')

    save_model(model, path=f"{model_dir}/final_model.pth")
    torch.save(loss_history, f"{model_dir}/loss_history.pth")
    plt.figure()
    plt.plot(range(len(loss_history)), loss_history)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss for Material {idx} - Phase 1")
    plt.savefig(f"{model_dir}/training_loss.png")
    plt.close()
    return model, np.array(loss_history)


def train_phase2(config, dataset, idx, model):
    loss_history = []
    optimizer_hyper = torch.optim.Adam(lr=config["learning_rate"], params=model.hyper_net.parameters())
    optimizer_hypo = torch.optim.Adam(lr=config["learning_rate"], params=model.hypo_net.parameters())
    scheduler_hyper = torch.optim.lr_scheduler.StepLR(optimizer_hyper, step_size=config["lr_scheduler_step_size"],
                                                      gamma=config["lr_scheduler_gamma"])
    scheduler_hypo = torch.optim.lr_scheduler.StepLR(optimizer_hypo, step_size=config["lr_scheduler_step_size"],
                                                     gamma=config["lr_scheduler_gamma"])

    model_dir = f"./models/model_{idx}_hyperbssrdf_phase2"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    for epoch in tqdm(range(config["total_steps_phase2"])):
        dataloader = get_dataloader(dataset, idx, config["num_samples_per_epoch"], batch_size=config["batch_size"])
        batch_bar = tqdm(dataloader, desc=f"Phase 2 - Epoch {epoch + 1}/{config['total_steps_phase2']}")
        epoch_losses = []
        for material_params, (inputs, outputs) in batch_bar:
            # Split data into X1 and X2
            num_samples = inputs.size(0)
            split_idx = num_samples // 2
            X1, X2 = (material_params[:split_idx], inputs[:split_idx], outputs[:split_idx]), (
            material_params[split_idx:], inputs[split_idx:], outputs[split_idx:])

            # Update F using X1
            for _ in range(config["k1_steps"]):
                optimizer_hypo.zero_grad()
                material_params, inputs, outputs = X1[0].cuda(), X1[1][:, 4:9].cuda(), X1[2].cuda()

                predictions = model.hypo_net({'coords': inputs})['model_out']
                l1_loss = torch.nn.functional.l1_loss(predictions, outputs)
                mse_loss = torch.nn.functional.mse_loss(predictions, outputs)

                total_loss = l1_loss + 10. * mse_loss
                total_loss.backward()
                optimizer_hypo.step()

            # Update M and F using X2
            for _ in range(config["k2_steps"]):
                optimizer_hyper.zero_grad()
                optimizer_hypo.zero_grad()
                material_params, inputs, outputs = X2[0].cuda(), X2[1][:, 4:9].cuda(), X2[2].cuda()

                predictions = model(material_params, {'coords': inputs})['model_out']
                l1_loss = torch.nn.functional.l1_loss(predictions, outputs)
                mse_loss = torch.nn.functional.mse_loss(predictions, outputs)

                total_loss = l1_loss + 10. * mse_loss
                epoch_losses.append(total_loss.item())

                total_loss.backward()
                optimizer_hyper.step()
                optimizer_hypo.step()

            batch_bar.set_postfix(loss=total_loss.item())

        scheduler_hyper.step()
        scheduler_hypo.step()
        average_loss = sum(epoch_losses) / len(epoch_losses)
        loss_history.append(average_loss)
        print(f'Epoch [{epoch + 1}/{config["total_steps_phase2"]}], Average Loss: {average_loss}')

    save_model(model, path=f"{model_dir}/final_model.pth")
    torch.save(loss_history, f"{model_dir}/loss_history.pth")
    plt.figure()
    plt.plot(range(len(loss_history)), loss_history)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss for Material {idx} - Phase 2")
    plt.savefig(f"{model_dir}/training_loss.png")
    plt.close()
    return model, np.array(loss_history)


if __name__ == '__main__':
    train_idx = [0, 1]
    data_path = 'data/new_materials_dataset.pth'
    dataset = MaterialDataset(data_path, num_materials=2)

    for i in train_idx:
        # Phase 1
        model, history = train_phase1(config, dataset, i)
        # Phase 2
        model, history = train_phase2(config, dataset, i, model)
