import torch
from torch.utils.data import Dataset, DataLoader, Subset


class MaterialDataset(Dataset):
    def __init__(self, data_path, num_materials=100):
        data = torch.load(data_path)
        self.inputs = data.tensors[0]
        self.outputs = data.tensors[1]
        self.num_materials = num_materials
        self.samples_per_material = self.inputs.shape[0] // num_materials

    def get_material_data(self, material_index):
        start_idx = material_index * self.samples_per_material
        end_idx = start_idx + self.samples_per_material
        return self.inputs[start_idx:end_idx], self.outputs[start_idx:end_idx]

    def get_random_subset(self, material_index, num_samples):
        inputs, outputs = self.get_material_data(material_index)
        indices = torch.randperm(inputs.size(0))[:num_samples]
        return inputs[indices], outputs[indices]

    def __len__(self):
        return self.num_materials

    def __getitem__(self, idx):
        inputs, outputs = self.get_material_data(idx)
        return inputs, outputs


def get_dataloader(dataset, material_index, num_samples, batch_size=32, shuffle=True):
    inputs, outputs = dataset.get_random_subset(material_index, num_samples)
    subset = torch.utils.data.TensorDataset(inputs, outputs)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)
