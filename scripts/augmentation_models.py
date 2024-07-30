import numpy as np
import torch.nn as nn
import torch
from typing import Optional
from tqdm import tqdm


def augmentaion_noise_jittering(input_data: np.ndarray, 
                                jitter_ratio: float = 0.2) -> np.ndarray:
    
    std_emp = np.std(input_data.flatten())
    noise = np.random.normal(0, jitter_ratio * std_emp, input_data.shape)
    jittered_data = input_data + noise
    
    return jittered_data


class LSTMBaseline:
    def __init__(self, input_dim: int, 
                 use_weights: Optional[str]=None):

        self.aug_model = LSTMGenerator(input_dim=input_dim, hidden_dim=64, 
                                       num_layers=2, output_dim=input_dim)
        
        self.device = device()
        
        if use_weights is not None:
            self.aug_model.load_state_dict(
                torch.load('use_weights', map_location=self.device))


    def generate(self, data: torch.Tensor) -> np.array:
        
        self.aug_model.eval()
        with torch.no_grad():
            generated_series = self.aug_model(
                data.to(self.device)).cpu().numpy()

        return generated_series
    

    def model_training(self, train_loader, test_loader, 
                       epochs: int, lr: float, 
                       save_best_weights: Optional[str]=None):
        
        save_best = True if save_best_weights is not None else False
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.aug_model.parameters(), lr=lr, weight_decay=0.02)

        self.aug_model.to(self.device)

        history = train_aug(self.aug_model, epochs, 
                            train_loader, test_loader,
                            criterion, optimizer, 
                            save_best=save_best, path_to_save=save_best_weights)
        
        train_loss, test_loss = zip(*history)

        return train_loss, test_loss


#TODO Transformer baseline





class LSTMGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMGenerator, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)
        return out


def train_epoch_transformer(train_loader, model, optimizer):
    model.train()
    for batch in train_loader:

        loss = model(
            past_values=batch["past_values"].to(device()),
            future_values=batch["future_values"].to(device()),
            past_time_features=batch["past_time_features"].to(device()),
            future_time_features=batch["future_time_features"].to(device()),
            past_observed_mask=batch["past_observed_mask"].to(device()),
        ).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def eval_epoch_transformer(loader, model):
    model.eval()
    losses = []

    with torch.no_grad():
        for batch in loader:

            loss = model(
                past_values=batch["past_values"].to(device()),
                future_values=batch["future_values"].to(device()),
                past_time_features=batch["past_time_features"].to(device()),
                future_time_features=batch["future_time_features"].to(device()),
                past_observed_mask=batch["past_observed_mask"].to(device()),

            ).loss
            losses.append(loss.item())

    return np.mean(losses)


def train_aug(model, epochs, train_loader, val_loader, 
              criterion, optimizer, scheduler=None, 
              save_best=False, path_to_save=None, transformer=False):

    history = []
    best_val_loss = 1000
    val_loss = None
    for epoch in tqdm(range(1, epochs+1)):
        if transformer:
            train_epoch_transformer(train_loader, model, optimizer)
            train_loss = eval_epoch_transformer(train_loader, model)


        else:
            train_epoch(train_loader, model, criterion, optimizer)
            train_loss = eval_epoch(train_loader, model, criterion)
            val_loss = eval_epoch(val_loader, model, criterion)


        if scheduler is not None:
            scheduler.step()

        if val_loss is not None:
            if save_best:
                if best_val_loss > val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), path_to_save)

            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Test Loss {val_loss:.4f}')
            history.append((train_loss, val_loss))

        else:
            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}')
            history.append(train_loss)

    if save_best:
        model.load_state_dict(torch.load(path_to_save, map_location=device()))

    return history


def train_epoch(train_loader, model, criterion, optimizer):
    model.train()
    for data, y in train_loader:
        data = data.to(device())
        y = y.to(device())
        out = model(data)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def eval_epoch(loader, model, criterion):
    model.eval()
    losses = []
    with torch.no_grad():
        for data, y in loader:
            data = data.to(device())
            y = y.to(device())
            out = model(data)
            loss = criterion(out, y)
            losses.append(loss.item())
    return np.mean(losses)

def device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



