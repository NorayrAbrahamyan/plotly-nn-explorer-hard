import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import plotly.graph_objects as go
from model.train import FeedforwardNN
from sklearn.preprocessing import StandardScaler
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def loss_landscape():
    df = pd.read_csv("data/dataset.csv")
    X = df[['x', 'y', 'z']].values.astype(np.float32)
    y = df['label'].values.astype(np.float32).reshape(-1, 1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_t = torch.tensor(X)
    y_t = torch.tensor(y)
    model = FeedforwardNN()
    criterion = nn.BCELoss()
    model.load_state_dict(torch.load("model/model.pth"))
    model.eval()

    theta = parameters_to_vector(model.parameters()).detach()
    d1 = torch.randn_like(theta)
    d2 = torch.randn_like(theta)
    d1 = d1 / torch.norm(d1)
    d2 = d2 / torch.norm(d2)

    alphas = np.linspace(-1, 1, 40)
    betas = np.linspace(-1, 1, 40)
    loss_grid = np.zeros((len(alphas), len(betas)))

    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            theta_new = theta + a * d1 + b * d2
            vector_to_parameters(theta_new, model.parameters())
            with torch.no_grad():
                preds = model(X_t)
                loss = criterion(preds, y_t).item()
            loss_grid[i, j] = loss

    vector_to_parameters(theta, model.parameters())

    with torch.no_grad():
        actual_loss = criterion(model(X_t), y_t).item()

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=alphas,
            y=betas,
            z=loss_grid,
            contours_z=dict(
                show=True,
                usecolormap=True,
                project_z=True
            )
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[0],
            y=[0],
            z=[actual_loss],
            mode='markers',
            marker=dict(color='red', size=5),
            name='Trained model'
        )
    )

    fig.update_layout(
        title="Loss Landscape",
        template="plotly_dark",
        scene=dict(
            xaxis_title="alpha",
            yaxis_title="beta",
            zaxis_title="loss"
        )
    )

    fig.write_html("outputs/loss_landscape.html")

if __name__ == "__main__":
    loss_landscape()