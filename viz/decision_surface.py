import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from model.train import FeedforwardNN

def decision_surface():
    df = pd.read_csv('data/dataset.csv')
    X_data = df[['x', 'y', 'z']].values
    y_data = df['label'].values
    scaler = StandardScaler()
    model = FeedforwardNN()
    scaler.fit(X_data)
    
    model.load_state_dict(torch.load('model/model.pth'))
    model.eval()
    x_min, x_max = X_data[:, 0].min(), X_data[:, 0].max()
    y_min, y_max = X_data[:, 1].min(), X_data[:, 1].max()
    z_min, z_max = X_data[:, 2].min(), X_data[:, 2].max()

    x = np.linspace(x_min, x_max, 35)
    y = np.linspace(y_min, y_max, 35)
    z = np.linspace(z_min, z_max, 35)

    xg, yg, zg = np.meshgrid(x, y, z)
    grid = np.c_[xg.ravel(), yg.ravel(), zg.ravel()]
    grid_s = scaler.transform(grid)
    grid_t = torch.tensor(grid_s, dtype = torch.float32)

    with torch.no_grad():
        pbs = model(grid_t).numpy().flatten()
    
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    surfaces = []

    for t in thresholds:
        surfaces.append(
            go.Isosurface(
                x = grid[:, 0],
                y = grid[:, 1],
                z = grid[:, 2],
                value = pbs,
                isomin=t,
                isomax=t,
                opacity=0.4,
                surface_count = 1,
                visible=(t==0.5)
            )
        )
    
    scatter = go.Scatter3d(
        x=X_data[:,0],
        y=X_data[:,1],
        z=X_data[:,2],
        mode='markers',
        marker=dict(
            size=3,
            color=y_data,
            colorscale='Viridis',
            opacity=0.7
        ),
    )

    fig = go.Figure(data=surfaces + [scatter])
    
    steps = []
    for i, t in enumerate(thresholds):
        step = dict(
            method="update",
            args=[{"visible": [j == i for j in range(len(surfaces))] + [True]}],
            label=f"{t}"
        )
        steps.append(step)

    fig.update_layout(
        title="3D Decision Boundary",
        template="plotly_dark",
        sliders=[dict(active=2, steps=steps)]
    )

    fig.write_html("outputs/decision_surface.html")


if __name__ == "__main__":
    decision_surface()