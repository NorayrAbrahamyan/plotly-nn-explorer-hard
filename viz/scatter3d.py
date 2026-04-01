import pandas as pd
import plotly.graph_objects as go

def scatter():
    """
    Creates an interactive 3D scatter plot of the dataset.
    This function labels and colors the data points by class (0 and 1). 
    It includes buttons to toggle the visibility of each class, helping 
    to see how the groups are mixed or separated in 3D space
    """
    df = pd.read_csv("data/dataset.csv")

    data0 = df[df['label'] == 0]
    data1 = df[df['label'] == 1]

    class0 = go.Scatter3d(
        x = data0['x'],
        y = data0['y'],
        z = data0['z'],
        mode = 'markers',
        marker = dict(size = 4, opacity=0.7, color = 'coral'),
        name = 'class 0'
    )

    class1 = go.Scatter3d(
        x = data1['x'],
        y = data1['y'],
        z = data1['z'],
        mode = 'markers',
        marker = dict(size = 4, opacity = 0.7, color = 'cornflowerblue'),
        name = 'class 1'
    )

    fig = go.Figure(data = [class0, class1])

    fig.update_layout(
        title = '3D Interactive scatter plot', 
        template = 'plotly_dark',
        scene = dict(
            xaxis_title = 'x',
            yaxis_title = 'y',
            zaxis_title = 'z'
        ),
        updatemenus = [
            dict(
                type = 'buttons',
                buttons = [
                    dict(
                        label = 'show all',
                        method = 'update',
                        args = [{'visible': [True, True]}]
                    ),
                    dict(
                        label = 'class 0',
                        method  = 'update',
                        args = [{'visible': [True, False]}]
                    ),
                    dict(
                        label = 'class 1',
                        method = 'update',
                        args = [{'visible': [False, True]}]
                    ),
                ]
            )
        ]
    )

    fig.write_html('outputs/scatter3d.html')

if __name__ == '__main__':
    scatter()