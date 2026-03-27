import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

with open('model/history.json', 'r') as f:
    history = json.load(f)

train_loss = history['train_loss']
val_loss = history['val_loss']
train_acc = history['train_acc']
val_acc = history['val_acc']

epochs = list(range(1, len(train_loss )+1))

best_epoch = val_loss.index(min(val_loss)) + 1
best_val_loss = min(val_loss)

fig = make_subplots(
    rows = 1, cols = 2,
    subplot_titles=("train loss/val loss", "train acc/val acc")
)
#left
fig.add_trace(
    go.Scatter(x = epochs, y = train_loss, mode = 'lines', name = 'Train loss'),
    row = 1, col = 1
)

fig.add_trace(
    go.Scatter(x = epochs, y = val_loss, mode = "lines", name = "Val loss"),
    row = 1, col = 1
)
#right
fig.add_trace(
    go.Scatter(x = epochs, y = train_acc, mode = 'lines', name = "Train acc"),
    row = 1, col = 2
)

fig.add_trace(
    go.Scatter(x = epochs, y = val_acc, mode = 'lines', name = "Val acc"),
    row = 1, col = 2
)

#best_epoch
fig.add_vline(
    x = best_epoch,
    line_color = 'white',
    annotation_text = f'best epoch - {best_epoch}'
)

fig.update_layout(
    title = "Training Curves",
    template = 'plotly_dark'
)

fig.update_xaxes(rangeslider_visible = True)

fig.update_xaxes(title_text="Epoch", row=1, col=1)
fig.update_xaxes(title_text="Epoch", row=1, col=2)
fig.update_yaxes(title_text="Loss", row=1, col=1)
fig.update_yaxes(title_text="Accuracy", row=1, col=2)

fig.write_html("outputs/training_curves.html")