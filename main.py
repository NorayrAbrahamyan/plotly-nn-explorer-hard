from data.generate import generate_dataset
from model.train import train_model
from viz.scatter3d import scatter
from viz.training_curves import training_curves
from viz.decision_surface import decision_surface
from viz.loss_landscape import loss_landscape

def main():
    #Generate dataset
    df = generate_dataset()
    df.to_csv("data/dataset.csv", index=False)

    #Train model
    train_model()

    #Visualizations
    scatter()
    training_curves()
    decision_surface()
    loss_landscape()


if __name__ == "__main__":
    main()