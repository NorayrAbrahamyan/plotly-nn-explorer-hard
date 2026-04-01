# plotly-nn-explorer-hard

This project implements and analyzes a neural network trained on a synthetic 3D classification dataset.

The dataset consists of two interleaved spirals in 3D space, making it a non-linearly separable problem. This allows evaluation of how well a feedforward neural network can learn complex decision boundaries.

The project includes the full pipeline:

- Dataset generation  
- Training of a neural network using PyTorch  
- Tracking training metrics (loss and accuracy)  
- Visualization of results using Plotly  

Several visualizations are implemented to better understand the model:
- A 3D scatter plot of the dataset, showing how the classes are distributed  
![3d scatter1](outputs/3dscatter1.png)
![3d scatter2](outputs/3dscatter2.png)
![3d scatter3](outputs/3dscatter3.png)
- Training curves, illustrating the learning process over epochs  
![Training curves](outputs/training_curves.png)
- A 3D decision boundary surface, showing how the model separates the two classes
![Decision Surface 0.3](outputs/3ddec0.3.png)
![Decision Surface 0.4](outputs/3ddec0.4.png)
![Decision Surface 0.5](outputs/3ddec0.5.png)
![Decision Surface 0.6](outputs/3ddec0.6.png)
![Decision Surface 0.7](outputs/3ddec0.7.png)
- A loss landscape visualization, showing how the loss changes when modifying model weights  
![Loss Landscape](loss_landscape.png)

All components are connected through a single entry point (`main.py`), which runs the entire pipeline and generates all outputs.

