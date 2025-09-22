# Marker Position Prediction from raw IMU data
Shah, V. R., & Dixon, P. C. (2025). Bridging the Methodological Gap Between Inertial Sensors and Optical Motion Capture: Deep Learning as the Path to Accurate Joint Kinematic Modelling Using Inertial Sensors. Sensors, 25(18), 5728. https://doi.org/10.3390/s25185728

## Installation
- Install [Miniconda](https://docs.anaconda.com/miniconda/) for your operating system 
- Clone this repository
- Navigate to the repository folder on your local machine
- Install the virtual environment:
```
conda env create -f environment.yml
conda activate Marker-position
```
Note: you may need to add your environment to the list of jupyter kernels:
```
python -m ipykernel install --user --name=Marker-position
```

## Running Jupyter notebooks
- Navigate to the notebooks subfolder
- Launch jupyter
- Navigate to appropriate notebook on your browser

## Step 1 : Prediction of Marker positions from raw IMU data and Visulizing predicted markers
  --> Run Marker_Position_Prediction_Notebook.ipynb

## Step 2 : Calculating Joint angles in MATLAB and Comparing against optical motion capture joint angles.
 --> Add biomechZoo toolbox in MATLAB search path.
 
 --> Run Kinematics_calculation.m

#### Toolboxes and/or Supporting Materials

- [Kinetics Toolkit](https://kineticstoolkit.uqam.ca/doc/index.php)
- [biomechZoo](https://github.com/PhilD001/biomechZoo)
