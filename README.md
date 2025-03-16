# Marker Position Prediction from raw IMU data
As a part of the Article: Bridging The Gap Between Optical Motion Capture and Inertial Measurement Unit Technology: A Deep Learning Approach to Joint Kinematic Modeling

Journal Name: Journal of Biomechanics 

DOI: (Will be added)

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
