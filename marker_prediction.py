import tensorflow as tf
import keras
import sys
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.signal import butter, filtfilt
import GaitLab2Go as GL2G
import os
import kineticstoolkit.lab as ktk
import matplotlib.pyplot as plt
lab=GL2G.data_processing()

def convert_zoo2pickle(fld= os.path.join(".","data", "pp054", "imu"),ext='.zoo',subject_list = ['pp054']):
    # Get the IMU variables from lab's variables zoo and store in 'variables'
    variables = lab.variables_zoo_IMU()

    # Remove the last 18 variables from the list
    #variables = variables[:-18]

    # Find files in the specified folder 'fld' with the extension specified in 'ext'
    fl = lab.find_files(path=fld, ext=ext)

    # Convert files from Zoo format to dictionary format using 'variables' and store the result
    lab.zoo2dictionary(fl, variables)

    # Set file extension to '.pkl' for further processing
    ext = '.pkl'

    # Find files again in 'fld' directory, but now with the new extension '.pkl'
    fl = lab.find_files(path=fld, ext=ext)

    # Specify the list of subjects for processing (in this case, a single subject 'pp054')


    # List of files to process (found previously with the '.pkl' extension)
    file_list = fl

    # Process the specified subject files using the subject and file lists
    print(subject_list)
    print(file_list)
    lab.process_subject_files(subject_list, file_list)

    # Load the first file in the file list (pickled data) into a Pandas DataFrame 'x'
    x = pd.read_pickle(fl[0])

    # Extract a list of variable names from the 'data' attribute in the loaded file
    variable = list(x.data.keys())

# Function to process data for specific subjects and tasks, based on provided file list and variables
def process_subject_data_task(subject_list, file_list, variable):
    # Initialize an empty dictionary 'data' to store processed data for each variable
    data = {}

    # Initialize each variable in 'data' with an array of zeros with shape (1, 101)
    for var in variable:
        data[var] = np.zeros([1, 101])

    # Add subject and task keys to 'data' with initial values
    data['subject'] = np.array('test')
    data['task'] = np.array([99])

    # Loop over each subject in the subject list
    for sub in subject_list:
        # Loop over each file in file_list that contains the subject identifier
        for fn in file_list[np.char.find(file_list, sub) > 0]:
            print(fn)  # Print file name for debugging

            # Extract task identifier from file name
            Task = fn.split('/')[-1].split('_')[2]

            # Determine the task type based on Task identifier and assign task_num accordingly
            if Task == '01' or Task == '05':
                task_num = 0
                print('walk')
            elif Task == '02' or Task == '04':
                task_num = 1
                # Uncomment below to print 'jog' if needed
                # print('jog')
            elif Task == '03':
                task_num = 2
                # Uncomment below to print 'run' if needed
                # print('run')

            # Load data from the file as a Pandas DataFrame 'x'
            x = pd.read_pickle(fn)

            # Append the subject identifier to 'data' under 'subject' key
            data['subject'] = np.append(data['subject'], sub)

            # Append the task number to 'data' under 'task' key
            data['task'] = np.append(data['task'], task_num)

            # Loop over each variable in the loaded data
            for var in x.data.keys():
                # Append the Ncycle data for each variable along axis 0
                data[var] = np.append(data[var], x.Ncycle_data[var], axis=0)

    # Return the processed data dictionary
    return data

def process_knee_data(data,sensor,Marker):
# Create an empty dictionary to store processed knee data
    knee_data = {}

    # Loop through keys in the input data
    for var in data.keys():
        # Include only specific variables related to shank, thigh, knee angles, and subject
        if sensor[0] in var or sensor[1] in var or sensor[2] in var or Marker in var or 'subject' in var:
            knee_data[var] = data[var]

    # Get the shape of the 'shankR_Acc_X' array (assuming it exists)
    x = knee_data[f'{sensor[0]}_Acc_X'].shape

    # Lambda function to reshape and concatenate accelerometer and gyroscope data
    reshape_and_concat = lambda acc_gyr: np.concatenate(
        [knee_data[f'{part}_{axis}'].reshape(x[0], x[1], 1)
          for part in sensor#,'footR','shankL','thighL','footL']
          for axis in ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']], axis=2)

    # Reshape and concatenate accelerometer and gyroscope data
    nA=map(reshape_and_concat,['Acc'])
    nA1=list(nA)
    nA1=nA1[0]

    # Create a dictionary containing processed data
    knee_d = {
        'train': nA1,
        'subject': knee_data['subject'],
        #'test': nE,
        'task':data['task']
    }

    return knee_d

def extracting_markerwise_data(result_data):
    # List of markers representing body landmarks
    markers = ['LASI', 'RASI', 'LPSI', 'RPSI', 'LTHI', 'LKNE', 'LTIB', 'LANK', 'LHEE', 'LTOE',
               'RTHI', 'RKNE', 'RTIB', 'RANK', 'RHEE', 'RTOE']

    # Nested list of sensor groups, each containing sensors associated with specific body segments
    sensors = [['trunk', 'thighR', 'thighL'],    # Sensors for trunk and thighs
               ['thighR', 'shankR', 'footR'],    # Sensors for right thigh, shank, and foot
               ['thighL', 'shankL', 'footL']]    # Sensors for left thigh, shank, and foot

    # Mapping each marker to the appropriate sensor group
    # 0 corresponds to the first group, 1 to the second group, 2 to the third group
    nums = [0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1]

    # Iterate over all markers
    for i in range(0, len(markers)):
        # Call the process_knee_data function for each marker
        # Pass the corresponding sensor group (from nums) and marker name
        print(f'Saving data for {markers[i]}')
        knee_d = process_knee_data(result_data, sensor=sensors[nums[i]], Marker=markers[i])

        # Save the processed data as a pickle file
        # The file is saved in a directory with the marker's name as the filename
        pd.to_pickle(knee_d, os.path.join(".","data","pp054","predicted_markers",f"{markers[i]}.pkl"))


def loading_predition_data(fld = os.path.join(".","data", "pp054", "predicted_markers")):
    # Path to the folder containing the marker position pickle files

    print('Loading IMU data for prediction')
    # Load pickle file for the LASI marker (Left Anterior Superior Iliac Spine)
    data_LASI = pd.read_pickle( os.path.join(fld,"LASI.pkl"))

    # Load pickle file for the RASI marker (Right Anterior Superior Iliac Spine)
    data_RASI = pd.read_pickle( os.path.join(fld,"RASI.pkl"))

    # Load pickle file for the LPSI marker (Left Posterior Superior Iliac Spine)
    data_LPSI = pd.read_pickle( os.path.join(fld,"LPSI.pkl"))

    # Load pickle file for the RPSI marker (Right Posterior Superior Iliac Spine)
    data_RPSI = pd.read_pickle( os.path.join(fld,"RPSI.pkl"))

    # Load pickle file for the LTHI marker (Left Thigh)
    data_LTHI = pd.read_pickle( os.path.join(fld,"LTHI.pkl"))

    # Load pickle file for the LKNE marker (Left Knee)
    data_LKNE = pd.read_pickle( os.path.join(fld,"LKNE.pkl"))

    # Load pickle file for the LTIB marker (Left Tibia)
    data_LTIB = pd.read_pickle( os.path.join(fld,"LTIB.pkl"))

    # Load pickle file for the LANK marker (Left Ankle)
    data_LANK = pd.read_pickle( os.path.join(fld,"LANK.pkl"))

    # Load pickle file for the LHEE marker (Left Heel)
    data_LHEE = pd.read_pickle( os.path.join(fld,"LHEE.pkl"))

    # Load pickle file for the LTOE marker (Left Toe)
    data_LTOE = pd.read_pickle( os.path.join(fld,"LTOE.pkl"))

    # Load pickle file for the RTHI marker (Right Thigh)
    data_RTHI = pd.read_pickle( os.path.join(fld,"RTHI.pkl"))

    # Load pickle file for the RKNE marker (Right Knee)
    data_RKNE = pd.read_pickle( os.path.join(fld,"RKNE.pkl"))

    # Load pickle file for the RTIB marker (Right Tibia)
    data_RTIB = pd.read_pickle( os.path.join(fld,"RTIB.pkl"))

    # Load pickle file for the RANK marker (Right Ankle)
    data_RANK = pd.read_pickle( os.path.join(fld,"RANK.pkl"))

    # Load pickle file for the RHEE marker (Right Heel)
    data_RHEE = pd.read_pickle( os.path.join(fld,"RHEE.pkl"))

    # Load pickle file for the RTOE marker (Right Toe)
    data_RTOE = pd.read_pickle( os.path.join(fld,"RTOE.pkl"))

    # Get the shape of the 'train' dataset from the LASI marker data
    shape_train = data_LASI['train'].shape

    # Reshape the 'train' dataset for LASI to add an extra dimension (e.g., channel dimension)
    data_LASI['train'] = data_LASI['train'].reshape(shape_train[0], shape_train[1], shape_train[2], 1)

    # Reshape the 'train' dataset for LKNE and RKNE similarly
    data_LKNE['train'] = data_LKNE['train'].reshape(shape_train[0], shape_train[1], shape_train[2], 1)
    data_RKNE['train'] = data_RKNE['train'].reshape(shape_train[0], shape_train[1], shape_train[2], 1)

    print('Combining IMU data for prediction')
    # Combine multiple sensor datasets along the last dimension (axis=3) to form a single training dataset
    # Includes subsets of LASI, LKNE, and RKNE data
    train = np.concatenate((
        data_LASI['train'][:, :, 0:3, :], data_LASI['train'][:, :, 3:6, :], data_LASI['train'][:, :, 6:9, :],
        data_LASI['train'][:, :, 9:12, :], data_LASI['train'][:, :, 12:15, :], data_LASI['train'][:, :, 15:18, :],
        data_LKNE['train'][:, :, 6:9, :], data_LKNE['train'][:, :, 9:12, :], data_LKNE['train'][:, :, 12:15, :], data_LKNE['train'][:, :, 15:18, :],
        data_RKNE['train'][:, :, 6:9, :], data_RKNE['train'][:, :, 9:12, :], data_RKNE['train'][:, :, 12:15, :], data_RKNE['train'][:, :, 15:18, :]
    ), axis=3)

    print('Converting IMU data Units to required units')

    # Iterate over specific indices of the training dataset
    # Convert gyroscope data (odd indices) from degrees to radians and scale accelerometer data (even indices)
    for i in [1, 3, 5, 7, 9, 11, 13]:
        train[:, :, :, i] = np.deg2rad(train[:, :, :, i])  # Convert degrees to radians
        train[:, :, :, i - 1] = train[:, :, :, i - 1] * 9.81  # Scale accelerometer data to m/s² (gravity factor)

    print('Angular velocity of gyroscope converted to radian/s from deg/s')
    print('Linear acceleration data in m/s from g by multiplying with 9.81')

    # Left shank sensor
    print('Rotating sensors to reorient gyroscope and accelerometer data')
    # Define a 180-degree rotation matrix around the z-axis for reorienting gyroscope and accelerometer data
    R_z_180 = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])
    # Apply the rotation matrix to left shank gyroscope and accelerometer data
    train[:, :, :, 4] = np.einsum('ij,klj->kli', R_z_180, train[:, :, :, 4])
    train[:, :, :, 5] = np.einsum('ij,klj->kli', R_z_180, train[:, :, :, 5])

    # Left foot sensor
    # Define a 180-degree rotation matrix for reorienting data
    R_z_180 = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ])
    # Apply the rotation matrix to left foot gyroscope and accelerometer data
    train[:, :, :, 8] = np.einsum('ij,klj->kli', R_z_180, train[:, :, :, 8])
    train[:, :, :, 9] = np.einsum('ij,klj->kli', R_z_180, train[:, :, :, 9])

    # Right foot sensor
    # Define a 180-degree rotation matrix for reorienting data
    R_z_180 = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ])
    # Apply the rotation matrix to right foot gyroscope and accelerometer data
    train[:, :, :, 12] = np.einsum('ij,klj->kli', R_z_180, train[:, :, :, 12])
    train[:, :, :, 13] = np.einsum('ij,klj->kli', R_z_180, train[:, :, :, 13])

    return data_LASI, train



def Normalized_gait( A):
        # Generate an array 'x' with values from 0 to the number of columns in A
        x = np.arange(A.shape[-1])

        # Replace any NaN (Not a Number) values in array A with 0
        A = np.where(np.isnan(A) == 1, 0, A)

        # Interpolate values using cubic interpolation with 101 points
        # Interpolate the data along axis 0 (assuming x is the axis)
        Y = interpolate.interp1d(x, A, kind='cubic')(np.linspace(x.min(), x.max(), 101))

        # Return the interpolated values
        return Y

def train_test(data_LASI, train, Select_subject='S01'):
    """
    Function to process and filter sensor data, with an option to select a specific subject.

    Args:
      data_LASI: A dictionary containing sensor data, typically structured with 'subject', 'task', and sensor values.
      train: A numpy array or similar data structure containing training data.
      Select_subject: A string specifying the subject to filter data for. Default is 'S01'.

    Returns:
      x_test_f: The filtered sensor data corresponding to the selected subject.
      subject_test_f: The subject labels for the selected subject.
    """

    # Extract the training data and relevant fields from the data_LASI dictionary
    X = train[:, :, :]  # Selecting all the dimensions from the training data array
    Subject = data_LASI['subject'][:]  # Extract subject labels from the data_LASI dictionary
    task = data_LASI['task'][:]  # Extract task information (currently unused in the function)

    # Create a new array `nx` to store the Euclidean norm of the 3D accelerometer data (x, y, z)
    nx = np.zeros((X.shape[0], X.shape[1], 14))  # Initialize an empty array for the norm calculation (14 for each data point)

    # Loop over each time-point (or sample) and calculate the Euclidean norm of the (x, y, z) accelerometer data
    for i in range(0, X.shape[3]):
        # Calculating the Euclidean norm of the accelerometer data (sqrt(x^2 + y^2 + z^2))
        nx[:, :, i] = np.sqrt(X[:, :, 0, i]**2 + X[:, :, 1, i]**2 + X[:, :, 2, i]**2)

    # Create a new array `nx1` to concatenate the calculated norm with the original accelerometer data
    nx1 = np.zeros((X.shape[0], X.shape[1], 4, X.shape[3]))  # New shape: adding 4th channel (norm) to the data

    # Loop over each sample in the time-series and concatenate the norm along the 3rd axis (channel-wise)
    for i in range(0, X.shape[3]):
        nx1[:, :, :, i] = np.concatenate((X[:, :, :, i], nx[:, :, i:i + 1]), axis=2)

    # Update the original data array `X` to include the new sensor data with norms
    X = nx1

    # Find the indices of the selected subject in the dataset
    rows = np.where(Subject == Select_subject)[0]

    # Filter the training data to only include the selected subject
    x_test_f = X[rows]

    # Extract the subject labels for the selected subject
    subject_test_f = Subject[rows]

    # Return the filtered data and subject labels for the selected subject
    return x_test_f, subject_test_f

def butter_lowpass_filter(data, cutoff=6, fs=200, order=4):
  """
  Applies a Butterworth low-pass filter to the input data.

  Args:
    data: The input data (e.g., a NumPy array).
    cutoff: The cutoff frequency in Hz.
    fs: The sampling frequency in Hz.
    order: The order of the Butterworth filter.

  Returns:
    The filtered data.
  """
  nyquist = 0.5 * fs
  normal_cutoff = cutoff / nyquist
  b, a = butter(order, normal_cutoff, btype='low', analog=False)
  y = filtfilt(b, a, data)
  return y

def predict_marker_position(fld,participants,data_LASI, train, Select_subject):
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging (0 = all logs, 1 = warnings, 2 = errors, 3 = fatal)

    # Static files initialization
    parti_data = {}  # Dictionary to store data for each participant

    Sub = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18']  # List of subjects
    # Loop through each participant
    for parti in participants:
        parti_data[parti] = {}  # Initialize dictionary for the current participant's data

        # Loop through each subject
        for s in Sub:
            Select_subject = parti  # Set the selected subject (current participant in this case)

            # Call the train_test function to get the test data and labels for the selected subject
            x_test_f, subject_test_f = train_test(data_LASI, train, Select_subject)

            # Check if there is a GPU available for TensorFlow (optional)
            device = tf.test.gpu_device_name()

            # Load the pre-trained model specific to the current subject
            model_2 = tf.keras.models.load_model(os.path.join(".","models",f"{s}_30_11_2024_walk_model.keras"), compile=False)

            # Predict using the model, feeding in the test data (note: slicing the input as required)
            pred = model_2.predict(x_test_f[:, :, :])

            data = {}  # Initialize an empty dictionary to store marker data
            markers = ['LASI', 'RASI', 'LPSI', 'RPSI', 'LTHI', 'LKNE', 'LTIB', 'LANK', 'LHEE', 'LTOE', 'RTHI', 'RKNE', 'RTIB', 'RANK', 'RHEE', 'RTOE']  # List of marker names

            # Loop over each marker and store the predicted data
            j = 0  # Initialize index for accessing different marker predictions
            for i in markers:
                # Store the predicted values (scaled by 10 for some units) for each marker
                data[i] = {'y_pred': pred[:, :, :, j] * 10}
                j += 1  # Increment the marker index

            # Apply a low-pass filter to the predicted data for each marker
            for i in markers:
                for j in range(3):  # Loop through the 3 dimensions (e.g., x, y, z)
                    data[i]['y_pred'][:, :, j] = butter_lowpass_filter(data[i]['y_pred'][:, :, j])  # Apply filter

            # Store the processed data for the current subject and participant
            parti_data[parti][s] = data

            # Save the processed data as a pickle file for future use
            pd.to_pickle(data, f'{fld}//' + Select_subject + 'marker_prediction_data.pkl')
    return parti_data

def average_prediction(fld,parti_data,participants):

    # Initialize an empty dictionary to store the mean data for each participant
    mean_data = {}

    # Loop through each participant in the 'participants' list
    for parti in participants:
        mean_data[parti] = {}  # Create a sub-dictionary for the current participant

        # Initialize the structure of 'mean_data' for each marker and condition in the first subject (S01)
        for marker in parti_data[parti]['S01'].keys():
            for condi in parti_data[parti]['S01'][marker].keys():
                mean_data[parti][marker] = {}  # Create a dictionary for the marker
                mean_data[parti][marker][condi] = {}  # Create a dictionary for the condition

        # Initialize a variable 'i' to control the loop for accumulating data
        i = 1

        # Loop through each subject in the current participant's data
        for sub in parti_data[parti].keys():

            # Loop through each marker in the current subject's data
            for marker in parti_data[parti][sub].keys():

                # Loop through each condition in the current marker's data
                for condi in parti_data[parti][sub][marker].keys():

                    # Fetch the data for the current condition (though it's not used directly here)
                    parti_data[parti][sub][marker][condi]

                    # If it's the first iteration (i==1), initialize the mean data for the condition
                    if i == 1:
                        mean_data[parti][marker][condi] = parti_data[parti][sub][marker][condi]
                    else:
                        # Otherwise, add the current data to the accumulated data
                        mean_data[parti][marker][condi] = mean_data[parti][marker][condi] + parti_data[parti][sub][marker][condi]

            # After processing the first subject, set i to 2 to accumulate data from subsequent subjects
            i = 2


    # Loop through each participant in the 'participants' list
    for parti in participants:

        # Loop through each marker in the current participant's mean data
        for marker in mean_data[parti].keys():

            # Loop through each condition for the current marker in the participant's data
            for condi in parti_data[parti][sub][marker].keys():

                # Normalize the accumulated data for each condition by dividing by 18
                # This assumes that there are 18 subjects (or data points) contributing to the total sum
                mean_data[parti][marker][condi] = mean_data[parti][marker][condi] / 18

    import pandas as pd  # Import pandas for data manipulation and saving in pickle format
    import pickle  # Import pickle for serializing Python objects (though it's not used explicitly here)
    from scipy.io import savemat  # Import savemat from scipy for saving data in .mat (MATLAB) format

    # Loop through each participant in the 'participants' list
    for parti in participants:
        print('Saving the average predtion data of markers in a .mat and .pkl file')
        # Save the processed mean data for the current participant as a .pkl (pickle) file
        # This serializes the data into a binary format which can later be loaded using pd.read_pickle
        pd.to_pickle(mean_data[parti], os.path.join(fld,"prediction_data",f"{parti}_walk.pkl"))

        # Save the processed mean data as a .mat (MATLAB) file
        # The 'savemat' function saves the data in a format that can be loaded in MATLAB
        # The data is saved under the variable name 'data' inside the .mat file
        savemat(os.path.join(fld,"prediction_data",f"{parti}_walk.mat"), {'data': mean_data[parti]})


def visulize_predicted_data():
    fld=os.path.join(".","data", "pp054", "predicted_markers","prediction_data")
    fl=lab.find_files(path=fld,ext='pkl')
    import pandas as pd
    data=pd.read_pickle(fl[0])

    import kineticstoolkit as ktk
    import numpy as np


    # Get a mask to identify the time indices within the desired range
    # Trim the time and all channels using the mask
    ts_trimmed = ktk.TimeSeries()
    time=np.array(range(0,101))/100
    ts_trimmed.time=time.reshape(101)
    for key1 in data.keys():
            for key2 in data[key1].keys():
                ndata=np.ones([101,4])
                ndata[:,0:3]=data[key1][key2][1,:,:]/100
                #if key2=='y_pred':
                    #ndata[:,0:3]=ndata[:,0:3]+[-0.0065,-0.12,0.04]
                ndata[:,0]=butter_lowpass_filter(ndata[:,0], 6, 200, order=4)
                ndata[:,1]=butter_lowpass_filter(ndata[:,1], 6, 200, order=4)
                ndata[:,2]=butter_lowpass_filter(ndata[:,2], 6, 200, order=4)

                ch_name=key2+':'+key1
                ts_trimmed = ts_trimmed.add_data(ch_name,ndata)  # Trim all channels

    interconnections = dict()  # Will contain all segment definitions
    interconnections["LLowerLimb"] = {
        "Color": (1, 0, 0),  # In RGB format (here, greenish blue)
        "Links": [  # List of lines that span lists of markers
            ["*LTOE", "*LHEE", "*LANK", "*LTOE"],
            ["*LANK", "*LKNE", "*LASI"],
            ["*LKNE", "*LPSI"],
        ],
    }
    interconnections["RLowerLimb"] = {
        "Color": (0, 1, 0),
        "Links": [
            ["*RTOE", "*RHEE", "*RANK", "*RTOE"],
            ["*RANK", "*RKNE", "*RASI"],
            ["*RKNE", "*RPSI"],
        ],
    }
    interconnections["TrunkPelvis"] = {
        "Color": (0, 0, 1),
        "Links": [
            ["*LPSI", "*LASI", "*RASI", "*RPSI", "*LPSI"]

        ],
    }
    p = ktk.Player(
        ts_trimmed,
        interconnections=interconnections,
        up="z",
        anterior="-y",
        target=(0, 0.5, 1),
        azimuth=0.1,
        zoom=1.5,
    )
    p.background_color = 'w'
    p.grid_size = 4
    p.grid_subdivision_size = 0.5
    p.grid_origin = (0.6, 0.7, 0.7)
    p.point_size = 10
    p.default_point_color = (0.30,0.30,0.80)
