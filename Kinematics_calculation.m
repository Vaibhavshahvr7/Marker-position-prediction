% Turn off warnings
warning off

% Store the current directory as the root folder
root_fld = cd;

% Define the folder containing prediction data
fld = '.\data\pp054\predicted_markers\prediction_data';

% List of subjects to process
Sub = {'pp054'};

%% Step 1: Create prediction folders for each subject
cd(fld)
create_pred_folders(Sub)
cd(root_fld)
% Convert prediction data into a specific format
prediction2zoo(fld, Sub)

%% Step 2: Return to the root directory
cd(root_fld)

% Define the folder containing static data
fld_static = '.\data\pp054\static';

%% Step 3: Perform kinematics calculations
prediction_kinematics(fld, fld_static, Sub)

%% Step 4: Process prediction data for visualization
fld = '.\data\pp054\predicted_markers\prediction_data';

% Define subject and condition names
subjects = {'pp054'};
conditions = {'prediction_data'};

% Define the list of channels to process
ch = {'RightHipAngle_x'; 'RightKneeAngle_x'; 'RightAnkleAngle_x';
    'LeftHipAngle_x'; 'LeftKneeAngle_x'; 'LeftAnkleAngle_x'};

% Convert prediction data into a table format for processing
pred_data = bmech_line2table_2024(fld, ch, subjects, conditions);

% Return to the root directory
cd(root_fld)

% Load true data from previously calculated joint data
true_data = load('.\data\pp054\omc\calculated_joint_data.mat');

%% Step 5: Plot comparison between predicted and true data
% Plot without Dynamic Time Warping (DTW)
dtw_condition = 0;
plot_angles(true_data, pred_data, dtw_condition)

% Plot with Dynamic Time Warping (DTW)
dtw_condition = 1;
plot_angles(true_data, pred_data, dtw_condition)

%% Functions

% Function to create folders for each subject
function create_pred_folders(Sub)
for i = 1:length(Sub)
    % Get the subject folder name
    mainFolderName = Sub{i};
    
    % Create the main folder if it does not exist
    if ~exist(mainFolderName, 'dir')
        mkdir(mainFolderName);
    end
end
disp('Folders created successfully');
end

% Function to convert prediction data into .zoo files
function prediction2zoo(fld, Sub)

fl = engine('fld', fld, 'ext', '.mat'); % Get list of .mat files
for file = 1:length(fl)
    % Load the data from the file
    data = load(fl{file});
    data = data.data;
    ch = fieldnames(data); % Extract channel names
    i = 1;
    
    % Get the length of the first channel's prediction data
    [Length, ~, ~] = size(data.(ch{i}).y_pred);
    
    for j = 1:Length
        for i = 1:length(ch)
            % Create file name and data structure for the first channel
            if i == 1
                if j < 100
                    zz = ['0', num2str(j)];
                elseif j < 10
                    zz = ['00', num2str(j)];
                else
                    zz = [num2str(j)];
                end
                filename_pred = [fld, filesep, Sub{file}, '\cycle_', zz, '.zoo'];
                ndata_pred.zoosystem = setZoosystem(filename_pred);
            end
            
            % Process channel data
            ch_name = ch{i};
            ch_data = data.(ch{i}).y_pred(j, :, :) * 10; % Scale data
            ch_data = double(reshape(ch_data, [101, 3])); % Reshape into 101x3
            ndata_pred = addchannel_data(ndata_pred, ch_name, ch_data, 'Video');
        end
        
        % Save the processed data to a .zoo file
        disp(['Saving data: ', filename_pred])
        zsave(filename_pred, ndata_pred)
    end
end
end

% Function to perform kinematics calculations
function prediction_kinematics(fld, fld_static, Sub)
fl_static = engine('fld', fld_static, 'ext', '.zoo'); % Get static files
for static = 1:length(fl_static)
    sdata = zload(fl_static{static}); % Load static data
    fl = engine('fld', fld, 'ext', '.zoo'); % Get prediction files
    fl = fl(contains(fl, Sub{static})); % Filter files for current subject
    
    for i = 1:length(fl)
        data = zload(fl{i}); % Load prediction data
        disp(['Processing data: ', num2str(length(fl)), ':', num2str(i)])
        
        % Define settings for kinematics calculations
        settings.graph = false; % Disable graphs
        settings.flat = true;  % Assume flat feet in static pose
        settings.comp = false; % Disable comparison
        
        % Update zoosystem with static data
        x = data.zoosystem.Video;
        data.zoosystem = sdata.zoosystem;
        data.zoosystem.Video = x;
        
        % Perform kinematics calculations
        sdata = makebones_data(sdata, 'static', settings.flat);
        sdata = kinematics_data(sdata);
        data = ankleoffsetPiG_data(data, sdata);
        data = makebones_data(data, 'dynamic');
        data = kinematics_data(data, settings);
        
        % Save updated data
        zsave(fl{i}, data)
    end
    
    % Save updated static data
    zsave(fl_static{static}, sdata)
end
end

% Function to plot true and predicted angles
function plot_angles(true_data, pred_data, dtw_condition)
% Define channels and titles for the plots
ch_pred = {'RightHipAngle_x'; 'RightKneeAngle_x'; 'RightAnkleAngle_x';
    'LeftHipAngle_x'; 'LeftKneeAngle_x'; 'LeftAnkleAngle_x'};
ch_true = {'Rhip'; 'Rknee'; 'Rankle'; 'Lhip'; 'Lknee'; 'Lankle'};
Title_names = {'Right Hip Flexion', 'Right Knee Flexion', 'Right Ankle Dorsiflexion',
    'Left Hip Flexion', 'Left Knee Flexion', 'Left Ankle Dorsiflexion'};

figure
for i = 1:length(ch_pred)
    for j = 1:2 % Iterate through two cycles
        subplot(2, 3, i)
        x = pred_data.(ch_pred{i}){j}; % Predicted data
        y = true_data.data.pp054.(['cycle_', num2str(j - 1)]).(ch_true{i}).'; % True data
        
        % Adjust knee angles
        if contains(ch_true{i}, 'knee')
            y = -y;
            y = y - min(y);
            x = x - min(x);
        end
        
        % Apply Dynamic Time Warping (DTW) if required
        if dtw_condition == 1
            [~, b, c] = dtw(x, y);
            x = normalize_line(x(b));
            y = normalize_line(y(c));
        end
        
        % Plot predicted and true data
        plot(x, '--b', 'LineWidth', 1)
        hold on
        plot(y, 'r', 'LineWidth', 1)
    end
    title(Title_names{i})
end
end
