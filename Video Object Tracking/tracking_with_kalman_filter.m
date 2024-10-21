clear; close all; clc
%% Read video into MATLAB using VideoReader
video = VideoReader('sevenup.m4v');
nframes = video.NumFrames;
%% Calculate the background image by averaging the first 90 frames
temp = zeros(video.Height, video.Width, 3);
for i = 1:90
    frame = readFrame(video);
    temp = double(frame) + temp;
end
imbkg = temp / 90;
%% Reinitialize video reader to read frames from the start
video.CurrentTime = 0;
%% Initial position and size of the object
initial_xcenter = 181; % x-coordinate of the center of the object
initial_ycenter = 95; % y-coordinate of the center of the object
xwidth = 69;  % width of the bounding box
ywidth = 133;  % height of the bounding box
%% Calculate the initial corners based on the center and size
xcorner = initial_xcenter - xwidth / 2;
ycorner = initial_ycenter - ywidth / 2;
%% Initialization for Kalman Filtering
centroidx = zeros(nframes, 1);
centroidy = zeros(nframes, 1);
actual = zeros(nframes, 2); % Only need 2 states: position and velocity
%%  Kalman filter parameters for tracking in the x direction
% Time step (assuming frame rate is 1 frame per second)
dt = 1; 
% Measurement noise covariance (variance of measurement noise)
R = 0.01;
% Process noise covariance (variance of process noise)
% Q controls how much we trust the model versus the measurements.
Q = [0.01, 0;
     0, 0.1]; 
% Initial error covariance matrix
P = [1, 0;
     0, 1]; 
% State transition matrix A for object moving only in the x direction
A = [1, dt;
     0, 1];
% Measurement matrix H, only measuring the x position
H = [1, 0];
%% Set the initial centroid
centroidx(1) = initial_xcenter;
centroidy(1) = initial_ycenter;
%% Initialize Kalman filter with the initial position
predicted = [centroidx(1); 0]; % [position; velocity]
%% Tracking loop
for i = 1:nframes
    frame = readFrame(video);
    imshow(frame);
    hold on;   
    imcurrent = double(frame);   
    %% Calculate the difference image to extract pixels with more than 80 (threshold) change
    % Lower threshold: More sensitivity to changes, more noise.
    % Higher threshold: Less sensitivity to changes, risk of missing the object.
    threshold = 80;
    diffimg = (abs(imcurrent(:,:,1) - imbkg(:,:,1)) > threshold) ...
            | (abs(imcurrent(:,:,2) - imbkg(:,:,2)) > threshold) ...
            | (abs(imcurrent(:,:,3) - imbkg(:,:,3)) > threshold);   
    %% Label the image and mark regions
    labelimg = bwlabel(diffimg, 4);
    markimg = regionprops(labelimg, 'BoundingBox', 'Centroid', 'Area');
    
    if isempty(markimg)
        % Use Kalman filter to predict, if no object is detected
        predicted = A * predicted;
    else
        % Find the object closest to the predicted centroid
        distances = arrayfun(@(m) abs(m.Centroid(1) - predicted(1)), markimg);
        [~, idx] = min(distances);
        cc = markimg(idx).Centroid;
        
        centroidx(i) = cc(1);
        centroidy(i) = cc(2);
        
        % Convert to column vector for consistency
        measurement = [centroidx(i); 0];
        
        % Apply Kalman filter
        Ppre = A * P * A' + Q;
        K = Ppre * H' / (H * Ppre * H' + R);
        actual(i, :) = (predicted + K * (measurement(1) - H * predicted))';
        P = (eye(2) - K * H) * Ppre;
        
        % Update prediction
        predicted = A * actual(i, :)';
    end   
    %% Plot the tracking rectangle after Kalman filtering 
    rectangle('Position', [(predicted(1) - xwidth / 2), (initial_ycenter - ywidth / 2), xwidth, ywidth], 'EdgeColor', 'r', 'LineWidth', 1.5);
    plot(predicted(1), initial_ycenter, 'rx', 'LineWidth', 1.5);
    drawnow;
end