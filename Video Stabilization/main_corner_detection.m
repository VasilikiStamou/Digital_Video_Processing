clear;clc;close all;
%% Extract Frames
% Read the video file
video = VideoReader('video1.avi');
% Choose two frames
frame1Idx = 10; 
% Read the two frames
video.CurrentTime = (frame1Idx-1) / video.FrameRate;
frame1 = readFrame(video);
% Convert frame to grayscale
frame1Gray = rgb2gray(frame1);
%% For Corner
windowSize = 7;
thres = 9*10^8;
tic;
[i, j] = cornerDetection(frame1Gray,windowSize,thres);
elapsedTime = toc;
fprintf('Time taken for corner detection computation: %.4f seconds\n', elapsedTime);
%% Using Matlab's built in 
tic;
corners = detectHarrisFeatures(frame1Gray, 'MinQuality', 0.0008, 'FilterSize', windowSize);
elapsedTime = toc;
fprintf('Time taken for matlabs corner detection computation: %.4f seconds\n', elapsedTime);
%% Plots
% Plot the original image
figure;subplot(1, 2, 1);imshow(frame1Gray, []);hold on;
% Plot the detected corners
plot(j, i, 'r+', 'MarkerSize', 5, 'LineWidth', 1);title('Detected Corners (Custom Implementation)');hold off;
% Plot the original image
subplot(1, 2, 2); imshow(frame1Gray, []);hold on;
% Highlight the detected corners
plot(corners.Location(:,1), corners.Location(:,2), 'r+', 'MarkerSize', 5, 'LineWidth', 1);
title('Detected Corners (MATLAB Built-In)');hold off;