clear;clc;close all;
%% Extract Frames
% Read the video file
video = VideoReader('video1.avi');
% Choose two frames
frame1Idx = 10; % frame index 1
frame2Idx = 11; % frame index 2
% Read the two frames
video.CurrentTime = (frame1Idx-1) / video.FrameRate;
frame1 = readFrame(video);
video.CurrentTime = (frame2Idx-1) / video.FrameRate;
frame2 = readFrame(video);
% Convert frames to grayscale
frame1Gray = rgb2gray(frame1);
frame2Gray = rgb2gray(frame2);
%% For Optical Flow
windowSize =7;
% Compute optical flow
tic;
[u, v] = opticalFlow_implimentation(frame1Gray, frame2Gray, windowSize);
elapsedTime = toc;
fprintf('Time taken for optical flow computation: %.4f seconds\n', elapsedTime);
%% Using Matlab's built in 
tic;
% Create optical flow object using the built-in function
opticFlow = opticalFlowLK('NoiseThreshold', 0.01);
% Estimate the flow for the first frame
estimateFlow(opticFlow, frame1Gray);
% Estimate the flow for the second frame
flow = estimateFlow(opticFlow, frame2Gray);
% Extract horizontal and vertical flow components
U = flow.Vx;
V = flow.Vy;
elapsedTime = toc;
fprintf('Time taken for matlabs optical flow computation: %.4f seconds\n', elapsedTime);
%% Plots 
% Display the first frame
figure;
subplot(1, 2, 1); 
imshow(frame1Gray);
hold on;
% Display the optical flow vectors
quiver(u, v, 'r', 'LineWidth', 2, 'MaxHeadSize', 2);
hold off;
title('Optical Flow (Custom Implementation)');
subplot(1, 2, 2);
imshow(frame1Gray);
hold on;
quiver(U, V, 'r');
hold off;
title('Optical Flow (MATLAB Built-In)');