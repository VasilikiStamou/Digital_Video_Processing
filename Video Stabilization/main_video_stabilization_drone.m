clear; clc; close all;
%% video_stabilization_drone
% Load the videos
video1 = VideoReader('video2.mp4');
video2 = VideoReader('StabilizedVideo_Drone_Custom.avi');

% Create a new figure for video playback
figure;
while hasFrame(video1) && hasFrame(video2) 
    % Read the next frame from each video
    frame1 = readFrame(video1);
    frame2 = readFrame(video2);
    
    % Display the first video
    subplot(1, 2, 1);
    imshow(frame1);
    title('Original Video');
    
    % Display the second video
    subplot(1, 2, 2);
    imshow(frame2);
    title('Stabilized Video');
    
    % Pause to allow for playback at the correct frame rate
    pause(1/video1.FrameRate);
end
