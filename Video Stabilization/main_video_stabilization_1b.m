clear; clc; close all;
%% videoStabilization2
%ret = videoStabilization2('video1.avi', 'StabilizedVideo_1b_Custom.avi');
%%
% Load the videos
video1 = VideoReader('video1.avi');
video2 = VideoReader('StabilizedVideo_1b_Matlab.avi');
video3 = VideoReader('StabilizedVideo_1b_Custom.avi');

% Create a new figure for video playback
figure;

while hasFrame(video1) && hasFrame(video2) && hasFrame(video3)
    % Read the next frame from each video
    frame1 = readFrame(video1);
    frame2 = readFrame(video2);
    frame3 = readFrame(video3);
    
    % Display the first video
    subplot(1, 3, 1);
    imshow(frame1);
    title('Original Video');
    
    % Display the second video
    subplot(1, 3, 2);
    imshow(frame2);
    title('Stabilized Video (Matlabs corner detection)');
    
    % Display the third video
    subplot(1, 3, 3);
    imshow(frame3);
    title('Stabilized Video (Custom)');
    
    % Pause to allow for playback at the correct frame rate
    pause(1/video1.FrameRate);
end
