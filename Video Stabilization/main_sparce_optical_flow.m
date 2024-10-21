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
%% Corner Detection
windowSize = 7;
thres = 9*10^8;
[i, j] = cornerDetection(frame1Gray,windowSize,thres);
%% Optical Flow at returnd corners
% Initialize the optical flow vectors
tic;
u = zeros(size(frame1Gray));
v = zeros(size(frame1Gray));

% Convert images to double precision
im1 = double(frame1Gray);
im2 = double(frame2Gray);

% Compute image gradients
Ix = conv2(im1, [-1 1; -1 1], 'same'); % gradient in x
Iy = conv2(im1, [-1 -1; 1 1], 'same'); % gradient in y
It = im2 - im1;

% Gaussian weighting window
gaussianWindow = fspecial('gaussian', [windowSize windowSize], 0.1);

% Compute products of derivatives at every pixel
Ix2 = Ix.^2;
Iy2 = Iy.^2;
Ixy = Ix.*Iy;
Ixt = Ix.*It;
Iyt = Iy.*It;

% Apply Gaussian filter to these products
Ix2 = conv2(Ix2, gaussianWindow, 'same');
Iy2 = conv2(Iy2, gaussianWindow, 'same');
Ixy = conv2(Ixy, gaussianWindow, 'same');
Ixt = conv2(Ixt, gaussianWindow, 'same');
Iyt = conv2(Iyt, gaussianWindow, 'same');

% Threshold for eigenvalues
threshold = 35 * 10^3;

% Compute optical flow only at the detected corners
for k = 1:length(i)
    x = i(k);
    y = j(k);

    % Ensure window stays within image boundaries
    minX = max(1, x - floor(windowSize / 2));
    maxX = min(size(im1, 1), x + floor(windowSize / 2));
    minY = max(1, y - floor(windowSize / 2));
    maxY = min(size(im1, 2), y + floor(windowSize / 2));

    % Construct matrix A
    Ix2_block = Ix2(minX:maxX, minY:maxY);
    Iy2_block = Iy2(minX:maxX, minY:maxY);
    Ixy_block = Ixy(minX:maxX, minY:maxY);
    Ixt_block = Ixt(minX:maxX, minY:maxY);
    Iyt_block = Iyt(minX:maxX, minY:maxY);

    ATA = [sum(Ix2_block(:)), sum(Ixy_block(:)); sum(Ixy_block(:)), sum(Iy2_block(:))];
    ATb = -[sum(Ixt_block(:)); sum(Iyt_block(:))];
    
    flow = ATA \ ATb;
    u(x, y) = flow(1);
    v(x, y) = flow(2); 
end
elapsedTime = toc;
fprintf('Time taken for sparce optical flow computation: %.4f seconds\n', elapsedTime);
%% Plot 
figure;imshow(frame1Gray);hold on;
% Display the corners
plot(j, i, 'ro', 'MarkerSize', 3, 'LineWidth', 0.5);
% Display the optical flow vectors
quiver(j, i, u(i + (j - 1) * size(im1, 1)), v(i + (j - 1) * size(im1, 1)), 'r', 'LineWidth', 0.7);
hold off;title('Optical Flow at Detected Corners');