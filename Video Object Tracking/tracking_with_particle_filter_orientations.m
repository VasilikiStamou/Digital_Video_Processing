function tracking_with_particle_filter_orientations
clear;clc;close all;
%% File and Folder Setup
imageFolder = 'Coke/img';  % Folder containing the images
imageFiles = dir(fullfile(imageFolder, '*.jpg'));  % List all jpg files in the folder
numImages = length(imageFiles);  % Number of images in the folder
groundTruthFile = 'Coke/groundtruth_rect.txt';  
% Read ground truth positions and sizes
groundTruth = readmatrix(groundTruthFile);
%% Video Writer Initialization 
videoWriter = VideoWriter('outputvideo2.avi'); 
open(videoWriter);
%% Parameters Setup
DIM1 = 480;  % Height
DIM2 = 640;  % Width
% Size of the rectangle to track [width, height]
rectSize = groundTruth(1, 3:4);  % Initial width and height from the first ground truth entry
% Coefficient for the noise covariance (bigger C => higher dispersion of particles)
C = 6; 
%% final
% Initial position of the rectangle 
i0 = groundTruth(1, 1) + rectSize(1)/2;  % x-coordinate of the center from ground truth
j0 = groundTruth(1, 2) + rectSize(2)/2;  % y-coordinate of the center from ground truth
theta0 = 0;  % Initial orientation angle in degrees

% Mean and variance for the noise model 
M = [0 0 0]';

V = C * [45 15 -0.25;...
         15 50 1;...
       -0.25 1 0.7];
% Number of particles
N = 100;  
%% Particle Initialization
% Initialize particles (position + orientation)
particles = mvnrnd(M, V, N) + repmat([i0 j0 theta0], N, 1);
% Initialize particle weights to the same value
w = ones(1, N) / N;
%% Process Each Image
%for frameCount = 1:50 
for frameCount = 1:numImages 
    %% Read and Process Frame
    imageFile = fullfile(imageFolder, imageFiles(frameCount).name);
    frame = imread(imageFile);
    % Convert frame to grayscale
    grayFrame = rgb2gray(frame); 

    % Create a mask for the original black rectangle region
    mask = false(DIM1, DIM2);
    i1 = max(round(i0 - rectSize(1)/2), 1);
    i2 = min(round(i0 + rectSize(1)/2), DIM1);
    j1 = max(round(j0 - rectSize(2)/2), 1);
    j2 = min(round(j0 + rectSize(2)/2), DIM2);
    mask(j1:j2, i1:i2) = true;
    
    % Apply edge detection only within the masked region
    grayFrameMasked = grayFrame;
    grayFrameMasked(~mask) = 0;

    % Apply Canny edge detector
    cannyEdges = edge(grayFrameMasked, 'Canny');
    combinedEdges = cannyEdges;
    % Distance transform of the inverse combined edge image
    D = bwdist(~combinedEdges);

    %% Likelihood Calculation
    threshold = 1e-20;
    % Get the likelihood for each particle
    for c = 1:N
        w(c) = calculateObservationLikelihood(rectSize, D, particles(c, 1), particles(c, 2), particles(c, 3), DIM1,DIM2);
        if w(c) < threshold
            w(c) = 0;
        end
    end
    %% Weight normalization
    if sum(w) == 0
        break;
    end
    w = w / sum(w);
    
    %% Effective Sample Size (ESS)
    ess = 1 / sum(w.^2);
    if ess < N / 2
        % Resample particles based on their weights
        new_particles = resample_particles(particles, w, N, M, V);
        particles = new_particles;
        w = ones(1, N) / N;
    end

    %% Create Image for Plotting Particles
    % Draw all particles into an RGB image with red color
    particle_image = zeros(DIM1, DIM2);
    for c = 1:N
        particle_image = particle_image + plotRotatedRectangle(rectSize, particles(c, 1), particles(c, 2), particles(c, 3),DIM1,DIM2);
    end
    mask = (particle_image > 1);
    particle_image(mask) = 1; 
    % Create the final image by overlaying particles on the original frame
    total_image = im2double(frame);
    particle_image_rgb = zeros(DIM1, DIM2, 3);
    particle_image_rgb(:, :, 1) = particle_image; % Red channel
    mask_particles = (particle_image_rgb > 0);
    total_image(mask_particles) = particle_image_rgb(mask_particles);

    %% Resampling Particles
    % Resample particles based on their weights and add noise
    for c = 1:N
        s = performSampling(w);
        M1 = [particles(s, 1); particles(s, 2); particles(s, 3)] + M;
        % Add noise to x and y only
        y(c,:) = M1 + mvnrnd(M,V,1)';
    end
    particles = y;

    %% Draw the target into an RGB image with black color (and set background to white)
    mask_t = (combinedEdges == 1);
    mask = (total_image(:, :, 1) == 0 & total_image(:, :, 2) == 0 & total_image(:, :, 3) == 0);
    for c = 1:3
        x = total_image(:, :, c);
        x(mask) = 1;
        x(mask_t) = 0;
        total_image(:, :, c) = x;
    end

    %% Calculate New Position and Draw Thick Line
    % Calculate the new position and orientation of the rectangle (target object)
    i0 = sum(particles(:, 1) .* w');
    j0 = sum(particles(:, 2) .* w');
    theta0 = sum(particles(:, 3) .* w');
    
    % Draw the target object (rotated black rectangle) in the current frame
    total_image = plotRectangleOutline(total_image, rectSize, i0, j0,theta0, [0 0 0]);
    
    % Draw the ground truth rectangle in green
    gt_i0 = groundTruth(frameCount, 1) + rectSize(1) / 2;
    gt_j0 = groundTruth(frameCount, 2) + rectSize(2) / 2;
    total_image = plotRectangleOutline(total_image, rectSize, gt_i0, gt_j0,theta0, [0 1 0]);

    
    % Display the current frame with tracked particles
    imshow(total_image);
    drawnow;
    % Write the frame with tracked particles to the output video
    writeVideo(videoWriter, im2frame(im2uint8(total_image)));
end

close(videoWriter);
end

%% Calculate likelihood of a particle based on the distance from the real center
function out = calculateObservationLikelihood(rectSize, D, i, j,theta,DIM1,DIM2)
% Decay factor
a = 2; 
% The particle rectangle image, rotated by theta
particle_image = plotRotatedRectangle(rectSize, i, j,theta, DIM1,DIM2);
f = exp(-a * D);
out = sum(sum(particle_image .* f));
end
%%
function new_particles = resample_particles(particles, w, N, M, V)
    new_particles = zeros(size(particles));
    cumulative_sum = cumsum(w);
    for i = 1:N
        random_number = rand;
        index = find(cumulative_sum >= random_number, 1);
        new_particles(i, 1:3) = particles(index, 1:3) + mvnrnd(M, V, 1);
    end
end
%%
function outimage = plotRotatedRectangle(rectSize, i0, j0, theta, DIM1, DIM2)
    w = rectSize(1);
    h = rectSize(2);

    % Calculate the top-left and bottom-right corners of the rectangle
    i1 = max(round(i0 - w/2), 1);
    i2 = min(round(i0 + w/2), DIM2);
    j1 = max(round(j0 - h/2), 1);
    j2 = min(round(j0 + h/2), DIM1);

    % Create a mask with a filled rectangle
    rect = zeros(DIM1, DIM2);
    rect(j1:j2, i1:i2) = 1;

    % Rotate the rectangle
    outimage = imrotate(rect, theta, 'crop');
end

%%
function outimage = plotRectangleOutline(image, rectSize, i0, j0, theta, color)
    w = rectSize(1);
    h = rectSize(2);
    
    % Calculate the four corners of the rectangle before rotation
    corners = [
        -w/2, -h/2;
         w/2, -h/2;
         w/2,  h/2;
        -w/2,  h/2
    ];
    
    % Rotation matrix
    R = [cosd(theta), -sind(theta); sind(theta), cosd(theta)];
    
    % Rotate and translate corners
    rotated_corners = (R * corners')' + [i0, j0];
    
    % Draw lines between the corners
    outimage = image;
    for k = 1:4
        x1 = rotated_corners(k, 1);
        y1 = rotated_corners(k, 2);
        x2 = rotated_corners(mod(k, 4) + 1, 1);
        y2 = rotated_corners(mod(k, 4) + 1, 2);
        outimage = insertShape(outimage, 'Line', [x1, y1, x2, y2], 'Color', color, 'LineWidth', 2);
    end
end
%%
function index = performSampling(w)
    % Resampling function based on weights
    index = find(rand <= cumsum(w), 1, 'first');
end
