function tracking_with_particle_filter
clear;clc;close all;
%% Video Reader and Writer Initialization
videoFile = 'sevenup.m4v'; 
videoReader = VideoReader(videoFile);
videoWriter = VideoWriter('outputvideo1.avi'); 
open(videoWriter);
%% Parameters Setup
% Size of the video frames
DIM = 600;
% Size of the rectangle to track [width, height]
rectSize = [185, 455]; 
% Coefficient for the noise covariance(bigger C => higher dispersion of particles)
C = 6;
% Initial position of the rectangle 
i0 = 455; j0 = 320; 
% Mean and variance for the noise model 
M = [0 0]';
% Diagonal elements represent the variances (spread) in x and y,
% Off-diagonal elements represent the covariance  
V = C * [14 0; 0 0.01];
% Number of particles
N = 100; 
%% Particle Initialization
% Initialize particles (add noise)

% This effectively shifts all the random particles by the vector [i0 j0].
% So, instead of the particles being centered around the mean vector M,
% they are now centered around [i0 j0].

particles = mvnrnd(M, V, N) + repmat([i0 j0], N, 1);
% Initialize particle weights to the same value
w = ones(1, N) / N;
frameCount = 0;
while hasFrame(videoReader) 
    %% Read and Process Frame
    frame = readFrame(videoReader);
    % Resize the frame
    resizedFrame = imresize(frame, [DIM, DIM]); 
    % Convert frame to grayscale
    grayFrame = rgb2gray(resizedFrame); 

    % Create a mask for the original black rectangle region
    mask = false(DIM, DIM);
    i1 = max(round(i0 - rectSize(1)/2), 1);
    i2 = min(round(i0 + rectSize(1)/2), DIM);
    j1 = max(round(j0 - rectSize(2)/2), 1);
    j2 = min(round(j0 + rectSize(2)/2), DIM);
    mask(j1:j2, i1:i2) = true;
    
    % Apply edge detection only within the masked region
    grayFrameMasked = grayFrame;
    grayFrameMasked(~mask) = 0;

    % Apply Canny edge detector
    cannyEdges = edge(grayFrameMasked, 'Canny');
    % Apply Sobel filter to detect vertical edges
    sobelVerticalEdges = edge(grayFrameMasked, 'Sobel', 'vertical');
    % Combine Canny edges with vertical edges
    combinedEdges = cannyEdges & sobelVerticalEdges;
    % Distance transform of the inverse combined edge image
    D = bwdist(~combinedEdges);

    %% Likelihood Calculation
    threshold = 1e-20;
    % Get the likelihood for each particle
    for c = 1:N
        w(c) = calculateObservationLikelihood(rectSize, D, particles(c, 1), particles(c, 2));
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
    particle_image = zeros(DIM, DIM);
    for c = 1:N
        particle_image = particle_image + plotRectangle(rectSize, particles(c, 1), particles(c, 2), DIM);
    end
    mask = (particle_image > 1);
    particle_image(mask) = 1; 
    % Create the final image by overlaying particles on the original frame
    total_image = im2double(resizedFrame);
    particle_image_rgb = zeros(DIM, DIM, 3);
    particle_image_rgb(:, :, 1) = particle_image; % Red channel
    mask_particles = (particle_image_rgb > 0);
    total_image(mask_particles) = particle_image_rgb(mask_particles);

    %% Resampling Particles
    % Do resampling based on the weights
    for c = 1:N
        s = performSampling(w);
        M1 = [particles(s, 1); particles(s, 2)] + M;
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
    % Calculate the new position of the rectangle (target object)
    i0 = sum(particles(:, 1) .* w');
    j0 = sum(particles(:, 2) .* w');
    % Draw the target object (black rectangle) in the current frame
    total_image = plotRectangleOutline(total_image,rectSize,i0,j0,[0 0 0]);
    % Display the current frame with tracked particles
    imshow(total_image);
    drawnow;
    % Write the frame with tracked particles to the output video
    writeVideo(videoWriter, im2frame(im2uint8(total_image)));
    % Increment frame counter
    frameCount = frameCount + 1;
end
close(videoWriter);
end

%% Calculate likelihood of a particle based on the distance from the real center
function out = calculateObservationLikelihood(rectSize, D, i, j)
% Decay factor
a = 2; 
% The particle rectangle image
particle_image = plotRectangle(rectSize, i, j, size(D, 1));
f = exp(-a * D);
out = sum(sum(particle_image .* f));
end

function new_particles = resample_particles(particles, w, N, M, V)
    new_particles = zeros(size(particles));
    cumulative_sum = cumsum(w);
    for i = 1:N
        random_number = rand;
        index = find(cumulative_sum >= random_number, 1);
        new_particles(i, :) = particles(index, :) + mvnrnd(M, V, 1);
    end
end

function outimage = plotRectangle(rectSize, i0, j0, DIM)
w = rectSize(1);
h = rectSize(2);

i1 = max(round(i0 - w/2), 1);
i2 = min(round(i0 + w/2), DIM);
j1 = max(round(j0 - h/2), 1);
j2 = min(round(j0 + h/2), DIM);

outimage = zeros(DIM, DIM);
outimage(j1:j2, i1:i2) = 1;
end

function outimage = plotRectangleOutline(image, rectSize, i0, j0, color)
w = rectSize(1);
h = rectSize(2);

i1 = max(round(i0 - w/2), 1);
i2 = min(round(i0 + w/2), size(image, 2));
j1 = max(round(j0 - h/2), 1);
j2 = min(round(j0 + h/2), size(image, 1));

% Draw rectangle outline using line functions
for c = 1:3
    channel = image(:,:,c);
    % Top edge
    channel(j1, i1:i2) = color(c);
    % Bottom edge
    channel(j2, i1:i2) = color(c);
    % Left edge
    channel(j1:j2, i1) = color(c);
    % Right edge
    channel(j1:j2, i2) = color(c);
    image(:,:,c) = channel;
end

outimage = image;
end
%% Do sampling given the weight function f
function out = performSampling(f)

x = rand;
acc = 0;
i = 1;

while 1
    acc = acc + f(i);
    if acc > x
        break;
    end
    i = i + 1;
end
out = i;

end
