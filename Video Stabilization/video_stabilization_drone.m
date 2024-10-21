clear; clc; close all;
%%
filename = 'video2.mp4'; 
hVideoSrc = VideoReader(filename);
outputVideo = VideoWriter('StabilizedVideo_Drone_Custom.avi');
open(outputVideo);
hVPlayer = vision.VideoPlayer;

% Process all frames in the video
movMean = im2gray(im2single(readFrame(hVideoSrc)));
imgB = movMean;
imgBp = imgB;
correctedMean = imgBp;
ii = 2;
cumulativeTform = eye(3);
windowSize = 7;  % Window size for custom corner detection
thres = 9 * 10^8;  % Threshold for corner detection
maxCorners = 3.5 * 10^4;  % Maximum number of corners to detect

% Parameters for moving average
N = 5; % Number of frames for moving average
transformBuffer = repmat(eye(3), 1, 1, N);
bufferIndex = 1;

%%
while hasFrame(hVideoSrc)  %&& ii < 700
    % Read in new frame
    imgA = imgB;
    imgAp = imgBp;
    imgB = im2gray(im2single(readFrame(hVideoSrc)));
    movMean = movMean + imgB;
    
    %Corner detection
    pointsA = detectFASTFeatures(imgA, 'MinContrast', 0.1);
    pointsB = detectFASTFeatures(imgB, 'MinContrast', 0.1);
    %{
    % Custom corner detection
    [iA, jA] = cornerDetection(imgA, windowSize, thres);
    [iB, jB] = cornerDetection(imgB, windowSize, thres);

    % Convert corner coordinates to point objects
    pointsA = cornerPoints([jA, iA]);
    pointsB = cornerPoints([jB, iB]);

    % Randomly select maxCorners points
    if length(pointsA) > maxCorners
        indicesA = randperm(length(pointsA), maxCorners);
        pointsA = pointsA(indicesA);
    end

    if length(pointsB) > maxCorners
        indicesB = randperm(length(pointsB), maxCorners);
        pointsB = pointsB(indicesB);
    end
    %}
    % Extract features for the corners
    [featuresA, pointsA] = extractFeatures(imgA, pointsA);
    [featuresB, pointsB] = extractFeatures(imgB, pointsB);
    
    % Match features which were computed from the current and the previous images
    indexPairs = matchFeatures(featuresA, featuresB);
    pointsA = pointsA(indexPairs(:, 1), :);
    pointsB = pointsB(indexPairs(:, 2), :);

    % Estimate affine transformation using RANSAC
    [tform, inlierPointsA, inlierPointsB] = estimateGeometricTransform(pointsB, pointsA, 'affine', 'MaxDistance', 4, 'Confidence', 99.99, 'MaxNumTrials', 2000);
    tformAffine = tform.T;

    % Convert a 3x3 affine transform to a scale-rotation-translation transform
    % Extract rotation and translation submatrices
    R = tformAffine(1:2, 1:2);
    translation = tformAffine(3, 1:2);
    theta = mean([atan2(R(2), R(1)) atan2(-R(3), R(4))]);
    scale = mean(R([1 4]) / cos(theta));
    R = [cos(theta) -sin(theta); sin(theta) cos(theta)];
    sRtTform = [[scale*R; translation], [0 0 1]'];

    % Update the transformation buffer with the new transformation
    transformBuffer(:, :, bufferIndex) = sRtTform;
    bufferIndex = mod(bufferIndex, N) + 1;
    
    % Calculate the moving average of the transformations
    cumulativeTform = mean(transformBuffer, 3);
    
    % Create a structure for imwarp
    tformStruct = maketform('affine', cumulativeTform);
    % Warp the current frame
    imgBp = imtransform(imgB, tformStruct, 'XData', [1 size(imgB, 2)], 'YData', [1 size(imgB, 1)]);

    % Write the frame to the output video
    writeVideo(outputVideo, imgBp);

    % Display as color composite with last corrected frame
    step(hVPlayer, imfuse(imgAp, imgBp, 'ColorChannels', 'red-cyan'));
    correctedMean = correctedMean + imgBp;
    
    ii = ii + 1;
end
%%
correctedMean = correctedMean / (ii - 2);
movMean = movMean / (ii - 2);
release(hVPlayer);
close(outputVideo);
