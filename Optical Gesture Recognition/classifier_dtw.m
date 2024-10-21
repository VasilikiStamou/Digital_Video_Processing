clear; clc; close all;
% Options: 'position_only', 'position_and_shape'
featureMode = 'position_and_shape'; 
%% Define the base folder containing all gesture datasets
baseFolder = pwd; 

% Define subfolders for each gesture category and the number of sequences
gestureFolders = {'Clic', 'No', 'Rotate', 'StopGraspOk'};
numSeqPerFolder = [15, 14, 13, 15];  

% Define the number of training sequences per gesture
numTrainSeq = 5;

% Initialize variables for storing training and testing data
trainingData = [];
trainingLabels = [];
testingData = [];
testingLabels = [];

%% Extract features from each gesture folder
for gestureIdx = 1:length(gestureFolders)
    folderName = gestureFolders{gestureIdx};
    numSeq = numSeqPerFolder(gestureIdx);
    
    % Loop through the sequences within each gesture folder
    for seqIdx = 1:numSeq
        % Set the folder for the current sequence
        seqFolder = fullfile(baseFolder, folderName, ['Seq' num2str(seqIdx)]);
        
        % List all .pnm files in the current sequence folder
        imageFiles = dir(fullfile(seqFolder, '*.pnm'));
        
        % Extract features (position and shape) for the current sequence
        features = extractFeatures(seqFolder, imageFiles,featureMode);
        
        if seqIdx <= numTrainSeq
            % Store features and label for training data
            trainingData = [trainingData; {features}];
            trainingLabels = [trainingLabels; gestureIdx];
        else
            % Store features and label for testing data
            testingData = [testingData; {features}];
            testingLabels = [testingLabels; gestureIdx];
        end
    end
end

%% Testing with Different k Values
ks = 1:10;
accuracies = zeros(length(ks), 1);

% Initialize cell array to store confusion matrices
confMatrices = cell(length(ks), 1);

for i = 1:length(ks)
    predictedLabels = zeros(size(testingLabels));
    
    % Loop through each test sequence
    for j = 1:length(testingLabels)
        % Get the current test sample
        testSample = testingData{j};
        
        % Calculate DTW distance between the test sample and all training samples
        dtwDistances = zeros(1, length(trainingLabels));
        for k = 1:length(trainingLabels)
            dtwDistances(k) = dtwDistance(testSample, trainingData{k});
        end
        
        % Find the k nearest neighbors based on DTW distance
        [~, nnIndices] = sort(dtwDistances, 'ascend');
        nnIndices = nnIndices(1:ks(i));
        
        % Predict the label based on majority voting
        predictedLabels(j) = mode(trainingLabels(nnIndices));
    end
    
    % Generate confusion matrix
    confMatrix = confusionmat(testingLabels, predictedLabels);
    % Store confusion matrix for current k
    confMatrices{i} = confMatrix;
    % Calculate accuracy
    accuracies(i) = sum(diag(confMatrix)) / sum(confMatrix(:));
end
%%
ConfMatrix = confMatrices{3};
% Display Confusion Matrix for k=3
disp('Confusion Matrix:');
disp(ConfMatrix);
% Display Total Accuracy for k=3
accuracy = sum(diag(ConfMatrix)) / sum(ConfMatrix(:));
disp(['Overall Accuracy: ', num2str(accuracy * 100), '%']);
% Display Per-Class Accuracy for k=3
classAccuracy = diag(ConfMatrix) ./ sum(ConfMatrix, 2);
disp('Per-Class Accuracy:');
disp(classAccuracy);
% Confusion Matrix Heatmap
figure;
heatmap(ConfMatrix, 'Title', 'Confusion Matrix', 'XLabel', 'Predicted Class', 'YLabel', 'True Class',...
    'XDisplayLabels', gestureFolders, 'YDisplayLabels', gestureFolders);


% Plot k vs accuracy
figure;
plot(ks, accuracies, '-o');
xlabel('k');
ylabel('Accuracy');
title('k-NN Accuracy for Different k Values with DTW');

% Display Confusion Matrices
figure;
for i = 1:length(ks)
    subplot(2, 5, i);
    heatmap(confMatrices{i}, 'Title', ['Confusion Matrix for k = ' num2str(ks(i))], ...
            'XLabel', 'Predicted Class', 'YLabel', 'True Class', ...
            'XDisplayLabels', gestureFolders, 'YDisplayLabels', gestureFolders);
end

%% Function to Extract Features
function features = extractFeatures(seqFolder, imageFiles,featureMode)
    features = [];
    
    for i = 1:length(imageFiles)
        img = imread(fullfile(seqFolder, imageFiles(i).name));
        
        % Reshape image into a 2D array where each row is a pixel, and columns are the RGB values
        reshapedImg = reshape(img, [], 3);

        % Estimate the background color as the most common color in the image
        backgroundColor = mode(double(reshapedImg), 1);

        % Calculate the Euclidean distance of each pixel to the background color
        distanceFromBackground = sqrt(sum((double(reshapedImg) - backgroundColor).^2, 2));

        % Reshape the distance array back to the image dimensions
        distanceImg = reshape(distanceFromBackground, size(img, 1), size(img, 2));

        % Threshold the distance image to create a binary mask 
        handMask = distanceImg > 90;  
        
        % Get the position of the hand (center of mass)
        [rows, cols] = find(handMask);
        if isempty(rows)
            continue;
        end
        
        centerX = mean(cols);
        centerY = mean(rows);
        
        % Normalize the features (position and shape)
        normalizedCenterX = centerX / 58;
        normalizedCenterY = centerY / 62;
        
        switch featureMode
            case 'position_only'
                %% Step (a) - Hand represented as position (i, j) only
                features = [features; normalizedCenterX, normalizedCenterY];
                
            case 'position_and_shape'
                %% Step (b) - Hand represented as position (i, j) and shape (bounding box dimensions)
                % Get the bounding box of the hand
                minRow = min(rows);
                maxRow = max(rows);
                minCol = min(cols);
                maxCol = max(cols);
                
                width = maxCol - minCol;
                height = maxRow - minRow;
                
                % Normalize the shape features (width and height)
                normalizedWidth = width / 58;
                normalizedHeight = height / 62;
                
                % Combine position and shape features
                features = [features; normalizedCenterX, normalizedCenterY, normalizedWidth, normalizedHeight];
        end                             
    end
end

%% Function to Compute DTW Distance
function dist = dtwDistance(seq1, seq2)
    % Compute DTW distance between two sequences
    n = size(seq1, 1);
    m = size(seq2, 1);
    
    % Initialize cost matrix
    costMatrix = inf(n+1, m+1);
    costMatrix(1, 1) = 0;
    
    % Compute cost matrix
    for i = 1:n
        for j = 1:m
            cost = sum((seq1(i, :) - seq2(j, :)).^2);
            costMatrix(i+1, j+1) = cost + min([costMatrix(i, j+1), costMatrix(i+1, j), costMatrix(i, j)]);
        end
    end
    
    % DTW distance is the value in the bottom-right corner of the cost matrix
    dist = sqrt(costMatrix(n+1, m+1));
end
