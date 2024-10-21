clear; clc; close all;
%%
% Define the base folder containing all gesture datasets
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

%% Loop through each gesture folder (representing different classes)
for gestureIdx = 1:length(gestureFolders)
    folderName = gestureFolders{gestureIdx};
    numSeq = numSeqPerFolder(gestureIdx);
    
    % Loop through the sequences within each gesture folder
    for seqIdx = 1:numSeq
        % Set the folder for the current sequence
        seqFolder = fullfile(baseFolder, folderName, ['Seq' num2str(seqIdx)]);
        
        % List all .pnm files in the current sequence folder
        imageFiles = dir(fullfile(seqFolder, '*.pnm'));
        
        % Extract MEI for the current sequence
        mei = extractMEI(seqFolder, imageFiles);
        
        if seqIdx <= numTrainSeq
            % Store MEI and label for training data
            trainingData = [trainingData; mei(:)'];  % Flatten MEI to a vector
            trainingLabels = [trainingLabels; gestureIdx];
        else
            % Store MEI and label for testing data
            testingData = [testingData; mei(:)'];
            testingLabels = [testingLabels; gestureIdx];
        end
    end
end

%% Implement k-NN classifier
k = 3;  
mdl = fitcknn(trainingData, trainingLabels, 'NumNeighbors', k);

% Predict using the k-NN model
predictedLabels = predict(mdl, testingData);

%% Confusion Matrix
% Generate Confusion Matrix
confMatrix = confusionmat(testingLabels, predictedLabels);

% Display Confusion Matrix
disp('Confusion Matrix:');
disp(confMatrix);

% Confusion Matrix Heatmap
figure;
heatmap(confMatrix, 'Title', 'Confusion Matrix', 'XLabel', 'Predicted Class', 'YLabel', 'True Class',...
    'XDisplayLabels', gestureFolders, 'YDisplayLabels', gestureFolders);

% Display Total Accuracy
accuracy = sum(diag(confMatrix)) / sum(confMatrix(:));
disp(['Overall Accuracy: ', num2str(accuracy * 100), '%']);

% Display Per-Class Accuracy
classAccuracy = diag(confMatrix) ./ sum(confMatrix, 2);
disp('Per-Class Accuracy:');
disp(classAccuracy);

%% Testing with Different k Values
ks = 1:10;
accuracies = zeros(length(ks), 1);

% Initialize cell array to store confusion matrices
confMatrices = cell(length(ks), 1);

for i = 1:length(ks)
    % Train k-NN model with current k
    mdl = fitcknn(trainingData, trainingLabels, 'NumNeighbors', ks(i));
    % Predict using the k-NN model
    predictedLabels = predict(mdl, testingData);
    % Generate confusion matrix
    confMatrix = confusionmat(testingLabels, predictedLabels);
    % Store confusion matrix for current k
    confMatrices{i} = confMatrix;
    % Calculate accuracy
    accuracies(i) = sum(diag(confMatrix)) / sum(confMatrix(:));
end

% Plot k vs accuracy
figure;
plot(ks, accuracies, '-o');
xlabel('k');
ylabel('Accuracy');
title('k-NN Accuracy for Different k Values');

% Display Confusion Matrices
for i = 1:length(ks)
    figure;
    heatmap(confMatrices{i}, 'Title', ['Confusion Matrix for k = ' num2str(ks(i))], ...
            'XLabel', 'Predicted Class', 'YLabel', 'True Class', ...
            'XDisplayLabels', gestureFolders, 'YDisplayLabels', gestureFolders);
end
