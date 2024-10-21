function mei = extractMEI(seqFolder, imageFiles)
    % Initialize MEI
    img = imread(fullfile(seqFolder, imageFiles(1).name));
    mei = zeros(size(img));
    
    % Loop through all images in the sequence
    for i = 2:length(imageFiles)
        % Read current and previous images
        imgPrev = imread(fullfile(seqFolder, imageFiles(i-1).name));
        imgCurr = imread(fullfile(seqFolder, imageFiles(i).name));
        
        % Calculate the difference image (binary)
        diffImg = abs(double(imgCurr) - double(imgPrev)) > 39; % Threshold for motion
        
        % Update MEI (aggregate motion)
        mei = mei + diffImg;
    end
    
    % Normalize MEI and convert to double
    mei = mat2gray(mei);  % Scale MEI to 0-1
    mei = double(mei);   

end
