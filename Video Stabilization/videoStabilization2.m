function ret = videoStabilization2(invideo, outvideo)
    try
        % Read the input video
        hVideoSrc = VideoReader(invideo);
        outputVideo = VideoWriter(outvideo);
        open(outputVideo);
        
        % Initialize the video player
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
        maxCorners = 10^4;  % Maximum number of corners to detect
        %%
        while hasFrame(hVideoSrc)
            % Read in new frame
            imgA = imgB; 
            imgAp = imgBp;
            imgB = im2gray(im2single(readFrame(hVideoSrc)));
            movMean = movMean + imgB;
            
            % Generate prospective points
            %pointsA = detectFASTFeatures(imgA, 'MinContrast', 0.1);
            %pointsB = detectFASTFeatures(imgB, 'MinContrast', 0.1);
            
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
            
            % Extract features for the corners
            [featuresA, pointsA] = extractFeatures(imgA, pointsA);
            [featuresB, pointsB] = extractFeatures(imgB, pointsB);
            
            % Match features which were computed from the current and the previous images
            indexPairs = matchFeatures(featuresA, featuresB);
            pointsA = pointsA(indexPairs(:, 1), :);
            pointsB = pointsB(indexPairs(:, 2), :);
            
            % Estimate affine transformation manually
            tformAffine = estimateAffine(pointsA.Location, pointsB.Location)';
            
            % Convert a 3x3 affine transform to a scale-rotation-translation transform
            % Extract rotation and translation submatrices
            R = tformAffine(1:2, 1:2);
            % Translation remains the same:
            translation = tformAffine(3, 1:2);
            % Compute theta from mean of stable arctangents
            theta = mean([atan2(R(2), R(1)) atan2(-R(3), R(4))]);
            % Compute scale from mean of two stable mean calculations
            scale = mean(R([1 4])/cos(theta));
            % Reconstitute new s-R-T (scale-rotation-translation) transform
            R = [cos(theta) -sin(theta); sin(theta) cos(theta)];
            sRtTform = [[scale*R; translation], [0 0 1]'];
            
            % Update cumulative transformation
            cumulativeTform = cumulativeTform * sRtTform;
            % Create a structure for imwarp
            tformStruct = maketform('affine', cumulativeTform);
            % Warp the current frame
            imgBp = imtransform(imgB, tformStruct, 'XData', [1 size(imgB, 2)], 'YData', [1 size(imgB, 1)]);
            
            % Write the frame to the output video
            writeVideo(outputVideo, imgBp);
            
            % Display as color composite with last corrected frame
            step(hVPlayer, imfuse(imgAp, imgBp, 'ColorChannels', 'red-cyan'));
            correctedMean = correctedMean + imgBp;
            
            disp(['Processing frame ', num2str(ii)]);
            
            ii = ii + 1;
        end
        %%
        % Normalize the means
        correctedMean = correctedMean / (ii - 2);
        movMean = movMean / (ii - 2);
        
        % Release the video player and close the output video file
        release(hVPlayer);
        close(outputVideo);
        
        % Return 1 indicating success
        ret = 1;
    catch
        % Return 0 indicating failure
        ret = 0;
    end
end