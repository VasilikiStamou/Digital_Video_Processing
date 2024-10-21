function ret = videoStabilization1(invideo, outvideo)
    try
        % Load video file
        InputVid = vision.VideoFileReader(invideo);
        output_object = VideoWriter(outvideo);
        open(output_object);

        % Initialize first frame
        fprintf('Start processing frames \n')
        I1 = rgb2gray(step(InputVid));
        u = zeros(size(I1));
        v = zeros(size(I1));
        writeVideo(output_object, I1);

        % Parameters
        windowSize = 7;  
        numOfFrames = 200;  

        for k = 2:numOfFrames
            % Protection against too short video
            if isDone(InputVid)
                break
            end
            I2 = rgb2gray(step(InputVid));

            % Compute flow
            [du, dv] = opticalFlow_implimentation(I1, I2, windowSize);
            %{
            opticFlow = opticalFlowLK('NoiseThreshold', 0.01); 
            estimateFlow(opticFlow, I1);
            flow = estimateFlow(opticFlow, I2);
            du = flow.Vx;
            dv = flow.Vy;
            %}
            u = u + du;
            v = v + dv;

            % Perform averaging to smooth the result
            u = imboxfilt(u, 5);
            v = imboxfilt(v, 5);

            % Generate grid
            [X, Y] = meshgrid(1:size(I2, 2), 1:size(I2, 1));
            
            % Warp the current frame (distort the current frame)
            I2_warp = interp2(X, Y, I2, X + u, Y + v);
            I2_warp(isnan(I2_warp)) = I2(isnan(I2_warp));

            % Save to file
            writeVideo(output_object, I2_warp);
            I1 = I2;

            fprintf('finished frame number = %d/%d\n', k, numOfFrames);
        end

        % Close the output video object
        close(output_object);
        release(InputVid);

        % Return 1 indicating success
        ret = 1;
    catch
        % Return 0 indicating failure
        ret = 0;
    end
end