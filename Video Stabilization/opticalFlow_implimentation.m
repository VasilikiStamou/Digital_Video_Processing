function [u, v] = opticalFlow_implimentation(im1, im2, windowSize)
    % Convert images to double precision
    im1 = double(im1);
    im2 = double(im2);
    
    % Compute image gradients    
    Ix = conv2(im1, [-1 1; -1 1], 'same'); % gradient in x
    Iy = conv2(im1, [-1 -1; 1 1], 'same'); % gradient in y
    It = im2 - im1;
    %It = conv2(im2, ones(2), 'same') - conv2(im1, ones(2), 'same'); % gradient in t

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

    % Initialize flow vectors
    u = zeros(size(im1));
    v = zeros(size(im1));

    % Threshold for eigenvalues
    threshold = 35*10^3;
    % Compute optical flow
    for i = 1:size(im1, 1)
        for j = 1:size(im1, 2)
            
           % Ensure window stays within image boundaries
            minX = max(1, i - floor(windowSize / 2));
            maxX = min(size(im1, 1), i + floor(windowSize / 2));
            minY = max(1, j - floor(windowSize / 2));
            maxY = min(size(im1, 2), j + floor(windowSize / 2));
            
            %Construct matrix A
            Ix2_block = Ix2(minX:maxX, minY:maxY);
            Iy2_block = Iy2(minX:maxX, minY:maxY);
            Ixy_block = Ixy(minX:maxX, minY:maxY);
            Ixt_block = Ixt(minX:maxX, minY:maxY);
            Iyt_block = Iyt(minX:maxX, minY:maxY);
            
            ATA = [sum(Ix2_block(:)), sum(Ixy_block(:)); sum(Ixy_block(:)), sum(Iy2_block(:))];
            ATb = -[sum(Ixt_block(:)); sum(Iyt_block(:))];
            
            % Compute eigenvalues of ATA
            eigenvalues = eig(ATA);
            lambda1 = eigenvalues(1);
            lambda2 = eigenvalues(2);
            
            % Check the four cases based on eigenvalues and threshold
            if lambda1 < threshold || lambda2 < threshold %new
                % Case 1: Both eigenvalues are small (flat region)
                u(i, j) = 0;
                v(i, j) = 0;
            elseif lambda1 > 10*lambda2
                % Case 2: lambda1 >> lambda2 (edge)
                flow = ATA\ATb;
                u(i, j) = flow(1); 
                v(i, j) = flow(2); 
            elseif lambda2 > 10*lambda1
                % Case 3: lambda2 >> lambda1 (edge)
                flow = ATA\ATb; 
                u(i, j) = flow(1); 
                v(i, j) = flow(2); 
            elseif (lambda1 > lambda2 || lambda2 > lambda1) && abs(lambda1 -lambda2)<10^5           
                % Case 4: Both eigenvalues are large and well-conditioned
                % matrix (corner)
                flow = ATA\ATb;
                u(i, j) = flow(1);
                v(i, j) = flow(2);
            else
                u(i, j) = 0;
                v(i, j) = 0;
            end
        end
    end
end
