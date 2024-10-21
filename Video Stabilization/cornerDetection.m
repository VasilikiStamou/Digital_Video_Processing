function [i,j] = cornerDetection(im,windowSize,thres)
    % Convert image to double precision
    im = double(im);
    
    % Compute image gradients    
    Ix = conv2(im, [-1 1; -1 1], 'same'); % gradient in x
    Iy = conv2(im, [-1 -1; 1 1], 'same'); % gradient in y
   
    % Gaussian weighting window
    gaussianWindow = fspecial('gaussian', [windowSize windowSize], 0.1);

    % Compute products of derivatives at every pixel
    Ix2 = Ix.^2;
    Iy2 = Iy.^2;
    Ixy = Ix.*Iy;
    
    % Apply Gaussian filter to these products
    Ix2 = conv2(Ix2, gaussianWindow, 'same');
    Iy2 = conv2(Iy2, gaussianWindow, 'same');
    Ixy = conv2(Ixy, gaussianWindow, 'same');
    
    [rows, cols] = size(im);
    R = zeros(rows, cols);
    % Compute optical flow
    for i = 1:rows
        for j = 1:cols
            
           % Ensure window stays within image boundaries
            minX = max(1, i - floor(windowSize / 2));
            maxX = min(size(im, 1), i + floor(windowSize / 2));
            minY = max(1, j - floor(windowSize / 2));
            maxY = min(size(im, 2), j + floor(windowSize / 2));
            
            %Construct matrix ATA
            Ix2_block = Ix2(minX:maxX, minY:maxY);
            Iy2_block = Iy2(minX:maxX, minY:maxY);
            Ixy_block = Ixy(minX:maxX, minY:maxY);
                       
            ATA = [sum(Ix2_block(:)), sum(Ixy_block(:)); sum(Ixy_block(:)), sum(Iy2_block(:))];    
    
            % Compute eigenvalues of ATA
            eigenvalues = eig(ATA);
            lambda1 = eigenvalues(1);
            lambda2 = eigenvalues(2);
            R(i,j) = lambda1 * lambda2 - 0.05 * (lambda1 + lambda2)^2;
        end
    end
    R(R < thres) = 0;
    corners = imregionalmax(R);
    % Find the coordinates of the corners
    [i, j] = find(corners);
end