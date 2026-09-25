function stats = detect_colour_blobs(frame, colour, threshold, min_area)
%DETECT_COLOUR_BLOBS Find red, green or blue objects in an RGB frame.
%   stats = detect_colour_blobs(frame, 'red')
%   stats = detect_colour_blobs(frame, 'green', 0.05, 300)
%
%   Subtracting the grayscale image from one colour channel leaves only
%   pixels that are much stronger in that channel than overall (a red ball
%   stays bright, a white wall cancels out). The difference is median
%   filtered, thresholded, cleaned of blobs smaller than min_area pixels and
%   labelled. Returns regionprops structs with BoundingBox, Centroid and Area.
if nargin < 3 || isempty(threshold)
    threshold = default_threshold(colour);
end
if nargin < 4
    min_area = 300;
end
channel = channel_index(colour);
diff_im = imsubtract(frame(:, :, channel), rgb2gray(frame));
diff_im = medfilt2(diff_im, [3 3]);
mask = imbinarize(diff_im, threshold);
mask = bwareaopen(mask, min_area);
stats = regionprops(bwlabel(mask, 8), 'BoundingBox', 'Centroid', 'Area');
end

function c = channel_index(colour)
switch lower(colour)
    case 'red',   c = 1;
    case 'green', c = 2;
    case 'blue',  c = 3;
    otherwise, error('colour must be red, green or blue');
end
end

function t = default_threshold(colour)
% Thresholds tuned on the webcam used for the project. Green needs a much
% lower threshold because the green channel dominates the luminance
% (grayscale) weights, so its difference image is weaker.
switch lower(colour)
    case 'red',   t = 0.24;
    case 'green', t = 0.05;
    case 'blue',  t = 0.15;
end
end
