function track_colour(colour, source, max_frames)
%TRACK_COLOUR Track red, green or blue objects live or in a video file.
%   track_colour                    % asks for a colour, uses webcam 1
%   track_colour('red')             % webcam
%   track_colour('blue', 'clip.mp4')
%   track_colour('green', [], 400)  % stop after 400 frames
%
%   Each detected object gets a yellow bounding box, a magenta centroid
%   marker and its (x, y) position.
if nargin < 1 || isempty(colour)
    colour = questdlg('Which colour would you like to track?', 'Colour', ...
                      'Red', 'Green', 'Blue', 'Red');
    if isempty(colour), return; end
end
if nargin < 3, max_frames = 400; end

use_file = nargin >= 2 && ~isempty(source);
if use_file
    reader = VideoReader(source);
else
    % Image Acquisition Toolbox; change 'winvideo' to 'macvideo' or
    % 'linuxvideo' on other platforms.
    vid = videoinput('winvideo', 1);
    set(vid, 'FramesPerTrigger', Inf, 'ReturnedColorspace', 'rgb');
    vid.FrameGrabInterval = 2;
    start(vid);
    cleanup = onCleanup(@() stop_camera(vid));
end

fig = figure('Name', sprintf('Tracking %s objects', lower(colour)));
n = 0;
while ishandle(fig) && n < max_frames
    if use_file
        if ~hasFrame(reader), break; end
        frame = readFrame(reader);
    else
        frame = getsnapshot(vid);
    end
    n = n + 1;
    stats = detect_colour_blobs(frame, colour);

    imshow(frame);
    hold on
    for k = 1:numel(stats)
        bc = stats(k).Centroid;
        rectangle('Position', stats(k).BoundingBox, 'EdgeColor', 'y', 'LineWidth', 2);
        plot(bc(1), bc(2), 'm+', 'MarkerSize', 10, 'LineWidth', 2);
        text(bc(1) + 15, bc(2), sprintf('X: %d  Y: %d', round(bc(1)), round(bc(2))), ...
             'FontWeight', 'bold', 'FontSize', 8, 'Color', 'k', 'BackgroundColor', 'w');
    end
    hold off
    drawnow;
end
end

function stop_camera(vid)
stop(vid);
flushdata(vid);
delete(vid);
end
