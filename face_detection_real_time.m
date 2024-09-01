% Initialize video reader
the_video = VideoReader("4_5824708160048334074.mp4");

if ~hasFrame(the_video)
    error('The video file is empty or cannot be read.');
end

% Read the first frame
video_frame = readFrame(the_video);

% Initialize face detector
face_detector = vision.CascadeObjectDetector();

% Detect face in the first frame
location_of_the_face = step(face_detector, video_frame);

% Check if a face was detected
if isempty(location_of_the_face)
    error('No face detected in the first frame.');
end

% Manually calculate the corners of the bounding box
rectangle_to_points = [
    location_of_the_face(1,1), location_of_the_face(1,2); % top-left
    location_of_the_face(1,1) + location_of_the_face(1,3), location_of_the_face(1,2); % top-right
    location_of_the_face(1,1) + location_of_the_face(1,3), location_of_the_face(1,2) + location_of_the_face(1,4); % bottom-right
    location_of_the_face(1,1), location_of_the_face(1,2) + location_of_the_face(1,4); % bottom-left
    location_of_the_face(1,1), location_of_the_face(1,2) % back to top-left
];

% Detect feature points
gray_frame = rgb2gray(video_frame);
feature_points = detectMinEigenFeatures(gray_frame, 'ROI', location_of_the_face);
feature_points = feature_points.Location;

if size(feature_points, 1) < 10
    error('Insufficient feature points detected.');
end

% Initialize point tracker
pointTracker = vision.PointTracker('MaxBidirectionalError', 2);
initialize(pointTracker, feature_points, video_frame);

% Initialize video player
left = 100;
bottom = 100;
width = size(video_frame, 2);
height = size(video_frame, 1);
video_player = vision.VideoPlayer('Position', [left bottom width height]);

previous_points = feature_points;

while hasFrame(the_video)
    video_frame = readFrame(the_video);

    % Track the points
    [feature_points, isFound] = step(pointTracker, video_frame);
    new_points = feature_points(isFound, :);
    old_points = previous_points(isFound, :);

    if size(new_points, 1) >= 2
        % Estimate geometric transformation
        [transformed_rectangle, old_points, new_points] = ...
            estimateGeometricTransform(old_points, new_points, ...
            'similarity', 'MaxDistance', 4);
        rectangle_to_points = transformPointsForward(transformed_rectangle, rectangle_to_points);

        % Reshape rectangle to polygon
        reshaped_rectangle = reshape(rectangle_to_points', 1, []);
        detected_frame = insertShape(video_frame, "polygon", reshaped_rectangle, "LineWidth", 2);

        % Insert markers
        detected_frame = insertMarker(detected_frame, new_points, '+', 'Color', 'White');

        % Update the previous points
        previous_points = new_points;
        setPoints(pointTracker, previous_points);
    else
        % If no points are found, continue without updating the tracker
        detected_frame = video_frame; % Show the frame as is
    end

    % Display the frame
    step(video_player, detected_frame);
end

% Release video player
release(video_player);

% Close any remaining figures
close all;
