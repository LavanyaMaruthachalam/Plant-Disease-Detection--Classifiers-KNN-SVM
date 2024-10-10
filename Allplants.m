folder = '/MATLAB Drive/Plant_leaf_diseases_dataset_without_augmentation/Plant_leaf_diseases_dataset_without_augmentation/Plant_leave_diseases_dataset_without_augmentation';
imds = imageDatastore(folder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');


% Resize images for consistency
imds.ReadFcn = @(filename)imresize(imread(filename), [256, 256]);

% Select only the first 100 images
numImages = 50;
imdsSubset = subset(imds, 1:numImages);

% Dataset splitting
[trainingSet, testSet] = splitEachLabel(imdsSubset, 0.7, 'randomize');

% Feature extraction for training
trainingFeatures = [];
trainingLabels = [];
for i = 1:50 % Loop through the first 100 images
    [img, info] = read(trainingSet);
    features = extractFeatures(img);
    trainingFeatures = [trainingFeatures; features];
    trainingLabels = [trainingLabels; info.Label];

    % Visualize only the first 2 images
    if i <= 2
        figure;
        imshow(img);
        title(['Training Image ', num2str(i)]);

        [segmented, edges] = segmentAndDetectEdges(img);
        % The segmentAndDetectEdges function already includes visualization
    end
end

% Train SVM for multi-class classification
svmModel = fitcecoc(trainingFeatures, trainingLabels);

% Train KNN
knnModel = fitcknn(trainingFeatures, trainingLabels);

% Feature extraction for testing
testFeatures = [];
testLabels = [];
for i = 1:50 % Loop through the first 100 images
    [img, info] = read(testSet);
    features = extractFeatures(img);
    testFeatures = [testFeatures; features];
    testLabels = [testLabels; info.Label];
end

% Model evaluation
svmPred = predict(svmModel, testFeatures);
svmAccuracy = sum(svmPred == testLabels) / numel(testLabels);
knnPred = predict(knnModel, testFeatures);
knnAccuracy = sum(knnPred == testLabels) / numel(testLabels);

% Confusion matrix for SVM
figure;
confusionchart(testLabels, svmPred);
title('SVM Confusion Matrix');

% Confusion matrix for KNN
figure;
confusionchart(testLabels, knnPred);
title('KNN Confusion Matrix');

% Function definitions

function features = extractFeatures(image)
    % Extract color features
    r = mean2(image(:,:,1));
    g = mean2(image(:,:,2));
    b = mean2(image(:,:,3));
    features = [r, g, b];
end

function [segmented, edges] = segmentAndDetectEdges(image)
    % Convert to grayscale and visualize
    grayImage = rgb2gray(image);
    figure;
    imshow(grayImage);
    title('Grayscale Image');

    % Edge detection and visualization
    edges = edge(grayImage, 'Canny');
    figure;
    imshow(edges);
    title('Edge Detection');

    % Image segmentation and visualization
    level = graythresh(grayImage);
    segmented = imbinarize(grayImage, level);
    figure;
    imshow(segmented);
    title('Segmented Image');
end
