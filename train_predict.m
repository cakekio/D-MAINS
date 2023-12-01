clear;
clc;
close all;

%% load images

%Identify path of TRAIN folder with fullfile
% 
trainpath=fullfile('/Users/sarahhe/Downloads/training1');


%label all images in the path based on folder names
imds = imageDatastore(trainpath, 'IncludeSubfolders',true,'LabelSource','foldernames');

%eliminate images that aren't 100X100
for k = 1:length(imds.Files)

 imageData=imread(imds.Files{k});
 pixels=size(imageData);
 
    if ~isequal(pixels, [100,100])
        delete(imds.Files{k})
    end
end


labelCount = countEachLabel(imds); %count labels and images they contain

percentTrain=0.8; %use percentage of each file instead of a certain number
[imdsTrain,imdsValidation] = splitEachLabel(imds,percentTrain,'randomize');



%% create layers
layers = [
    imageInputLayer([100 100 1])% for 100X100 pixel images
   
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)

   
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer
   
    maxPooling2dLayer(2,'Stride',2)
   
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer
     
    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,128,'Padding','same')
    batchNormalizationLayer
    reluLayer
       
    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,128,'Padding','same')
    batchNormalizationLayer
    reluLayer
   
    maxPooling2dLayer(2,'Stride',2)
   
    convolution2dLayer(3,128,'Padding','same')
    batchNormalizationLayer
    reluLayer
       
    fullyConnectedLayer(6)
    softmaxLayer
 
    classificationLayer

    ];

options = trainingOptions('adam', ...
    'InitialLearnRate',0.01, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',10, ...
    'MaxEpochs',20, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',50, ...
    'Verbose',true, ...
    'MiniBatchSize',128, ...
    'L2Regularization',0.005, ...
    'Plots','training-progress', ...
    'ValidationPatience',5);

Newest_Net = trainNetwork(imdsTrain,layers,options);
save Newest_Net;

%% Confusion Matrix

predicted=classify(Newest_Net,imdsValidation);
actual=imdsValidation.Labels;

plotconfusion(actual,predicted)


%% Copy Incorrectly Predicted Images

% Create output folder
outputFolder = 'incorrect_images';
mkdir(outputFolder);

% Get file paths and predicted labels
imdsTest = imageDatastore(trainpath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
YPred = classify(Newest_Net, imdsTest);
YFiles = imdsTest.Files;

% Iterate over each image
for i = 1:numel(YFiles)
    if YPred(i) ~= imdsTest.Labels(i)
        [~, filename, ext] = fileparts(YFiles{i});
        correctCategory = char(imdsTest.Labels(i));
        subfolder = fullfile(outputFolder, correctCategory);
        mkdir(subfolder);
        newFilename = sprintf('%s_%s%s', filename, char(YPred(i)), ext);
        destination = fullfile(subfolder, newFilename);
        copyfile(YFiles{i}, destination);
    end
end

% %% Prediction
% newImagesPath = '/Users/sarahhe/Downloads/test';
% imdsNew = imageDatastore(newImagesPath);
% 
% %eliminate images that aren't 90X90
% for k = 1:length(imdsNew.Files)
% 
%  imageData=imread(imdsNew.Files{k});
%  pixels=size(imageData);
%  
%     if ~isequal(pixels, [100,100])
%         delete(imdsNew.Files{k})
%     end
% end
% 
% YPred = classify(Newest_Net, imdsNew);
% 
% % Get the predicted labels and count the occurrences of each category
% predictedLabels = countcats(YPred);
% categoryNames = categories(YPred);
% categoryCounts = zeros(numel(categoryNames), 1);
% 
% % % Create folders for each category
% % for i = 1:numel(categoryNames)
% %     categoryFolder = fullfile(newImagesPath, categoryNames{i});
% %     if ~exist(categoryFolder, 'dir')
% %         mkdir(categoryFolder);
% %     end
% % end
% 
% 
% % Copy the images to the respective category folders
% for i = 1:length(imdsNew.Files)
%     % Get the predicted category for the image
%     predictedCategory = categoryNames{YPred(i)};
% 
%     % Get the file path and name
%     imagePath = imdsNew.Files{i};
%     [~, imageName, imageExt] = fileparts(imagePath);
%     
% % %     % Create the destination folder path
% %     destinationFolder = fullfile(newImagesPath, predictedCategory);
% % %     
% % %     % Copy the image to the destination folder
% %     destinationPath = fullfile(destinationFolder, [imageName, imageExt]);
% %     copyfile(imagePath, destinationPath);
%     
%     % Update the category count
%     categoryCounts(strcmp(categoryNames, predictedCategory)) = ...
%         categoryCounts(strcmp(categoryNames, predictedCategory)) + 1;
% end
% 
% % Display the category counts
% categoryTable = table(categoryNames, categoryCounts, 'VariableNames', {'Category', 'Count'});
% disp(categoryTable);

