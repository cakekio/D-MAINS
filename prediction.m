load('Newest_Net.mat');

newImagesPath = 'E:\Gong Lab\231025_152302_IMM Session\231025_153046_A1 20X PL FL Phase (6)\CATEGORY_All';
imdsNew = imageDatastore(newImagesPath);

%eliminate images that aren't 100X100
for k = 1:length(imdsNew.Files)

 imageData=imread(imdsNew.Files{k});
 pixels=size(imageData);
 
    if ~isequal(pixels, [100,100])
        delete(imdsNew.Files{k})
    end
end

YPred = classify(Newest_Net, imdsNew);

% Get the predicted labels and count the occurrences of each category
predictedLabels = countcats(YPred);
categoryNames = categories(YPred);
categoryCounts = zeros(numel(categoryNames), 1);

% Create folders for each category
for i = 1:numel(categoryNames)
    categoryFolder = fullfile(newImagesPath, categoryNames{i});
    if ~exist(categoryFolder, 'dir')
        mkdir(categoryFolder);
    end
end


% Copy the images to the respective category folders
for i = 1:length(imdsNew.Files)
    % Get the predicted category for the image
    predictedCategory = categoryNames{YPred(i)};

    % Get the file path and name
    imagePath = imdsNew.Files{i};
    [~, imageName, imageExt] = fileparts(imagePath);
    
%     % Create the destination folder path
    destinationFolder = fullfile(newImagesPath, predictedCategory);
%     
%     % Copy the image to the destination folder
    destinationPath = fullfile(destinationFolder, [imageName, imageExt]);
    copyfile(imagePath, destinationPath);
    
    % Update the category count
    categoryCounts(strcmp(categoryNames, predictedCategory)) = ...
        categoryCounts(strcmp(categoryNames, predictedCategory)) + 1;
end

% Display the category counts
categoryTable = table(categoryNames, categoryCounts, 'VariableNames', {'Category', 'Count'});
disp(categoryTable);

