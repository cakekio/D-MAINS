clc;
close all;
clear;

folder = dir('*.tif'); % Identify all .tif files in the current folder

for k = 1:length(folder)   
    name = getfield(folder, {k}, 'name'); % Isolate names from struct folder information

    images = imread(name);
    imageData = im2gray(images(:, :, 1));
 
    %% FORMAT MASK
    contrastAdjusted = adapthisteq(imageData, 'ClipLimit', 0.05);  % Adjust the 'ClipLimit' value as needed (0.1 is just an example)
    se = strel('disk', 15);
    data2 = imclearborder(contrastAdjusted); % Eliminate objects on the border
    data3 = wiener2(data2, [5 5]); % Adaptive filtering through a 5x5 window
    dilatedImg = imdilate(data3, se);
    
    % Create structuring element for erosion
    seErode = strel('disk', 3); % Adjust the size of the structuring element as needed
    
    % Perform erosion
    erodedImage = imerode(dilatedImg, seErode);
    
    % Fill the gaps between edges
    filledImage = imfill(erodedImage, 'holes');


    bw1 = imbinarize(filledImage); % Binarize the data
    bw2 = imfill(bw1, 'holes'); % Fill in the holes
    bw3 = imopen(bw2, strel('disk', 4)); % Open with a disk
    se = strel('disk', 5); % Define structuring element
    erodedImage = imerode(bw3, se);
    bw4 = bwareaopen(erodedImage, 2000);
   % Display the original and dilated images
%     subplot(1,3,1); imshow(bw1); title('Binarized and Filled Image');
%     subplot(1,3,2); imshow(erodedImage); title('Eroded Image');
%     subplot(1,3,3); imshow(imageData); title('Object Recognition');
%     
    cc = bwconncomp(bw4, 26);%identify clusters of pixels comprising cells
    CellCounts(k)  = cc.NumObjects;

    AreaStruct = regionprops(cc,'Area');

    centers = regionprops(bw4,'Centroid');%identify centroids of those clusters
    centroids=zeros(CellCounts(k),2);%create empty array to fill with centroid locations

mkdir CATEGORY_All %create folder in current working path

%keep categories together, subfolders will be auto-generated. 
% Example: B1_All will have subfolders B1_1, B1_2.....B1_n for number of images

marked = zeros(size(imageData), 'uint8');
marked(bw4) = imageData(bw4);


    for n = 1:length(centers) 
        
        centroids(n,:)=getfield(centers,{n},'Centroid'); %remove centroid location

        x=centroids(n,1);%x value of centroid
        y=centroids(n,2);%y value of centroid

        cropped = imcrop(imageData,[x-50 y-50 99 99]); %crop a 90X90 window around the x and y coordinates
        filename=sprintf('%s_%d_%s_%d.tif','CATEGORY',k,'cell',n); %create name of image file (rename category)
        newname=string(filename);
        names(n,k)=newname;
        path=pwd; %identify current path (up to new croppedimages folder)
        
        fulldestination=fullfile(path,'CATEGORY_All',filename); %(rename category) 
        %define entire destination using current path +new folder
        
        pixels=size(cropped);
  
        Area(n,k)=getfield(AreaStruct,{n},'Area'); %get matrix of all areas
        
        %imshow(contrastAdjusted);
       
%         if Area(n,k)>250 && Area(n,k)<1100 && isequal(pixels, [40,40])
        imwrite(cropped, fulldestination); %write image to defined path
%             area_name(i)=Area(n,k);
%             name_area(i)=names(n,k);
%             i=i+1;
%            
%         end
        
    end  
end