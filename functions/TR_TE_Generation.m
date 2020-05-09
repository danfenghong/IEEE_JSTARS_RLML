function [Training_samples, Test_samples, Training_labels, Test_labels, Training_locations, Test_locations] = TR_TE_Generation(Image, GT, num_class)

%% Training and test samples generation
% Image: M * N *D (length * width * bands)
% GT (ground truth): M * N
% num_class: the number of training samples for each class
 
[M, N, ~] = size(Image);
% The number of classes
num = length(unique(GT)) - 1;

% 3D to 2D
samples = hyperConvert2d(Image);
labels = hyperConvert2d(GT);

% Generate coordiates to find the spatial neighbors

coordiate = zeros(M, N, 2);

for i = 1 : M
    for j = 1: N
        coordiate(i, j, :) = [i;j];
    end
end

locations = hyperConvert2d(coordiate);

Training_samples = [];
Test_samples = [];
Training_labels = [];
Test_labels = [];
Training_locations = [];
Test_locations = [];

for i = 1 : num
    
    index = find(labels == i);
    sub_smaples = samples(:, index);
    sub_labels = labels(:,index);
    sub_locations = locations(:,index);
    
    random_num = randperm(length(index)); % random number generation
    Training_samples = [Training_samples, sub_smaples(:, random_num(1 : num_class(i)))];
    Training_labels = [Training_labels, sub_labels(:, random_num(1 : num_class(i)))];
    Test_samples = [Test_samples, sub_smaples(:, random_num(num_class(i) + 1 : end))];
    Test_labels = [Test_labels, sub_labels(:, random_num(num_class(i) + 1 : end))];
    Training_locations = [Training_locations, sub_locations(:, random_num(1 : num_class(i)))];
    Test_locations = [Test_locations, sub_locations(:, random_num(num_class(i) + 1 : end))];
end

end