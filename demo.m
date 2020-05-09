clc;
close all;
clear all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%data
addpath('Data'); %load data
addpath('functions'); %load functions
load Indian_pines.mat
load Indian_pines_gt.mat

% The number of training samples setting in our case
% If you want to run your own code, please set them accordingly
num_class = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 15, 10, 5, 5];

% Generate samples and labels as well as the corresponding spatial
% information used for learning the spatial-spectral feature representation
[Training_samples, Test_samples, Training_labels, Test_labels, Training_locations, Test_locations]...
= TR_TE_Generation(indian_pines, indian_pines_gt, num_class);

TR_TE_samples = [Training_samples, Test_samples];
Spatial_Neighbors_index = [Training_locations, Test_locations];

[D, ~] = size(TR_TE_samples);

% Run robust local manifold representation (RLMR) to generate the features
param.K = 2 * 100;
param.f_K = 100;
param.d = 80;
param.alfa = 0.2;

fea = RLMR(TR_TE_samples, Spatial_Neighbors_index, param);

% Training and test features
traindata_fea = fea(:, 1 : size(Training_samples, 2));
testdata_fea = fea(: , size(Training_samples, 2) + 1 : size(TR_TE_samples, 2));

% 1NN classifier
mdl = ClassificationKNN.fit(traindata_fea', Training_labels', 'NumNeighbors', 1, 'distance', 'euclidean');
characterClass = predict(mdl, testdata_fea');  
totalaccuracy = sum(characterClass == Test_labels') / length(Test_labels);

