%This is a toy example of how to use the encoding method ST-VLMPF, presented
%in the paper: "Spatio-Temporal Vector of Locally Max Pooled Features for Action Recognition in Videos", 
%Ionut Cosmin Duta; Bogdan Ionescu; Kiyoharu Aizawa; Nicu Sebe.
%In Computer Vision and Pattern Recognition (CVPR), 2017.
%
%Please cite our work when using this code!


%In this example we use "fake" features by generating them randomly. 
%Of course, in practice you should use features extracted from your data, as presented in the paper. 


%generate the features for creating the vocabulary

dimFeat = 256; %the size of the features
nFeatVocab = 5000; %the number of features from which the vocabulary is generated. Use much more features in real cases!!!!
featCreateVocab = rand(nFeatVocab, dimFeat);

%create the features vocabulary with the standard k-means, can be quite slow process
k1 = 256; % the number of visual words for the vocabulary of the features
[~, vocabFeatures] = kmeans(featCreateVocab, k1); 


%generate the position of the features for creating the vocabulary
posCreateVocab = rand(nFeatVocab, 3);

%create the vocabulary of the features position with the standard k-means
k2 = 32; % the number of visual words for the vocabulary of the features position
[~, vocabPositions] = kmeans(posCreateVocab, k2); 

nFeatures = 200;
%generate the features for which the encoding is performed
features = rand(nFeatures, dimFeat);

%generate the features position
positions = rand(nFeatures, 3);


%obtain the final ST-VLMPF encoding
ST_VLMPF_encoding = ST_VLMPF(features, vocabFeatures, positions, vocabPositions);

%after you obtain the final representation, before classification, 
%apply L2 normalization for making unit length
norm_ST_VLMPF_encoding = NormalizeRowsUnit(ST_VLMPF_encoding);


