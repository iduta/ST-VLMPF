function [ST_VLMPF_encoding] = ST_VLMPF(features, vocabFeatures, positions, vocabPositions, alpha)
%ST_VLMPF_encoding = ST_VLMPF(features, vocabFeatures, positions, vocabPositions)
%Compute ST-VLMPF (Spatio-Temporal Vector of Locally Max Pooled Features) 
%encoding for a set of features with their pozitions as presented in the paper:
%"Spatio-Temporal Vector of Locally Max Pooled Features for Action Recognition in Videos".
%
% input:
%    features: n x d matrix of features (n - number of features, d - the dimensionality of the features)
%    vocabFeatures: k1 x d the learned vocabulary of the features (k1 - number of visual words (centroids), d - the dimensionality of each visual word)
%    positions: n x 3 the features position (in our paper the asocieted position to each feature is represented by 3 values (x, y, t))
%    vocabPositions: k2 x 3 the learned vocabulary of the features position (k2 - the number of learned video divisions)
%    alpha: the parameter for the Power Normalization (for the membership information), default alpha=0.5   
% output
%    ST_VLMPF_encoding: k1×d + k2×(d+k1) ST_VLMPF encoding vector.
%
%       Ionut Cosmin Duta - 2017




d = size(features, 2); % % the feature dimensionality
k1 = size(vocabFeatures, 1);  % % the size of the vocabulary for the features



%Calculate the similarity using Euclidean distance
%"distmj" is a function which efficiently compute the Euclidean distance
%between two vectors
distance = distmj(features, vocabFeatures);

%perform the hard assignment for the features
[~, assign] = min(distance, [], 2);

%to retain VLMPF for each word of the vocabulary
VLMPF = cell(1, k1);

%to save the membership information for each feature
memb = zeros(size(features,1), k1);

%Compute VLMPF for each word of the vocabulary
for i = 1:k1
    
    assigned=(assign==i); % get the features assigned to the cluster i;
    
    if sum(assigned)>0 % compute VLMPF for each visual word (cluster) that has at least one features assigned
      
        %perform similarity max-pooling over grouped features (while keeping the initial sign for the returned final result)
        tFeat=features(assigned, :);
        [~, idx]=max(abs(tFeat), [], 1);
        VLMPF{i}=tFeat(sub2ind(size(tFeat), idx, 1:size(tFeat,2)));
        
        %save membership information for each feature
        memb(assigned, i)=1;
        
        
    else
        % no features in the cluser then put zeros
        VLMPF{i}=zeros(1, d); 

    end 
        
        
end

%Concatenate all the VLMPF vectors for each cluster to create the final
%VLMP vector
VLMPF=cat(2, VLMPF{:});


%the assignment for position of the features needed for the
%spatio-temporal pooling

stDistance = distmj(positions, vocabPositions);
[~, stAssign] = min(stDistance, [], 2);

%the size of vocabulary for the location of the features
k2 = size(vocabPositions, 1);

%to save the spatio-temporal max-pooling over the features
stFeatPool = cell(1, k2);

%to save the spatio-temporal sum-pooling over the membership information
stMembPool = cell(1, k2);

%compute the ST Encoding (stFeatPool and stMembPool)
for i = 1:k2
    stAssigned = (stAssign==i); 
    nAssigned = sum(stAssigned);
    
    if nAssigned>0
        
        %the spatio-temporal max-pooling of the features
        tFeat = features(stAssigned, :);
        [~, idx] = max(abs(tFeat), [], 1);
        stFeatPool{i} = tFeat(sub2ind(size(tFeat), idx, 1:size(tFeat,2)));
        
         %the spatio-temporal sum-pooling of the membership information
        stMembPool{i} = sum(memb(stAssigned, :), 1);
    else
        stFeatPool{i} = zeros(1, d);
        stMembPool{i} = zeros(1, k1);
    end
end


stFeatPool = cat(2, stFeatPool{:});

%apply Power Normalization over the membership representation
if nargin<5
    alpha = 0.5;
end
stMembPool = PowerNormalization(cat(2,stMembPool{:}), alpha);

%concatenate all information to create the final representation
ST_VLMPF_encoding = cat(2,VLMPF, stFeatPool, stMembPool);
end