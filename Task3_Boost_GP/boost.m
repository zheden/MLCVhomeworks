
% read dataset
images = (loadMNISTImages('train-images.idx3-ubyte'))';
labels = loadMNISTLabels('train-labels.idx1-ubyte');

imagesTest = (loadMNISTImages('t10k-images.idx3-ubyte'))';
labelsTest = loadMNISTLabels('t10k-labels.idx1-ubyte');

%I = reshape(images(1, :), 28, 28 );

numToTest = 1000;

%%
clc;
% learning rate
lr = 0.1; %0.1 0.25 0.5
% number of training cycles == number of weak classifiers (trees in this case)
numTrees = 1000; % 250 500
rng(1); % For reproducibility
% template for weak classifier
t = templateTree('MaxNumSplits',10,'Surrogate','on'); % 1, 5
BoostClassifier = fitensemble(images(1:numToTrain, :), labels(1:numToTrain),'Bag',numTrees,t,...
    'Type','classification');

% final training error (all weak classifiers)
trainingError = resubLoss(BoostClassifier)

% prediction
[labelsPredict score] = predict(BoostClassifier, imagesTest(1:numToTest, :));
classificationRate = sum((labelsPredict == labelsTest(1:numToTest))) / numToTest

% plot training error
close all; figure;
rsLoss = resubLoss(BoostClassifier,'Mode','Cumulative');
plot(rsLoss);
xlabel('Number of Learning Cycles');
ylabel('Resubstitution Loss');
