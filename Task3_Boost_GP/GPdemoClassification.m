disp('See http://www.gaussianprocess.org/gpml/code/matlab/doc/ for details.')

%% start , generate training data
disp(' '); disp('clear all, close all')
clear all, close all
write_fig = 0;
disp(' ')

n1 = 80; n2 = 40;   % number of data points from each class
S1 = eye(2); S2 = [1 0.95; 0.95 1]; % the two covariance matrices
m1 = [0.75; 0]; m2 = [-0.75; 0]; % the two means

% generate training data from mean and std
x1 = bsxfun(@plus, chol(S1)'*gpml_randn(0.2, 2, n1), m1);       
x2 = bsxfun(@plus, chol(S2)'*gpml_randn(0.3, 2, n2), m2);         

% plot training data
x = [x1 x2]'; y = [-ones(1,n1) ones(1,n2)]';
figure(6)
plot(x1(1,:), x1(2,:), 'b+', 'MarkerSize', 12); hold on
plot(x2(1,:), x2(2,:), 'r+', 'MarkerSize', 12);

% plot lines ión figure to show in what rules was data denerated
[t1 t2] = meshgrid(-4:0.1:4,-4:0.1:4);
t = [t1(:) t2(:)]; n = length(t);               % these are the test inputs
tmm = bsxfun(@minus, t, m1');
p1 = n1*exp(-sum(tmm*inv(S1).*tmm/2,2))/sqrt(det(S1));
tmm = bsxfun(@minus, t, m2');
p2 = n2*exp(-sum(tmm*inv(S2).*tmm/2,2))/sqrt(det(S2));
set(gca, 'FontSize', 24)
contour(t1, t2, reshape(p2./(p1+p2), size(t1)), [0.1:0.1:0.9])
[c h] = contour(t1, t2, reshape(p2./(p1+p2), size(t1)), [0.5 0.5]);
set(h, 'LineWidth', 2)
colorbar
grid
axis([-4 4 -4 4])


%% fit model to generated data and draw splitting lines for classes

% parameters of GP to tune
hyp.mean = 0;
covfunc = @covSEard;   hyp.cov = log([1 1 1]); % @covSEard;
likfunc = @likErf; % @likErf
inffunc = @infEP; % @infEP
meanfunc = @meanConst;

% traininga
hyp = minimize(hyp, @gp, -40, inffunc, meanfunc, covfunc, likfunc, x, y);
[a b c d lp] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, t, ones(n,1));

% plot training data
figure(7)
set(gca, 'FontSize', 24)
plot(x1(1,:), x1(2,:), 'b+', 'MarkerSize', 12); hold on
plot(x2(1,:), x2(2,:), 'r+', 'MarkerSize', 12)
% plot testing data - in this case it is whole 2D space - so plot as lines
% to see how classes are splitted - same as lines before
contour(t1, t2, reshape(exp(lp), size(t1)), 0.1:0.1:0.9)
[c h] = contour(t1, t2, reshape(exp(lp), size(t1)), [0.5 0.5]);
set(h, 'LineWidth', 2)
colorbar
grid
axis([-4 4 -4 4])

%% now generate tesing data in similar way as training data and check correctness
ntt1 = 170; ntt2 = 170;
tt1 = bsxfun(@plus, chol(S1)'*gpml_randn(0.2, 2, ntt1), m1);       
tt2 = bsxfun(@plus, chol(S2)'*gpml_randn(0.3, 2, ntt2), m2);  
tt = [tt1 tt2]'; ytt = [-ones(1,ntt1) ones(1,ntt2)]';

% do prediction
[a b c d lpt] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, tt, ones(n,1));

% plot results
figure(8)
set(gca, 'FontSize', 24)
% plot(x1(1,:), x1(2,:), 'b+', 'MarkerSize', 10); hold on
% plot(x2(1,:), x2(2,:), 'r+', 'MarkerSize', 10)
plot(tt1(1,:), tt1(2,:), 'bo', 'MarkerSize', 10); hold on
plot(tt2(1,:), tt2(2,:), 'ro', 'MarkerSize', 10)
contour(t1, t2, reshape(exp(lp), size(t1)), 0.1:0.1:0.9)
[c h] = contour(t1, t2, reshape(exp(lp), size(t1)), [0.5 0.5]);
set(h, 'LineWidth', 2)
colorbar
grid
axis([-4 4 -4 4])

% check how many are classified correctly. 
% a - result of prediction. Closer to (-1 or 1) - the more we sure in result. Class 1 == a< 0, class 2 == a> 0
classificationRate = 1 - sum(abs((a > 0) - (ytt > 0))) / size(ytt, 1)

%% evaluate classifier  with respect to its uncertainity estimation
correctClassification = ((a > 0) == ytt ) + (-(a < 0) == ytt );

% dind data points classified corect and incorrect and plot it on
% uncertainty histogram
indCorrect = find(correctClassification > 0);
indInCorrect = find(correctClassification == 0);
% lpt == log(p)
probab = exp(lpt);
plotUncert(probab, indCorrect, indInCorrect);

%% and now compare with boosting
% use same data for training and testing
templTree = templateTree('MaxNumSplits',15,'Surrogate','on');
ClassTreeEns = fitensemble(x, y, 'Bag', 1000, templTree, 'Type', 'classification');
[labelsPredict score] = predict(ClassTreeEns, tt);
classificationRateBoost = sum((labelsPredict == ytt)) / size(ytt, 1)

% plot uncertainity
class1 = labelsPredict == -1;
class2 = labelsPredict == 1;
% score is from [0, 1]. higher is better. So same as our probability of
% classes. First col - class 1, second - class 2
% score = (P(-1), P(1))
probab1 = class1 .*  score(:, 1);
probab2 = class2 .*  score(:, 2);
probab = probab1 + probab2;
indCorrect = labelsPredict == ytt;
indInCorrect = labelsPredict ~= ytt;
plotUncert(probab, indCorrect, indInCorrect);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% do same but on large scale
disp('large scale classification using the FITC approximation')
[u1,u2] = meshgrid(linspace(-2,2,5)); u = [u1(:),u2(:)]; clear u1; clear u2
nu = size(u,1);
covfuncF = {@covFITC, {covfunc}, u};
inffunc = @infFITC_EP;                     % one could also use @infFITC_Laplace
hyp = minimize(hyp, @gp, -40, inffunc, meanfunc, covfuncF, likfunc, x, y);
[a b c d lp] = gp(hyp, inffunc, meanfunc, covfuncF, likfunc, x, y, t, ones(n,1));
disp(' ')
figure(8)
set(gca, 'FontSize', 24)
plot(x1(1,:), x1(2,:), 'b+', 'MarkerSize', 12); hold on
plot(x2(1,:), x2(2,:), 'r+', 'MarkerSize', 12)
contour(t1, t2, reshape(exp(lp), size(t1)), 0.1:0.1:0.9)
[c h] = contour(t1, t2, reshape(exp(lp), size(t1)), [0.5 0.5]);
set(h, 'LineWidth', 2)
plot(u(:,1),u(:,2),'ko', 'MarkerSize', 12)
colorbar
grid
axis([-4 4 -4 4])
if write_fig, print -depsc f8.eps; end