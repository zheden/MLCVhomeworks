function [ ] = plotUncert( prob, indCorrect, indInCorrect)
    probCorrect = prob(indCorrect);
    probInCorrect = prob(indInCorrect);

    hCorrect = -(probCorrect .* log(probCorrect) + (1-probCorrect).*log(1-probCorrect));
    hInCorrect = -(probInCorrect .* log(probInCorrect) + (1-probInCorrect).*log(1-probInCorrect));
    figure;
    subplot(1, 2, 1);
    histogram(hCorrect,10);
    xlabel('uncertainity of correct classification');
    ylabel('num');

    subplot(1, 2, 2);
    histogram(hInCorrect,10);
    xlabel('uncertainity of incorrect classification');
    ylabel('num');


end

