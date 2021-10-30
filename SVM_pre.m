function [Y_pre,Y_allpre] = SVM_pre(X,W)
mean_X = mean(mean(X));
X = X-mean_X;
Y_allpre = X*W';
[~,Y_pre] = max(Y_allpre,[],2);
end