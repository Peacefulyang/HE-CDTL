%???????

function W = SVM_train(X,Y,Isw,num_class,alpha,Max_step,reg)
    
    mean_X = mean(mean(X));
    X = X-mean_X;
    num_fea = size(X,2);
    W = normrnd(0,1,[num_class num_fea])*0.001;
    
    for step = 1:Max_step
        [loss,grad] = lossAndGradNaive(X,Y,Isw,num_class,W,reg);
        W = W - alpha*grad;
        %     Acc = Svm_Acc(X,Y,W);
        %    fprintf('step = %d, loss = %f\n',step,Acc)
    end
end

function [loss,dW] = lossAndGradNaive(X,Y,Isw,num_class,W,reg)
    dW=zeros(size(W));
    loss = 0;
    
    num_X= size(X,1);
    for i = 1:num_X
        scores = X(i,:)*W';
        cur_scores=scores(Y(i));
        for j = 1:num_class
            if j ~= Y(i)
                margin=scores(j)-cur_scores+1;
                if margin > 0
                    if ~isempty(Isw)
                        loss = loss+margin*Isw(i);   %?????
                        dW(j,:) = dW(j,:)+ X(i,:)*Isw(i);
                        dW(Y(i),:)= dW(Y(i),:)-X(i,:)*Isw(i);
                    else
                        loss = loss+margin;
                        dW(j,:) = dW(j,:)+ X(i,:);
                        dW(Y(i),:)= dW(Y(i),:)-X(i,:);
                    end
                end
            end
        end
    end
    loss = loss/num_X;
    dW = dW/num_X;
    loss = loss+ reg*sum(sum(W.*W));  %?????
    dW = dW+2*reg*W;
end

function Acc = Svm_Acc(X,Y,W)
    [Y_pre,Y_allpre] = SVM_pre(X,W);
    Acc=mean(Y_pre==Y);
end

