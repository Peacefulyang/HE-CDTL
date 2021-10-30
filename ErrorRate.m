function  ER = ErrorRate(LabelR, LabelH,Weight)

% ER 样本分类的错误率
% weight 样本权重
% LabelR 样本真实标签
% LabelH 样本预测标签
if(nargin <3)
    Weight = ones(size(LabelR,1),1);
end
err = LabelH==LabelR;
ER = sum((Weight.*err)/sum(Weight));

    if(ER>0.5)
        ER = 0.5;
    end
    if(ER==0)
        ER=0.001;
    end
end