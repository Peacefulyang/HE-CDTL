function Src_data = CORAL_update(Src_origX,Cur_Xt,Src_data,W)
    for i = 1:length(Src_data)
        L = size(Cur_Xt,2);
        Xs = Src_origX{i};
        Xs = double(Xs);
        Cur_Xt = double(Cur_Xt);
        if ~isempty(W)
            CW = W{i};
            cov_source = cov(repmat(CW,1,L).*Xs) + eye(size(repmat(CW,1,L).*Xs, 2));
        else
            cov_source = cov(Xs) + eye(size(Xs, 2));
        end
        cov_target = cov(Cur_Xt) + eye(size(Cur_Xt, 2));
        A_coral = cov_source^(-1/2)*cov_target^(1/2);
        Ys = Src_data{i}(:,1);
        XXS = Xs*A_coral;
        Src_data{i} = [Ys,XXS];
    end
end