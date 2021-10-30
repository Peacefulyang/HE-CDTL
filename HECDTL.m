function [NTear_Acc,ht] = HECDTL(Cur_targetdata,Cur_test,Src_data,ht,Options)
    
    hs = [];
    Num_class = Options.Num_class;
    Max_update = Options.Corp_update;
    alpha = Options.SVM_alpha;
    Max_step = Options.SVM_step;
    reg = Options.SVM_reg;
    
    Xt = Cur_targetdata(:,2:end);
    Yt = Cur_targetdata(:,1);
    Src_origX = [];
    for i = 1:length(Src_data)
        Src_origX{i} = Src_data{i}(:,2:end);
    end
    
%     Src_data = CORAL_update(Src_origX,Xt,Src_data,[]);
    for i = 1:length(Src_data)
        hs{i} = SVM_train(Src_data{i}(:,2:end), Src_data{i}(:,1), [], Num_class, alpha, Max_step, reg);
    end
    
    Tar_m = length(ht)+1;
    ht{Tar_m} = SVM_train(Xt, Yt, [], Num_class, alpha, Max_step, reg);
    Tar_all_pred = predict_base(Xt,ht);
    [Cwt,Mwt] = Init_w(Tar_all_pred, Yt, [],Num_class);
    
    Src_m = length(Src_data);
    Src_all_pred = predict_base(Xt,hs);
    [~,Mws] = Init_w(Src_all_pred, Yt, [],Num_class);
    
    Ytest = Cur_test(:,1);
    
    Wtt = [];
    for i = 1:Tar_m
        Wtt = [Wtt;Cwt{i}];
    end
    NWtt = Wtt./repmat(sum(Wtt,1),Tar_m,1);
    Tar_all_pred = predict_base(Xt,ht);
    Post_prec = zeros(length(Yt),Num_class);
    for i = 1:Tar_m
        kk = repmat(NWtt(i,:),length(Yt),1).*Tar_all_pred{i}.soft;
        Post_prec = Post_prec+kk;
    end
    [F_max,Tar_pred]=max(Post_prec,[],2);
    Awt = (sum(Tar_pred==Yt)/length(Yt));
    
    Secind = find(Awt>Mws);
    [Hs,update_num,Mws] = Corp_update(hs,ht,Cwt,Mws,Mwt,Src_data,Num_class,Max_update,alpha,Max_step,reg,Secind,Src_origX,Xt,Yt);
    YWt = sum(mean(Wtt,2));%
    
    Tar_all_pred = predict_base(Cur_test(:,2:end),ht);
    NPost_prec = zeros(length(Ytest),Num_class);
    for i = 1:Tar_m
        NPost_prec = NPost_prec+repmat(NWtt(i,:),length(Ytest),1).*Tar_all_pred{i}.soft;
    end
    NPost_prec = YWt*NPost_prec;
    
    Post_precs = zeros(length(Ytest),Num_class);
    for i = 1:Src_m
        prec = zeros(length(Ytest),Num_class);
        for j = 1:update_num
            chs{1} = Hs{j,i};
            Src_all_pred = predict_base(Cur_test(:,2:end),chs);
            prec = Src_all_pred{1}.soft;
        end
        Post_precs = Post_precs+Mws(i)*prec;
    end
    [F_max,NTar_pred]=max(NPost_prec+Post_precs,[],2);
    NTear_Acc = sum(NTar_pred==Ytest)/length(Ytest);
    if length(Mwt)>15
        [Minwtt,~] = min(NWtt(1:Tar_m,:),[],2);
        [~,remove_idx] = min(Minwtt);
        ht(remove_idx) = [];
    end
end

function [CW,W] = Init_w(all_pred, label,Isw,Num_class)
    m = length(all_pred);
    Merr = [];
    CW = [];
    for i=1:m
        err2 = zeros(1,Num_class);
        pre = all_pred{i}.label;
        if ~isempty(Isw)
            iswi = Isw;
        else
            iswi = ones(length(pre),1);
        end
        for j = 1:length(label)
            err2(label(j)) = err2(label(j))+iswi(j)'*(pre(j)~=label(j));
        end
        for j = 1: Num_class
            err2(j) = err2(j)/length(find(label==j));
        end
        Merr(i) = (sum(iswi'*(pre~=label))/length(label));
        W(i) = 1-Merr(i);
        z = (1-err2);
        CW{i}= z;
    end
end

function pred = predict_base(data,model)  %??????concept????????
    m = length(model);
    pred = [];
    for i=1:m
        clf = model{i};
        [label,Posterior] = SVM_pre(data,clf);
        pred{i}.soft = Posterior;
        pred{i}.label = label;
    end
end

function [Isw,BetaS] = Iw_update(Src_data,hs,BetaS,Tar_m,YWt,Ws,Mwt,ht,Num_class,Wss,Secind,beta)
    Isw = Wss;
    
    for i = 1:length(Secind)
        sec_index = Secind(i);
        weight = Wss{sec_index};
        srcdata = Src_data{sec_index};
        X = srcdata(:,2:end);
        Y = srcdata(:,1);
        truelable = zeros(length(Y),Num_class);
        for j = 1:length(Y)
            truelable(j,Y(j)) = 1;
        end
        Tar_all_pred = predict_base(X,ht);
        Post_prec = zeros(length(Y),Num_class);
        Wtt = [];
        for j = 1:Tar_m
            Wtt = [Wtt;YWt{j}];
        end
        Wtt = Wtt./repmat(sum(Wtt,1),Tar_m,1);
        for j = 1:Tar_m
            Post_prec = Post_prec+Wtt(j,:).*Tar_all_pred{j}.soft;
        end
        Post_prec = sum(Mwt)*Post_prec;
        
        Src_all_pred = predict_base(X,hs);
        yPost_precs = zeros(length(Y),Num_class);
        for j = 1:length(hs)
            yPost_precs = yPost_precs+Ws(j)*Src_all_pred{j}.soft;
            Post_prec = Post_prec+yPost_precs;
            [~,pre_y]=max(Post_prec,[],2);
        end
        margin = (pre_y~=Y);
        Isw{sec_index} = weight.*exp(-beta(i)*margin);
    end
end

function [Hs,update_num,Ws] = Corp_update(hs,ht,Wt,Ws,Mwt,Src_data,Num_class,Max_update,alpha,Max_step,reg,Secind,Src_origX,Xt,Yt)
    
    Tar_m = length(ht);
    Hs = [];
    hhs = [];
    BetaS = ones(1,length(hs));
    Wss = [];
    update_num = 1;
    n = [];
    
    for i = 1:length(hs)
        n = [n;size(Src_data{i},1)];
        for j = 1:Max_update
            Hs{j,i} = hs{i};
            hhs{i} = hs{i};
        end
        Wss{i} = ones(size(Src_data{i},1),1);
    end
    n = n+length(Yt);
    beta  = 1/(1+sqrt(2*log(n/Max_update)));
    if ~isempty(Secind)
        update_num = Max_update;
        for Updat_k = 2:Max_update
            
            [Wss,BetaS] = Iw_update(Src_data,hhs,BetaS,Tar_m,Wt,Ws,Mwt,ht,Num_class,Wss,Secind,beta);
            
            for i = 1:length(Secind)
                hhs{i} = SVM_train(Src_data{Secind(i)}(:,2:end), Src_data{Secind(i)}(:,1),Wss{i},Num_class,alpha,Max_step,reg);
                Hs{Updat_k,Secind(i)} = hhs{i};
            end
            Src_all_pred = predict_base(Xt,hs);
            [~,Ws] = Init_w(Src_all_pred, Yt,[],Num_class);
            Src_data = CORAL_update(Src_origX,Xt,Src_data,Wss);
        end
    end
end

