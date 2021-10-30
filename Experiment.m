function Experiment()
    
    % source_domains = {'amazon.mat','dslr.mat'};
    % target_domain = 'webcam.mat';
    % Experiment: the main function used to run HomOTL-ODDM.
    %
    %--------------------------------------------------------------------------
    % Input:
    %      source_domains: a set of source dataset_name, e.g. {'PIE1.mat','PIE2.mat','PIE3.mat','PIE4.mat'}
    %      target_domain: target dataset_name, e.g. 'PIE5.mat'
    %--------------------------------------------------------------------------
    
    %% run experiments:
    Options.theta = 0.001;
    Options.SVM_alpha = 0.001;
    Options.SVM_step = 100;
    Options.SVM_reg = 0.5;
    Options.Corp_update = 5;
    Options.Num_class = 4;
    
    result = [];
    for run = 1:20
        
        filename = sprintf('ROTdata%d.mat', run);
        load(sprintf('data/%s', filename));
        Src_data = ROTdata.Src_data;
        Tar_train = ROTdata.Tar_train;
        Tar_test = ROTdata.Tar_test;
        chunk_Max = length(Tar_train);
        
        ht = [];
        for chunk_num = 1:chunk_Max
            Cur_targetdata = Tar_train{chunk_num};
            Cur_test = Tar_test{chunk_num};
            [Tar_AccCorp,ht] = HECDTL(Cur_targetdata,Cur_test,Src_data,ht,Options);
            result(run,chunk_num) = Tar_AccCorp;
        end
        filename = sprintf('result%d', run);
        save (filename, 'result');
    end  
end

