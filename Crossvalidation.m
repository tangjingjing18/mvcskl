function [max_view_acc,max_view_gmean,max_view_fscore,max_view_auc] = Crossvalidation(data,rbf_sig,para,sigtau,stop,nag,theta,datanum)

    x1 = data.x;
    x2 = data.x2;
    y = data.y;
    x1t = mapminmax(x1',0,1);
    x1 = x1t';
    x2t = mapminmax(x2',0,1);
    x2 = x2t';
    
  
    [n,~] = size(x1);
    x1 = [x1,ones(n,1)];
    
    [n,~] = size(x2);
    x2 = [x2, ones(n,1)];
    


    [M,N]=size(y);
    indices=crossvalind('Kfold',y(1:M,N),3);
    for k=1:3
        test = (indices == k); 
        train = ~test;
        train_x1=x1(train,:);
        train_x2=x2(train,:);
        train_target=y(train,:);
        test_x1=x1(test,:);
        test_x2=x2(test,:);
        test_target=y(test,:);


        [modela,modelb] = ADMM(train_x1,train_x2,train_target,rbf_sig,para,sigtau,stop,nag,theta,k,datanum);

        [acc0,gmean0,fscore0,auc_arr0] = predict_svm2(modela, modelb,test_x1,test_x2,test_target,rbf_sig,0);%f_A
        [acc1,gmean1,fscore1,auc_arr1] = predict_svm2(modela, modelb,test_x1,test_x2,test_target,rbf_sig,1);%f_B
        [acc2,gmean2,fscore2,auc_arr2] = predict_svm2(modela, modelb,test_x1,test_x2,test_target,rbf_sig,2);%f_weight
        [acc3,gmean3,fscore3,auc_arr3] = predict_svm2(modela, modelb,test_x1,test_x2,test_target,rbf_sig,3);%f_half
        
        acc_A(k) = [acc0];
        acc_B(k) = [acc1];
        acc_W(k) = [acc2];
        acc_H(k) = [acc3];
        
        gmean_A(k) = [gmean0];
        gmean_B(k) = [gmean1];
        gmean_W(k) = [gmean2];
        gmean_H(k) = [gmean3];
        
        fscore_A(k) = [fscore0];
        fscore_B(k) = [fscore1];
        fscore_W(k) = [fscore2];
        fscore_H(k) = [fscore3];
        
        auc_A(k) = [auc_arr0];
        auc_B(k) = [auc_arr1];
        auc_W(k) = [auc_arr2];
        auc_H(k) = [auc_arr3];
        
        
    end

    max_view_acc = maxview(acc_A,acc_B,acc_W,acc_H);
    max_view_gmean = maxview(gmean_A,gmean_B,gmean_W,gmean_H);
    max_view_fscore = maxview(fscore_A,fscore_B,fscore_W,fscore_H);
    max_view_auc = maxview(auc_A,auc_B,auc_W,auc_H);
end
