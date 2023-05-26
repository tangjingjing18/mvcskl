function [acc,gmean,fscore,auc_arr]=predict_svm2(modela, modelb,test_x1,test_x2,test_target,rbf,type)
%不同type下输出的decision function
    if type == 0 %f_A(x)=Ka*alpha
        label=kernel(test_x1,modela.x,'rbf',rbf)*modela.w;
    elseif type == 1 %f_B(x)
        label=kernel(test_x2,modelb.x,'rbf',rbf)*modelb.w;
    elseif type == 2 %theta1*f_A(x) + f_B(x)
        label=kernel(test_x1,modela.x,'rbf',rbf)*modela.w*modela.theta + kernel(test_x2,modelb.x,'rbf',rbf)*modelb.w;
    else %0.5*f_A(x) + 0.5*f_B(x)
        label=kernel(test_x1,modela.x,'rbf',rbf)*modela.w*0.5 + kernel(test_x2,modelb.x,'rbf',rbf)*modelb.w*0.5 ;
    end
    
    %%% AUC
% 	[val,ind] = sort(label,'descend');%val储存降序后的数值，ind储存数值之前的位置序号
% 	roc_y = test_target(ind);
% 	if sum(roc_y == 1) ~= 0
%         stack_x = cumsum(roc_y == -1)/sum(roc_y == -1);
%         stack_y = cumsum(roc_y == 1)/sum(roc_y == 1);
%         [stack_x,stack_y,thre,auc]=perfcurve(test_target,label,1);
%     end
%     auc_arr = auc;
    auc_arr=AUC(test_target,label);
    
    
    %%% acc,gmean,fscore
    [n,~] = size(label);


    P = 0;
    N = 0;
    TP = 0;
    FN = 0;
    TN = 0;
    FP = 0;
    for i=1:n
        if test_target(i) == 1
            P = P + 1; 
            if label(i)>=0
                TP = TP+1;
            else
                FN = FN+1;
            end
        else
            N = N + 1;
            if label(i)>=0
                FP = FP+1;
            else
                TN = TN+1;
            end
        end
    end
    acc = (TP+TN)/(TP+TN+FP+FN);
    pre = TP/(TP+FP);
    recall = TP/(TP+FN);
    spec = TN/(TN+FP);
    sen = TP/(TP+FN);
    gmean = (recall*spec)^0.5;
    fscore = 2*pre*recall/(pre+recall);
end