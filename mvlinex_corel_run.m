function [result] = mvlinex_corel_run(datapath,savepath,datanum)
pathsave = savepath;
fsave = strcat(pathsave,'corel_',datanum,' result','.csv');
gridsave = strcat(pathsave,'corel_',datanum,' GridOpt','.csv');
errorsave = strcat(pathsave,'corel_',datanum,' errorsave','.csv');
dlmwrite(errorsave,zeros(1,7),'delimiter',',');


data = load(datapath);
[max_acc,max_gmean,max_fscore,max_auc,search_save]=search_admm_qp(data,errorsave,datanum);
'result'
result = [max_acc.value,max_acc.std,max_acc.id;
          max_gmean.value,max_gmean.std,max_gmean.id;
          max_fscore.value,max_fscore.std,max_fscore.id;
          max_auc.value,max_auc.std,max_auc.id];



metricNames = {'acc';'gmean';'fscore';'auc'};
varNames = {'metric','value','std','id'};
T_result = table(metricNames,result(:,1),result(:,2),result(:,3),'VariableNames',varNames);
writetable(T_result,fsave);

paraNames = {'a','b','c1','c2','c3','rho','rbf',...
    'sig','tau','tol','eta',...
    'acc_value','acc_id','acc_std',...
    'gmean_value','gmean_id','gmean_std',...
    'fscore_value', 'fscore_id','fscore_std',...
    'auc_value','auc_id','auc_std','theta'};
s = search_save;
T_grid = table(s(:,1),s(:,2),s(:,3),s(:,4),s(:,5),s(:,6),s(:,7),...
    s(:,8),s(:,9),s(:,10),s(:,11),...
    s(:,12),s(:,13),s(:,14),...
    s(:,15),s(:,16),s(:,17),...
    s(:,18),s(:,19),s(:,20),...
    s(:,21),s(:,22),s(:,23),s(:,24),'VariableNames',paraNames);
writetable(T_grid,gridsave);
end
