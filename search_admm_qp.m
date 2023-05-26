function [max_acc,max_gmean,max_fscore,max_auc,search_save]=search_admm_qp(data,errorsave,datanum)


    A = [0.1,1,10];
%     B = [0.1,1,10];
    C1 = [0.1,1,10];
    C2 = [0.1,1,10];
    C = [0.1,1,10];
    RHO = [1];%未用到，为了减少循环框架的修改就随便赋一个值
    RBF = [5,100,1000];
    THETA = [0.1,1,10];



    [TAU,ETA] = deal(1.618,0.01);
    SIGMA = [3.8*1e-5];
    TOL = [1e-3];


    [max_acc.value,max_gmean.value,max_fscore.value,max_auc.value] = deal(0);
    [max_acc.std,max_gmean.std,max_fscore.std,max_auc.std] = deal(0);
    [max_acc.id,max_gmean.id,max_fscore.id,max_auc.id] = deal(0);
    search_save = [];
    Type = [0,1,2,3];
    
    
    for i = 1:numel(A)
        para.a = A(i);
        for i = 1:numel(C1)
            para.c1 = C1(i);
            for i = 1:numel(C2)
                para.c2 = C2(i);
                for i = 1:numel(C)
                    para.c3 = C(i);
                    for i = 1:numel(RHO)
                        para.rho = RHO(i);
                        for rbf = RBF
                            
                            for sig = SIGMA
                                for tau = TAU
                                    for tol = TOL
                                        for eta = ETA
                                            for theta = THETA
                                                

                                                para.b=para.a;
                                                [sigtau.sig1,sigtau.sig2,sigtau.sig3,sigtau.sig4] = deal(sig);

                                                [sigtau.tau1,sigtau.tau2] = deal(tau);

                                                [stop.iter1,stop.iter2,stop.tol,stop.tol2] = deal(1000,1000,tol,0.001);%admm最大迭代次数iter1调大
                                                [nag.eta,nag.r] = deal(eta);

                                                parameter = [para.a,para.b,para.c1,para.c2,para.c3,para.rho,rbf];
                                                para4loss = [sig,tau,tol,eta];
                                                dlmwrite(errorsave,[parameter,para4loss],'delimiter',',','-append');
                                                try
                                                    [max_view_acc,max_view_gmean,max_view_fscore,max_view_auc] = Crossvalidation(data,rbf,para,sigtau,stop,nag,theta,datanum);
                                                    search_save_temp = [parameter,para4loss,...
                                                        max_view_acc.value, max_view_acc.id, max_view_acc.std,...
                                                        max_view_gmean.value, max_view_gmean.id, max_view_gmean.std,...
                                                        max_view_fscore.value, max_view_fscore.id, max_view_fscore.std,...
                                                        max_view_auc.value, max_view_auc.id, max_view_auc.std,theta];
                                                    search_save = [search_save;search_save_temp];
                                                catch
                                                    'try_catch errors happen'
                                                    para
                                                    dlmwrite(errorsave,[parameter,para4loss,zeros(1,2)],'delimiter',',','-append');%全为0的哪一行上面为出错参数
                                                    [max_view_acc.value,max_view_gmean.value,max_view_fscore.value,max_view_auc.value] = deal(0);
                                                end
                                                
                                                for search_type = Type
                                                    if search_type == 0%acc
                                                        if max_view_acc.value>max_acc.value
                                                            max_acc.value = max_view_acc.value;
                                                            max_acc.std = max_view_acc.std;
                                                            max_acc.id = max_view_acc.id;
                                                        end
                                                    elseif search_type == 1%gmean
                                                        if max_view_gmean.value>max_gmean.value
                                                            max_gmean.value = max_view_gmean.value;
                                                            max_gmean.std = max_view_gmean.std;
                                                            max_gmean.id = max_view_gmean.id;
                                                        end
                                                    elseif search_type == 2%fscore
                                                        if max_view_fscore.value>max_fscore.value
                                                            max_fscore.value = max_view_fscore.value;
                                                            max_fscore.std = max_view_fscore.std;
                                                            max_fscore.id = max_view_fscore.id;
                                                        end
                                                    elseif search_type == 3%auc
                                                        if max_view_auc.value>max_auc.value
                                                            max_auc.value = max_view_auc.value;
                                                            max_auc.std = max_view_auc.std;
                                                            max_auc.id = max_view_auc.id;
                                                        end
                                                    end
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                            
                        end
                    end
                end
            end
        end
    end     
end