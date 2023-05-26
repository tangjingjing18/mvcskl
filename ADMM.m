function [modela,modelb,Alp_save] = ADMM(X1,X2,Y,rbf_sig,para,sigtau,stop,nag,theta,kfold,datanum)
%% 参数说明：
% a,b为linex损失内的系数；C1为A视角损失权重；C2为B视角损失权重；C3为两视角差的权重； 
% rbf_sig是kernel的参数；sig1-4是admm函数二范数损失的权重；tau是优化u变量时的学习率；
% maxiter1和tol是admm的最大迭代次数与误差容忍；
% maxiter2,tol2是NAG(or GD)的最大迭代次数和误差容忍；eta是NAG的学习率，r是过去冲量影响的权重
%% 构造核矩阵
[n, n1] = size(X1);
[m, m1] = size(X2); 


One = ones(1,n);

Ka = kernel(X1,X1,'rbf',rbf_sig);
Kb = kernel(X2,X2,'rbf',rbf_sig);
% Ka = 1e-50*kernel(X1,X1,'rbf',rbf_sig);%不和规范化的防爆炸处理同时操作,不推荐
% Kb = 1e-50*kernel(X2,X2,'rbf',rbf_sig);

%% 参数赋值
theta1 = theta;

a = para.a;
b=para.b;
C1=para.c1;
C2=para.c2;
C3=para.c3;

sig1 = sigtau.sig1;
sig2 = sigtau.sig2;
sig3 = sigtau.sig3;
sig4 = sigtau.sig4;
tau1 = sigtau.tau1;
tau2 = sigtau.tau2;

maxiter1 = stop.iter1;
maxiter2 = stop.iter2;
tol = stop.tol;
tol2 = stop.tol2;

eta = nag.eta;
r = nag.r;

%% initialize
% Alp=zeros(n,1);
% Bet=zeros(m,1);
% ksiA=zeros(n,1);
% ksiB=zeros(m,1);
% 
% pi1=zeros(n,1);
% pi2=zeros(m,1);
% pi3=zeros(n,1);
% pi4=zeros(m,1);
% 
% u1=zeros(n,1);
% u2=zeros(m,1);
% u3=zeros(n,1);
% u4=zeros(m,1);

Alp=rand(n,1);
Bet=rand(m,1);
ksiA=rand(n,1);
ksiB=rand(m,1);

pi1=rand(n,1);
pi2=rand(m,1);
pi3=rand(n,1);
pi4=rand(m,1);

u1=rand(n,1);
u2=rand(m,1);
u3=rand(n,1);
u4=rand(m,1);


%% 调查文件覆写
% dlmwrite('./results/down1.csv',zeros(1,6),'delimiter',',');
% dlmwrite('./results/down2.csv',zeros(1,6),'delimiter',',');
% dlmwrite('./results/down3.csv',zeros(1,6),'delimiter',',');
% dlmwrite('./results/Alp_up.csv',zeros(1,6),'delimiter',',');
% dlmwrite('./results/Alp_down.csv',zeros(1,6),'delimiter',',');
% 
% dlmwrite('./results/Alp.csv',zeros(1,6),'delimiter',',');
% dlmwrite('./results/Bet.csv',zeros(1,6),'delimiter',',');
% dlmwrite('./results/ksiA.csv',zeros(1,6),'delimiter',',');
% dlmwrite('./results/ksiB.csv',zeros(1,6),'delimiter',',');
% dlmwrite('./results/pi.csv',zeros(1,6),'delimiter',',');
% dlmwrite('./results/u.csv',zeros(1,6),'delimiter',',');

        %% ADMM
        tic;
        for iter = 1 : maxiter1
            
            fprintf('\n***************************** ADMM iter: %d K: %d Dataset：%s ******************************\n', iter',kfold,datanum);
            fprintf('\n******** theta: %4.4e a: %4.4e  b: %4.4e C1: %4.4e C2: %4.4e C3: %4.4e rbf: %4.4e sig: %4.4e ********\n',theta,a,b,C1,C2,C3,rbf_sig,sig1);
            
            Alp_pre = Alp; Bet_pre = Bet; ksiA_pre = ksiA; ksiB_pre = ksiB;
            
            % update Alp
            %%%%%% add 1e-10*eye(size(Alp)) try to fix the badly scale
            Alp=(theta1*Ka+2*C3*(Ka'*Ka)+sig1*(Ka'*Ka))\(2*C3*Ka'*Kb*Bet-Y.*(Ka*u1)+sig1*Y.*(Ka*(One'-ksiA+pi1)));
%             Alp = Alp/norm(Alp);%防爆炸处理，规范化
            %Replace inv(A)*b with A\b, Replace b*inv(A) with b/A
            
            % update Bet
            Bet=(Kb+2*C3*(Kb'*Kb)+sig2*(Kb'*Kb))\(2*C3*(Kb'*Ka)*Alp-Y.*(Kb*u2)+sig2*Y.*(Kb*(One'-ksiB+pi2)));
%             Bet=Bet/norm(Bet);%防爆炸处理，规范化
            %             Bet = ones(m,1);参数固定
            
            % update ksiA
            %             [ksiA,iterA,gradA]=NAG(Ka,Alp,ksiA,Y,a,C1,pi1,pi3,u1,u3,sig1,sig3,maxiter2,r,eta,tol2);
            [ksiA,iterA,gradA]=GD(Ka,Alp,ksiA,Y,C1,pi1,pi3,u1,u3,a,sig1,sig3,maxiter2,eta,tol2);
            NAG_A.iter = iterA;
            NAG_A.grad = gradA;
            %             ksiA = 0.1*ones(n,1);%参数固定
            
            %update ksiB
            %[ksiB,iterB,gradB]=NAG(Kb,Bet,ksiB,Y,b,C2,pi2,pi4,u2,u4,sig2,sig4,maxiter2,r,eta,tol2);
            [ksiB,iterB,gradB]=GD(Kb,Bet,ksiB,Y,C2,pi2,pi4,u2,u4,b,sig2,sig4,maxiter2,eta,tol2);
            NAG_B.iter = iterB;
            NAG_B.grad = gradB;
            %     iter2
            %     grad
            
            % update pi1
            pi1=pos(u1/sig1+(Y.*(Ka*Alp)-1+ksiA));%pos()定义在最后
            %             pi1 = 0.5*ones(n,1);%参数固定
            % update pi2
            pi2=pos(u2/sig2+(Y.*(Kb*Bet)-1+ksiB));
            % update pi3
            pi3=pos(u3/sig3+ksiA);
            % update pi4
            pi4=pos(u4/sig4+ksiB);
            
            % updating multipliers
            % update u1 (multiplier)
            u1=u1+tau1*sig1*(Y.*(Ka*Alp)+ksiA-1-pi1);
            % update u2 (multiplier)
            u2=u2+tau2*sig2*(Y.*(Kb*Bet)+ksiB-1-pi2);
            % update u3 (multiplier)
            u3=u3+tau1*sig3*(ksiA-pi3);
            % update u4 (multiplier)
            u4=u4+tau2*sig4*(ksiB-pi4);
            
            
            %% 向量输出检查
            %             Alp_down = 2*C3*Ka'*Kb*Bet-Y.*(Ka*u1)+sig1*Y.*(Ka*(One'-ksiA+pi1));
            %             Alp_up = inv(theta1*Ka+2*C3*(Ka'*Ka)+sig1*Ka'*Ka);
            %             down1 = Ka'*Kb*Bet;
            %             down2 = Y.*(Ka*u1);
            %             down3 = sig1*Y.*(Ka*(One'-ksiA+pi1));
            %
            %             dlmwrite('./results/down1.csv',[iter,down1'],'delimiter',',','-append');
            %             dlmwrite('./results/down2.csv',[iter,down2'],'delimiter',',','-append');
            %             dlmwrite('./results/down3.csv',[iter,down3'],'delimiter',',','-append');
            %             dlmwrite('./results/Alp_up.csv',Alp_up,'delimiter',',');
            % %             dlmwrite('./results/Alp_up.csv',[iter,Alp_up'],'delimiter',',','-append');
            %             dlmwrite('./results/Alp_down.csv',[iter,Alp_down'],'delimiter',',','-append');
            %
            %             dlmwrite('./results/Alp.csv',[iter,Alp'],'delimiter',',','-append');
            %             dlmwrite('./results/Bet.csv',[iter,Bet'],'delimiter',',','-append');
            %             dlmwrite('./results/ksiA.csv',[iter,ksiA'],'delimiter',',','-append');
            %             dlmwrite('./results/ksiB.csv',[iter,ksiB'],'delimiter',',','-append');
            %             dlmwrite('./results/pi.csv',[01,iter,pi1';02,iter,pi2';03,iter,pi3';04,iter,pi4'],'delimiter',',','-append');
            %             dlmwrite('./results/u.csv',[01,iter,u1';02,iter,u2';03,iter,u3';04,iter,u4'],'delimiter',',','-append');
            
            %% 变量打印区域
            cal = [norm(Alp),norm(Bet),norm(ksiA),norm(ksiB);...
                norm(pi1),norm(pi2),norm(pi3),norm(pi4);...
                norm(u1),norm(u2),norm(u3),norm(u4)];
            variablename = ['Alp: %4.4e \n','Bet: %4.4e \n','ksiA: %4.4e \n','ksiB: %4.4e \n'];
            fprintf(variablename, cal(1,:));
            piname = ['pi1: %4.4e \n','pi2: %4.4e \n','pi3: %4.4e \n','pi4: %4.4e \n'];
            fprintf(piname,cal(2,:));
            uname = ['u1: %4.4e \n','u2: %4.4e \n','u3: %4.4e \n','u4: %4.4e \n'];
            fprintf(uname,cal(3,:));
            
            %%% calculate objective value
            fval(iter) = 1/2*theta1*Alp'*Ka*Alp+1/2*Bet'*Kb*Bet...
                +C1*One*(exp(a*Y.*ksiA)-a*Y.*ksiA-1)+C2*One*(exp(b*Y.*ksiB)-b*Y.*ksiB-1)...
                +C3*One*((Ka*Alp-Kb*Bet).*(Ka*Alp-Kb*Bet));
            res_pri_vector = [(Y.*(Ka*Alp)+ksiA-1-pi1);(Y.*(Kb*Bet)+ksiB-1-pi2);(ksiA-pi3);(ksiB-pi4)];
            res_pri(iter) = norm(res_pri_vector);

            fprintf('fval: %4.4e \n',fval(iter));
            fprintf('res_pri: %4.4e \n',res_pri(iter));
            
            %k=k+1;
            %end
            
            %% stopCond
            if isnan(norm(Alp))==1 || isnan(norm(Bet))==1
                disp(' !!!ADMM is Exploding!!! ');  break;
            end
            stopCond1 = norm(Alp - Alp_pre)/norm(Alp_pre);
            stopCond2 = norm(Bet - Bet_pre)/norm(Bet_pre);
            stopCond3 = norm(ksiA- ksiA_pre)/norm(ksiA_pre);
            stopCond4 = norm(ksiB- ksiB_pre)/norm(ksiB_pre);
            stopCond = max([stopCond1 stopCond2 stopCond3 stopCond4 ]);
            %stopCond = max([stopCond1 stopCond2]);
            %     stopPath(iter) = stopCond;% 没有使用该变量
            fprintf('stopCound1: %4.4e \n', stopCond1);
            fprintf('stopCound2: %4.4e \n', stopCond2);
            fprintf('stopCound3: %4.4e \n', stopCond3);
            fprintf('stopCound4: %4.4e \n', stopCond4);
            fprintf('ADMM stopFormulaVal: %4.4e \n', stopCond);
            if (iter> 10) &&  (stopCond < tol )
                disp(' !!!stopped by ADMM termination rule!!! ');  break;
            end
        end
        toc;

    modela.w = Alp;
    modela.x = X1;
    modela.theta = theta1;
    
    
    modelb.w = Bet;
    modelb.x = X2;
%     modelb.theta = theta2;
end