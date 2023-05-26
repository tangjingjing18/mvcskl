function [theta1,theta2]=QP(X1,X2,Alp,Bet,rho,rbf_sig)
%rho是theta^2的系数
    [n,n1] = size(X1);
    Ka = kernel(X1,X1,'rbf',rbf_sig);
    Kb = kernel(X2,X2,'rbf',rbf_sig);

    options = optimset;    % Options是用来控制算法的选项参数的向量，创建 options 结构体，其中所有字段设置为 []
    options.LargeScale = 'off'; %当设为'on'时使用大型算法，若设为'off'则使用中型问题的算法
    options.Display = 'off';    %不显示输出
    
    H = rho*eye(2,2);
    f = [Alp'*Ka*Alp, Bet'*Kb*Bet];
%   f = cat(1,zeros(plen,1),-ones(nlen,1));
    A = [];
    b = [];
    Aeq = ones(1,2); 
    beq = 1;
    lb = zeros(2,1); %下界
    ub = []; %上界
    a0 = zeros(n,1);  % a0是解的初始近似值；对于 'interior-point-convex' 算法quadprog 将忽略a0。
     H=H+1e-10*eye(size(H)); %eye()是单位阵；1e-10是1*10^(-10)，是number1不是L的小写;此处是一个形同H的单位阵乘上了10^(-10)
     options.Algorithm='interior-point-convex';

       %[a,fval,eXitflag,output,lambda]  = quadprog(H,f,A,b,Aeq,beq,lb,ub,a0,options);
     % 标准形式 x = quadprog(H,f,A,b,Aeq,beq,lb,ub,x0,options);
    [theta,fval]  = quadprog(H,f,A,b,Aeq,beq,lb,ub,a0,options);
    fprintf('QP fval: %4.4e \n',fval);
    fprintf('QP theta: %4.4e \n',theta);
    
    theta1=theta(1);
    theta2=theta(2);
%     epsilon = 1e-10; 
%     sv_label = find(abs(a)>epsilon);  %0<a<a(max)则认为x为支持向量
%     a = a(sv_label);
%     Xsv = x(sv_label,:);
%     Ysv = y(sv_label);
%     svnum = length(sv_label);
%     model.Xsv = Xsv;
%     model.Ysv = Ysv;
%     model.kerType=kerType;
%     model.gamma=gamma;
%     model.svnum = svnum;
%     num = length(Ysv);
     
end