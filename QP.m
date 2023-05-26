function [theta1,theta2]=QP(X1,X2,Alp,Bet,rho,rbf_sig)
%rho��theta^2��ϵ��
    [n,n1] = size(X1);
    Ka = kernel(X1,X1,'rbf',rbf_sig);
    Kb = kernel(X2,X2,'rbf',rbf_sig);

    options = optimset;    % Options�����������㷨��ѡ����������������� options �ṹ�壬���������ֶ�����Ϊ []
    options.LargeScale = 'off'; %����Ϊ'on'ʱʹ�ô����㷨������Ϊ'off'��ʹ������������㷨
    options.Display = 'off';    %����ʾ���
    
    H = rho*eye(2,2);
    f = [Alp'*Ka*Alp, Bet'*Kb*Bet];
%   f = cat(1,zeros(plen,1),-ones(nlen,1));
    A = [];
    b = [];
    Aeq = ones(1,2); 
    beq = 1;
    lb = zeros(2,1); %�½�
    ub = []; %�Ͻ�
    a0 = zeros(n,1);  % a0�ǽ�ĳ�ʼ����ֵ������ 'interior-point-convex' �㷨quadprog ������a0��
     H=H+1e-10*eye(size(H)); %eye()�ǵ�λ��1e-10��1*10^(-10)����number1����L��Сд;�˴���һ����ͬH�ĵ�λ�������10^(-10)
     options.Algorithm='interior-point-convex';

       %[a,fval,eXitflag,output,lambda]  = quadprog(H,f,A,b,Aeq,beq,lb,ub,a0,options);
     % ��׼��ʽ x = quadprog(H,f,A,b,Aeq,beq,lb,ub,x0,options);
    [theta,fval]  = quadprog(H,f,A,b,Aeq,beq,lb,ub,a0,options);
    fprintf('QP fval: %4.4e \n',fval);
    fprintf('QP theta: %4.4e \n',theta);
    
    theta1=theta(1);
    theta2=theta(2);
%     epsilon = 1e-10; 
%     sv_label = find(abs(a)>epsilon);  %0<a<a(max)����ΪxΪ֧������
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