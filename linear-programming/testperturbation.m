% 测试LP问题的crossover算法
% 2020/8/12
    % LP问题：求 min c^T x, subject to: A_mat x = b, 0 <= x <= u,
    % 其中, x,c,u\in\R^(m*1), b\in\R^(n*1), A_mat\in\R^(n*m)

clear

benchmark_lp = ["datt256_lp","brazil3","buildingenergy","chromaticindex1024-7",...
                "cont1","cont11","dbic1","degme","ds-big","ex10","fome13",...
                "irish-electricity","karted","L1_sixm250obs","L1_sixm1000obs","Linf_520c",...
                "neos","neos1","neos2","neos3","neos-5052403-cygnet","ns1644855",...
                "ns1687037","ns1688926","nug08-3rd","nug15","pds-40","pds-100",...
                "physiciansched3-3","qap15","rail02","rail4284","s100","s250r10",...
                "savsched1","self","shs1023","square41","stat96v1","stat96v4",...
                "stormG2_1000","stp3d","supportcase10","tp-6","ts-palko","watson_2",...
                "set-cover-model"];
problem =  benchmark_lp + '.mps'; 
% files = '../' + benchmark_lp + '_' + num2str(log10(eps)) + '.mat';

% MyCost = zeros(length(problem),1);
% MyOF = cell(length(problem),1);
% Mytime = zeros(length(problem),1);
% Myiter = zeros(length(problem),1);
% GurobiCost = zeros(length(problem),1);
flag = zeros(length(problem),1);
slack = zeros(length(problem),1);
gap_max = zeros(length(problem),1);
GurobiCostAll = zeros(length(problem),1);
GurobiTimeAll = zeros(length(problem),1);
GurobiCostBlurCross = zeros(length(problem),1);
GurobiTimeBlurCross = zeros(length(problem),1);
GurobiCostBlur = zeros(length(problem),1);
GurobiTimeBlur = zeros(length(problem),1);
Gurobicross = zeros(length(problem),1);
GurobiCostAll_c = zeros(length(problem),1);
GurobiTimeAll_c = zeros(length(problem),1);
GurobiCostBlurCross_pt = zeros(length(problem),1);
GurobiTimeBlurCross_pt = zeros(length(problem),1);
GurobiCostBlur_pt = zeros(length(problem),1);
GurobiTimeBlur_pt = zeros(length(problem),1);
GurobiOFBlurCross = cell(length(problem),1);
GurobiOFBlur = cell(length(problem),1);
err = zeros(length(problem),1);
gap = zeros(length(problem),1);
Gurobicross_pt = zeros(length(problem),1); 
good_or_not = zeros(length(problem),1);

XXX = [3 5 6 10 14 15 16 17 18 19 20 21 22 23 24 25 26];
more_col = setdiff(1:length(problem),XXX);

FFF = zeros(length(problem),1);
% class1 = [1 4 5 10 16 22 33 34 38 40 46];
% class2 = [2 6 7 8 12 13 21 24 29 31 32 35 36 42 44 45];
% FFF(class1) = 1;
% FFF(class2) = 2;
FFF(7) = 1;


for k = 1:length(problem)
    
    if FFF(k) == 0 
        continue
    end
    disp(' ');
    fprintf(' ##### Solve problem  '+ benchmark_lp(k) + '  #####');
    disp(' ');
    
    %% 初始化
    
    % load('C:\wcwj\Smart Crossover\LP problem\netlib_copt\QAP15.mat');
    % A_mat = Problem.A;
    % l = Problem.aux.lo;
    % u = Problem.aux.hi;
    % b = Problem.b;
    % c = Problem.aux.c;
    
    cmd = sprintf('read(%s)', "C:\wcwj\Smart Crossover\LP problem\lp benchmark\"+char(problem(k))); % Read the problem from file
    [rcode, res] = mosekopt(cmd);
    c = res.prob.c;
    b_l = res.prob.blc;
    b_u = res.prob.buc;
    l = res.prob.blx;
    u = res.prob.bux;
    A_mat = res.prob.a;

    if b_l == b_u
        b = b_l;
    else   
        % Add slack:
        slack(k) = 1;
        s = find(res.prob.blc<res.prob.buc);
        s0 = find(res.prob.blc==res.prob.buc);
        ss1 = find(res.prob.blc==-Inf); 
        ss2 = find(res.prob.buc== Inf); 
        s1 = setdiff(s,ss1);
        s2 = setdiff(s,ss2);
        b = [res.prob.blc(s0) ; res.prob.blc(s1) ; res.prob.buc(s2)];
        I1 = -speye(length(s1));
        I2 =  speye(length(s2));
        A_mat = [A_mat(s0,:) , sparse(length(s0),length(s1)+length(s2)) ;
                 A_mat(s1,:) , I1 , sparse(length(s1),length(s2)) ;      
                 A_mat(s2,:) , sparse(length(s2),length(s1)) , I2        ];
        l = [l ; zeros(length(s1)+length(s2),1)];
        u = [u ; inf*ones(length(s1)+length(s2),1)];
        c = [c ; zeros(length(s1)+length(s2),1)];
    end

    n = size(b,1);
    m = size(c,1);
    
    eps = 1e-4;

    % 判断问题是否满足这样的特征: c是A的行的线性组合；
%     [L,U,P] = lu(A_mat');
%     [L1,U1,P1] = lu([A_mat',c]);
%     u = diag(U); u1 = diag(U1);
%     ind = find(u.*sign(u)>1e-6); ind1 = find(u1.*sign(u1)>1e-6);
%     if length(ind) < length(ind1)
%         flag(k) = 0;
%         continue;
%     else
%         flag(k) = 1;
%     end

%     mu = A_mat.'\c;
%     gap = (A_mat.'*mu-c).*sign(A_mat.'*mu-c);
%     gap_max(k) = max(gap);
%     if gap_max(k)<1e-6
%         flag(k) = 1;
%     else
%         continue;
%     end

%     c_true = c;
%     c = rand(m,1)*100;
    
%     disp(' ');
%     disp('   good example!   ');
%     disp(' ');
%     continue;
    
    %% 求解原问题
    % [GurobiCostAll(k),GurobiOFAll,GurobiTimeAll(k),GurobiIterAll,~] = GurobiLPAll(A_mat,b,c,l,u);

    [GurobiCostBlurCross(k),GurobiOFBlurCross(k),GurobiTimeBlurCross(k),~] = GurobiLPBlurCross(A_mat,b,c,l,u,eps);
    
    [GurobiCostBlur(k),GurobiOFBlur(k),GurobiTimeBlur(k)] = GurobiLPBlur(A_mat,b,c,l,u,eps);
   
    Gurobicross(k) = GurobiTimeBlurCross(k) - GurobiTimeBlur(k);
    
    %% 扰动/替换c，求解新问题
%     xx = GurobiOFBlur(k); x = xx{1};
%     ind = find(x>1e-15);

%     totaldim = m
%     bassize = n
%     optface_dim = length(ind)
%     
%     theta = 1e-5;
%     Theta = randn(length(ind),1)*theta;
%     c_pt = c; c_pt(ind) = c(ind) + Theta;
    

    theta = 1e-6;
    xx = GurobiOFBlur(k); x = xx{1};
    Theta = theta*rand(m,1).*(1 + 9*atan(1./(x+1e-16))/1.5708);
    %Theta = theta*(1 + 9*atan(1./(x+1e-16))/1.5708);
    c_pt = c + Theta.*sign(Theta);
    
    
%     c_true = c;
%     c = rand(m,1)*max(c_true);
% 
%     % [GurobiCostAll_c(k),GurobiOFAll_c,GurobiTimeAll_c(k),~,~] = GurobiLPAll(A_mat,b,c,l,u);
%     
    [GurobiCostBlurCross_pt(k),GurobiOFBlurCross_pt,GurobiTimeBlurCross_pt(k),~] = GurobiLPBlurCross(A_mat,b,c_pt,l,u,eps);
%     
    [GurobiCostBlur_pt(k),GurobiOFBlur_pt,GurobiTimeBlur_pt(k)] = GurobiLPBlur(A_mat,b,c_pt,l,u,eps);
    
    
    error = c.'*GurobiOFBlurCross_pt{1}- GurobiCostBlurCross(k);
    err(k) = error*sign(error);
    gapp = GurobiCostBlurCross_pt(k) - GurobiCostBlurCross(k)
    gap(k) = gapp*sign(gapp);
    Gurobicross_pt(k) = GurobiTimeBlurCross_pt(k) - GurobiTimeBlur_pt(k);
    if err(k)<1e-1
        disp(' '); disp(    '    Perturbation is correct!    '); 
%         if Gurobicross_c(k) < Gurobicross(k)
%             disp(' '); disp('         and effective!      ');
%             good_or_not(k) = 1;
%         end    
        disp(' ');
    end
    
    cross = GurobiTimeBlurCross(k) - GurobiTimeBlur(k) 
    cross_pt = GurobiTimeBlurCross_pt(k) - GurobiTimeBlur_pt(k)
     
    %% 旧代码
    % X = GurobiOFBlur{1};
    % queue = SmartSort_LP2(A_mat,b,c,u,X);
    % basind = find(GurobiBasis == 0);
    % ratio = zeros(1,floor(log2(m/n)));
    % for i = 0:floor(log2(m/n))
    %     R = intersect(basind,queue(1:2^i*n)); 
    %     ratio(i+1) = length(R)/length(basind);    
    % end

    %[DRCost, DROF, DRTime, DRcount] = DimReduction_GurobiLP4(A_mat,b,c,l,u,X);
    % LP1 : 一直column generation，不去人工变量；
    % LP2 : 去掉人工变量后直接对原问题热启动求解；
    % LP3 : 不加人工变量；
    % LP4 : 去人工变量后继续column generation求解；

    %MyTime = DRTime

    %DRCost_true = c_true.'*DROF{1};

end









































