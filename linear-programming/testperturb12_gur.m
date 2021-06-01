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
GurobiCostAll = zeros(length(problem),1);
GurobiTimeAll = zeros(length(problem),1);
GurobiCostBlurCross = zeros(length(problem),1);
GurobiTimeBlurCross = zeros(length(problem),1);
GurobiCostBlur = zeros(length(problem),1);
GurobiTimeBlur = zeros(length(problem),1);
cross = zeros(length(problem),1);
GurobiCostAll_c = zeros(length(problem),1);
GurobiTimeAll_c = zeros(length(problem),1);
GurobiCostBlurCross_pt = zeros(length(problem),1);
GurobiTimeBlurCross_pt = zeros(length(problem),1);
GurobiCostBlur_pt = zeros(length(problem),1);
GurobiTimeBlur_pt = zeros(length(problem),1);
GurobiOFBlurCross = cell(length(problem),1);
GurobiOFBlur = cell(length(problem),1);
GurobiPi = cell(length(problem),1);
GurobiSlack = cell(length(problem),1);
err = zeros(length(problem),1);
gap = zeros(length(problem),1);
cross_pt = zeros(length(problem),1); 
good_or_not = zeros(length(problem),1);
optface = zeros(length(problem),1);
ratio = zeros(length(problem),1);
rows = zeros(length(problem),1);
cols = zeros(length(problem),1);
flag = ones(length(problem),1);
optface_pt = zeros(length(problem),1);
long_cross = zeros(length(problem),1);
rank = zeros(length(problem),1);
liszero = zeros(length(problem),1);
uisinf = zeros(length(problem),1);
rows = zeros(length(problem),1);
cols = zeros(length(problem),1);
GurobiCostFinal_1 = zeros(length(problem),1);
GurobiTimeFinal_1 = zeros(length(problem),1);
GurobiCostFinal_2 = zeros(length(problem),1);
GurobiTimeFinal_2 = zeros(length(problem),1);
vbasnum = zeros(length(problem),1);
cbasnum = zeros(length(problem),1);
GurobiTimeSub = zeros(length(problem),1);
GurobiCostSub = zeros(length(problem),1);
timeall  = zeros(length(problem),1);
timeall_pt1  = zeros(length(problem),1);
timeall_pt2  = zeros(length(problem),1);


FFF = zeros(length(problem),1);

blacklist = [1 12 14 15 17 20 47];
whitelist = [1 7 19 25 35 36 47];
longer_crossover = [1 2 4 7 8 12 13 14 15 16 17 19 20 22 23 24 25 26 29 30 33 35 36 37 39 40 44 45 46 47];

FFF(longer_crossover) = 1;
FFF(blacklist) = 0;


for k = 1:length(problem)
    
    if FFF(k) == 0 
        continue
    end
    disp(' ');
    fprintf(' ##### Solve problem  '+ benchmark_lp(k) + '  #####');
    disp(' ');
    
    %% 初始化
    
    cmd = sprintf('read(%s)', "C:\wcwj\Smart Crossover\LP problem\lp benchmark\"+char(problem(k))); % Read the problem from file
    [rcode, res] = mosekopt(cmd);
    c = res.prob.c;
    bl = res.prob.blc;
    bu = res.prob.buc;
    l = res.prob.blx;
    u = res.prob.bux;
    A = res.prob.a;
    
    ind_eq = find(bl==bu);
    A_1 = A(ind_eq,:);
    b_1 = bl(ind_eq);
    
    ind_bl = find(bl>-inf);
    ind_bu = find(bu<+inf);
    ind_bl = setdiff(ind_bl,ind_eq);
    ind_bu = setdiff(ind_bu,ind_eq);
    A_2 = [A(ind_bl,:) ; -A(ind_bu,:)];
    b_2 = [bl(ind_bl) ; -bu(ind_bu)];
    
    ind_l = find(l>-inf);
    ind_u = find(u<+inf);
    u(ind_l) = u(ind_l)-l(ind_l);
    l(ind_l) = 0;
    I = speye(length(c));
    A_2 = [A_2 ; -I(ind_u,:)];
    b_2 = [b_2 ; -u(ind_u)];
    
    ind_free = find((l==-inf)&(u==inf));
    A_1 = [A_1 , -A_1(:,ind_free)];
    A_2 = [A_2 , -A_2(:,ind_free)];
    c = [c ; -c(ind_free)];

    A_mat = [A_1 ; A_2];
    b = [b_1 ; b_2];
    sense = strings(length(b),1);
    sense(1:length(b_1)) = '=';
    sense(length(b_1)+1:length(b)) = '>';
    sense = char(sense);
    
    n = size(b,1);
    m = size(c,1);
    rows(k) = n;
    cols(k) = m;
    
    l = zeros(m,1);
    u = inf*ones(m,1);

    rank(k) = sprank(A_mat);
    

    
    %% 求解原问题
    % [GurobiCostAll(k),GurobiOFAll,GurobiTimeAll(k),GurobiIterAll,~] = GurobiLPAll(A_mat,b,c,l,u);

    eps = 1e-10;
    
    [GurobiCostBlurCross(k),GurobiOFBlurCross(k),GurobiTimeBlurCross(k),Gurobivbasis,Gurobicbasis] = gurLPBlurCross(A_mat,b,c,l,u,eps,sense);
  
    [GurobiCostBlur(k),GurobiOFBlur(k),GurobiTimeBlur(k),~,GurobiPi(k),GurobiSlack(k)] = gurLPBlur_pi(A_mat,b,c,l,u,eps,sense);
    
    pp = GurobiPi(k); pi = pp{1};
    ss = GurobiSlack(k); sp = ss{1};
    xx = GurobiOFBlur(k); x = xx{1};
    sd = c - A_mat'*pi;
    sigma = find(x>=sd);
    tau = find(pi>=sp);
    
    primaloptface = length(sigma)
    dualoptface = length(tau)
    
    
    %% 扰动/替换c，求解新问题
    
    theta = 1e-6;
    xx = GurobiOFBlur(k); x = xx{1};
    noise = 1/6*randn(m,1) + 1/2;
    noise = bsxfun(@max,noise,0.01);
    noise = bsxfun(@min,noise,0.99);
    %Theta = theta*rand(m,1).*(1 + 99*atan(x+1e-16)/1.5708);
    %Theta = theta*noise.*(1 + 99*atan(1./(x+1e-16))/1.5708);
    Theta = theta*noise.*(1 + 9*atan(x+1e-16)/1.5708);
    %Theta = theta*(1 + 9*atan(1./(x+1e-16))/1.5708);
    c_pt = c + Theta.*sign(Theta);
    
    [GurobiCostBlurCross_pt(k),GurobiOFBlurCross_pt,GurobiTimeBlurCross_pt(k),vbasis_1,cbasis_1] = gurLPBlurCross(A_mat,b,c_pt,l,u,eps,sense);
    [GurobiCostFinal_1(k),~,GurobiTimeFinal_1(k)] = gurLPFinal(A_mat,b,c,l,u,sense,vbasis_1,cbasis_1);
    
    [GurobiCostSub(k),GurobiOFSub,~,GurobiTimeSub(k),vbasis_2,cbasis_2] = gurLPSub_bar(A_mat,b,c_pt,l,u,eps,sense,sigma,tau);
    [GurobiCostFinal_2(k),GurobiOFFinal,GurobiTimeFinal_2(k)] = gurLPFinal(A_mat,b,c,l,u,sense,vbasis_2,cbasis_2);
    
    %[GurobiCostBlur_pt(k),GurobiOFBlur_pt,GurobiTimeBlur_pt(k),~,PP] = GurobiLPBlur_pi(A_mat,b,c_pt,l,u,eps);
        
%     pi_pt = PP{1};
%     xx = GurobiOFBlur_pt; x_pt = xx{1};
%     s_pt = c_pt - A_mat'*pi_pt;
%     sigma_pt = find(x_pt>=s_pt);
    
%     optfacesize_pt = length(sigma_pt)
%     Ratio_pt = optfacesize_pt/vbasnum(k)
%     optface_pt(k) = length(sigma_pt);
    
%     cross(k) = GurobiTimeBlurCross(k) - GurobiTimeBlur(k);
%     cross_pt(k) = GurobiTimeBlurCross_pt(k) - GurobiTimeBlur_pt(k);
    timeall(k) = GurobiTimeBlurCross(k);
    timeall_pt1(k) = GurobiTimeBlurCross_pt(k) + GurobiTimeFinal_1(k);
    timeall_pt2(k) = GurobiTimeSub(k) + GurobiTimeFinal_2(k);
    
    X = zeros(m,1);
    X(sigma) = GurobiOFSub{1};
    error = GurobiCostFinal_1(k) - GurobiCostBlurCross(k);
    err(k) = error*sign(error);
    gapp = c.'*X - GurobiCostBlurCross(k);
    gap(k) = gapp*sign(gapp);
%     if err(k)<1e-8
%         disp(' '); disp(    '    Perturbation is correct!    '); 
%         if timeall_pt2(k) < timeall(k)
%             disp(' '); disp('         and effective!      ');
%             good_or_not(k) = 1;
%         end    
%         disp(' ');
%     end

     
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









































