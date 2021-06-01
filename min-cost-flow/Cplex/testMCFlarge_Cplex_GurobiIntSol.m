% 测试LP问题的crossover算法
% 2020/8/12
    % LP问题：求 min c^T x, subject to: A_mat x = b, 0 <= x <= u,
    % 其中, x,c,u\in\R^(m*1), b\in\R^(n*1), A_mat\in\R^(n*m)

clear
eps = 1e-2;
% rand('seed',987652);
% n = 5;
% m = 10;
% N = 1;         
% 
% tic;
% row = kron((1:n),ones(1,2*(n-1))); 
% col = zeros(1,2*(n-1)*n); val = zeros(1,2*(n-1)*n);
% for i = 1:n
%     col((2*(i-1)*(n-1)+1):(2*i*(n-1))) = [(i-1):(n-1):((i-2)*n+1), ((i-1)*(n-1)+1):(i*(n-1)) , (i*(n-1)+i):(n-1):((n-1)*(n-1)+i)];
%     val((2*(i-1)*(n-1)+1):(2*i*(n-1))) = [ones(1,i-1) , -ones(1,n-1) , ones(1,n-i)];
% end
% A_proto = sparse(row,col,val);
% t_proto = toc
% 
% tic;
% connected = false;
% while ~connected 
%     arcind = randperm(n*(n-1),m).';
%     A_mat = A_proto(:,arcind);
%     if sprank(A_mat) == n
%         connected = true;       % 如果A_mat的秩为n-1(用结构秩为n替代)，则连通
%     end    
% end
% t_mat = toc
% 
% u = randi(9,m,N,1); l = zeros(m,1);
% x_0 = min(0 , u - randi(3,m,N,1));
% b = A_mat*x_0;
% c = zeros(m,N);
% for k = 1:N
%     R = randi(9,1,m);  c(:,k) = R; 
% end

% load('C:\2020 Summer\Smart Crossover\Netlib lp\FIT2D.mat');
% A_mat = A;
% l = lbounds;
% u = ubounds;

% load('C:\2020 Summer\Smart Crossover\LP problem\hard problems\stat96v1.mat');
% A_mat = Problem.A;
% l = Problem.aux.lo;
% u = Problem.aux.hi;
% b = Problem.b;
% c = Problem.aux.c;

large_networkLP_benchmark = ["16_n14","i_n13","lo10","long15",...
                             "netlarge1","netlarge2","netlarge3",...
                             "netlarge6","square15","wide15"];
problems =  large_networkLP_benchmark + '.mps'; 
files = large_networkLP_benchmark + '_' + num2str(log10(eps)) + '.mat';

ratio = zeros(1,15,length(problems));
MyCost = zeros(length(problems),1);
MyOF = cell(length(problems),1);
Mytime = zeros(length(problems),1);
Myiter = zeros(length(problems),1);
CplexCost = zeros(length(problems),1);
CplexTime = zeros(length(problems),1);

for k = 1:length(problems)  
    
    disp(' ');
    fprintf(' ##### Solve problem  '+large_networkLP_benchmark(k)+'  #####');
    disp(' ');
    
%% 初始化
    cmd = sprintf('read(%s)', "C:\wcwj\Smart Crossover\MCF problem\Gurobi\"+char(problems(k))); % Read the problem from file
    [rcode, res] = mosekopt(cmd);
    c = res.prob.c;
    b_l = res.prob.blc;
    b_u = res.prob.buc;
    l = res.prob.blx;
    u = res.prob.bux;
    A_mat = res.prob.a;

    if b_l == b_u
        b = b_l;
    end    
    
%     mpsdata = mpsread("C:\wcwj\Smart Crossover\MCF problem\Gurobi\"+char(problems(k)));
%     c = mpsdata.f;
%     A_mat = mpsdata.Aeq;
%     b = mpsdata.beq;
%     l = mpsdata.lb;
%     u = mpsdata.ub;

    % if median(c) > 1
    %    c = c/median(c); 
    % end

    %[GurobiCostAll_t,GurobiOFAll_t,GurobiTimeAll_t,GurobiIterAll_t,GurobiBasisAll_t] = GurobiLPAll_ineq(A_mat,b_l,c,l,u);

    % Add slack:
    % s = find(res.prob.blc<res.prob.buc);
    % s0 = find(res.prob.blc==res.prob.buc);
    % ss1 = find(res.prob.blc==-Inf); 
    % ss2 = find(res.prob.buc== Inf); 
    % s1 = setdiff(s,ss1);
    % s2 = setdiff(s,ss2);
    % b = [res.prob.blc(s0) ; res.prob.blc(s1) ; res.prob.buc(s2)];
    % I1 = -speye(length(s1));
    % I2 =  speye(length(s2));
    % A_mat = [A_mat(s0,:) , sparse(length(s0),length(s1)+length(s2)) ;
    %          A_mat(s1,:) , I1 , sparse(length(s1),length(s2)) ;      
    %          A_mat(s2,:) , sparse(length(s2),length(s1)) , I2        ];
    % l = [l ; zeros(length(s1)+length(s2),1)];
    % u = [u ; inf*ones(length(s1)+length(s2),1)];
    % c = [c ; zeros(length(s1)+length(s2),1)];


    %% 求解问题与crossover

    % 求解器用NetworkFlow Simplex直接求解LP问题：
    %[CplexCostAll,CplexOFAll,CplexTimeAll,CplexBasisAll] = CplexMCFAll(A_mat,b,c,l,u);
    % [GurobiCostAll,GurobiOFAll,GurobiTimeAll,GurobiIterAll,GurobiBasisAll] = GurobiLPAll(A_mat,b,c,l,u);

    % 求解器用内点法求解LP问题（含crossover）：
    % [MosekCostBlurCross,MosekOFBlurCross,MosekTimeBlurCross] = MosekLPBlurCross(A_mat,b,c,l,u,iterlim);
    %[CplexCostBlurCross,CplexOFBlurCross,CplexTimeBlurCross] = CplexMCFBlurCross(A_mat,b,c,l,u,eps,large_networkLP_benchmark(k)+".log");
    % [GurobiCostBlurCross,GurobiOFBlurCross,GurobiTimeBlurCross] = GurobiLPBlurCross(A_mat,b,c,l,u,eps,large_networkLP_benchmark(k)+".log");

    % 求解器用内点法求解LP问题：
    % [MosekCostBlur,MosekOFBlur,MosekTimeBlur] = MosekLPBlur(A_mat,b,c,u,iterlim);
    % [GurobiCostBlur,GurobiOFBlur,GurobiTimeBlur] = GurobiLPBlur(A_mat,b,c,l,u,eps);
    %[CplexCostBlur,CplexOFBlur,CplexTimeBlur] = CplexMCFBlur(A_mat,b,c,l,u,eps);
    
    filename = char(files(k));
    % save(filename,'GurobiOFBlur');
    load("C:\wcwj\Smart Crossover\MCF problem\Gurobi\interior_solution_2\"+filename);
    X = GurobiOFBlur;
    % X = CplexOFBlur{1};
%     queue = SmartSort_LP2(A_mat,b,c,u,X);
%     basind = find(CplexBasisAll.colstat == 1);
%     m = size(c,1);
%     n = size(b,1);
%     for i = 0:floor(log2(m/n))
%         R = intersect(basind,queue(1:2^i*n)); 
%         ratio(1,i+1,k) = length(R)/length(basind);    
%     end

    [DRCost, DROF, DRTime, DRcount] = DimReduction_Cplex(A_mat,b,c,l,u,X);
    
    MyCost(k) = DRCost;
    MyOF{int16(k)} = DROF;
    Mytime(k) = DRTime;
    Myiter(k) = DRcount;
    %CplexCost(k) = CplexCostAll;
    %CplexTime(k) = CplexTimeAll;
    
%     % MosekCrossTime = MosekTimeBlurCross - MosekTimeBlur
%     GurobiCrossTime = GurobiTimeBlurCross - GurobiTimeBlur
%     %CplexCrossTime = CplexTimeBlurCross - CplexTimeBlur
%     MyTime = DRTime

end


%% 问题与备注







































