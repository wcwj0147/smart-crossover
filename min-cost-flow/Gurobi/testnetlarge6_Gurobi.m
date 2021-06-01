% 测试LP问题的crossover算法
% 2020/8/12
    % LP问题：求 min c^T x, subject to: A_mat x = b, 0 <= x <= u,
    % 其中, x,c,u\in\R^(m*1), b\in\R^(n*1), A_mat\in\R^(n*m)

clear

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
% large_networkLP_benchmark = ["netlarge6","square15","wide15"];
problems =  large_networkLP_benchmark + '.mps'; 
files = '../' + large_networkLP_benchmark + '_' + num2str(log10(eps)) + '.mat';
IterLim = [115,50,17,13,14,13,17,80,18,13];

problem_num = 1;    % Select a problem from large_networkLP_benchmark;

% eps = [1,5e-1,2e-1,1e-1,5e-2,2e-2,1e-2,5e-3,2e-3,1e-3,5e-4];    
eps = [1e-4,3e-5,1e-5,3e-6,1e-6];

ratio = zeros(length(eps),10);
MyCost = zeros(length(eps),1);
MyOF = cell(length(eps),1);
Mytime = zeros(length(eps),1);
Myiter = zeros(length(eps),1);
GurobiCost = zeros(length(eps),1);
GurobiTime = zeros(length(eps),1);
GurobiTimeBlurCross = zeros(length(eps),1);
GurobiTimeBlur = zeros(length(eps),1);
GurobiCross = zeros(length(eps),1);


%% 初始化
cmd = sprintf('read(%s)', "C:\wcwj\Smart Crossover\MCF problem\Gurobi\"+char(problems(problem_num))); % Read the problem from file
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

GurobiBasis = zeros(length(eps),length(l));
DRBasis = zeros(length(eps),length(l));

for k = 1:length(eps)

    disp(' ');
    disp(['     ##########  No.', num2str(k), '  ##########     '   ]);
    disp([' #####  Solve ', char(large_networkLP_benchmark(problem_num)), ' with eps = ',num2str(roundn(eps(k),-5)),'   #####']);
    disp(' ');
    
    
    %% 求解问题与crossover

    % 求解器用NetworkFlow Simplex直接求解LP问题：
    % [CplexCostAll,CplexOFAll,CplexTimeAll,CplexBasisAll] = CplexLPAll(A_mat,b,c,l,u);
%    [GurobiCostAll,GurobiOFAll,GurobiTimeAll,GurobiIterAll,GurobiBasisAll] = GurobiLPAll(A_mat,b,c,l,u);
%     GurobiTime(k) = GurobiTimeAll;

    % 求解器用内点法求解LP问题（含crossover）：
    % [MosekCostBlurCross,MosekOFBlurCross,MosekTimeBlurCross] = MosekLPBlurCross(A_mat,b,c,l,u,iterlim);
    % [CplexCostBlurCross,CplexOFBlurCross,CplexTimeBlurCross] = CplexLPBlurCross(A_mat,b,c,l,u,eps);
    [GurobiCostBlurCross,GurobiOFBlurCross,GurobiTimeBlurCross(k),~,GurobiBasis(k,:)] = GurobiLPBlurCross(A_mat,b,c,l,u,eps(k),large_networkLP_benchmark(8)+".log",1000);
    GurobiCost(k) = GurobiCostBlurCross;
   
    % 求解器用内点法求解LP问题：
    % [MosekCostBlur,MosekOFBlur,MosekTimeBlur] = MosekLPBlur(A_mat,b,c,u,iterlim);
    [GurobiCostBlur,GurobiOFBlur,GurobiTimeBlur(k)] = GurobiLPBlur(A_mat,b,c,l,u,eps(k),1000);
    % [CplexCostBlur,CplexOFBlur,CplexTimeBlur] = CplexLPBlur(A_mat,b,c,l,u,eps);
    
%    filename = char(files(k));
%    save(filename,'GurobiOFBlur');
    % load('GurobiOF_wide15_4');
    X = GurobiOFBlur;
    queue = SmartSort_LP1(A_mat,b,c,u,X);
    basind = find(GurobiBasis(k,:) == 0);
    m = size(c,1);
    n = size(b,1);
    for i = 0:floor(log2(m/n))
        R = intersect(basind,queue(1:2^i*n)); 
        ratio(k,i+1) = length(R)/length(basind);    
    end

    [DRCost, DROF, DRTime, DRcount, DRBasis(k,:)] = DimReduction_Gurobi(A_mat,b,c,l,u,X);
    
    MyCost(k) = DRCost;
    MyOF{int16(k)} = DROF;
    Mytime(k) = DRTime;
    Myiter(k) = DRcount;
    GurobiCross(k) = GurobiTimeBlurCross(k) - GurobiTimeBlur(k);
    
%     % MosekCrossTime = MosekTimeBlurCross - MosekTimeBlur
%     GurobiCrossTime = GurobiTimeBlurCross - GurobiTimeBlur
%     %CplexCrossTime = CplexTimeBlurCross - CplexTimeBlur
%     MyTime = DRTime


end


%% 当下遇到的问题







































