% 测试LP问题的crossover算法
% 2020/8/12
    % LP问题：求 min c^T x, subject to: A_mat x = b, 0 <= x <= u,
    % 其中, x,c,u\in\R^(m*1), b\in\R^(n*1), A_mat\in\R^(n*m)

clear


%% 初始化

rand('seed',12356789);
n = 4e3;
m = 3e6;
N = 1;         

tic;
row = kron((1:n),ones(1,2*(n-1))); 
col = zeros(1,2*(n-1)*n); val = zeros(1,2*(n-1)*n);
for i = 1:n
    col((2*(i-1)*(n-1)+1):(2*i*(n-1))) = [(i-1):(n-1):((i-2)*n+1), ((i-1)*(n-1)+1):(i*(n-1)) , (i*(n-1)+i):(n-1):((n-1)*(n-1)+i)];
    val((2*(i-1)*(n-1)+1):(2*i*(n-1))) = [ones(1,i-1) , -ones(1,n-1) , ones(1,n-i)];
end
A_proto = sparse(row,col,val);
t_proto = toc;

tic;
connected = false;
while ~connected 
    arcind = randperm(n*(n-1),m).';
    A_mat = A_proto(:,arcind);
    if sprank(A_mat) == n
        connected = true;       % 如果A_mat的秩为n-1(用结构秩为n替代)，则连通
    end    
end
t_mat = toc;

u = randi(9,m,N,1); l = zeros(m,1);
x_0 = min(0 , u - randi(3,m,N,1));
b = A_mat*x_0;
c = zeros(m,N);
for k = 1:N
    R = randi(9,1,m);  c(:,k) = R; 
end


eps = [1,5e-1,2e-1,1e-1,5e-2,2e-2,1e-2,5e-3,2e-3,1e-3,5e-4,2e-4,1e-4,5e-5];    

ratio = zeros(length(eps),10);
MyCost = zeros(length(eps),1);
MyOF = cell(length(eps),1);
Mytime = zeros(length(eps),1);
Myiter = zeros(length(eps),1);
GurobiCost = zeros(length(eps),1);
GurobiTime = zeros(length(eps),1);
GurobiTimeBlurCross = zeros(length(eps),1);
GurobiTimeBlur = zeros(length(eps),1);
GurobiBasis = zeros(length(eps),m);
GurobiCross = zeros(length(eps),1);
DRBasis = zeros(length(eps),m);


for k = 1:length(eps)

    disp(' ');
    disp(['     ##########  No.', num2str(k), '  ##########     '   ]);
    disp([' #####  Solve the example: n = ', num2str(n), ', m = ', num2str(m), ', with eps = ',num2str(roundn(eps(k),-5)),'   #####']);
    disp(' ');
    
    
    %% 求解问题与crossover

    % 求解器用NetworkFlow Simplex直接求解LP问题：
    % [CplexCostAll,CplexOFAll,CplexTimeAll,CplexBasisAll] = CplexLPAll(A_mat,b,c,l,u);
    % [GurobiCostAll,GurobiOFAll,GurobiTimeAll,GurobiIterAll,GurobiBasisAll] = GurobiLPAll(A_mat,b,c,l,u);
%     GurobiTime(k) = GurobiTimeAll;

    % 求解器用内点法求解LP问题（含crossover）：
    % [MosekCostBlurCross,MosekOFBlurCross,MosekTimeBlurCross] = MosekLPBlurCross(A_mat,b,c,l,u,iterlim);
    % [CplexCostBlurCross,CplexOFBlurCross,CplexTimeBlurCross] = CplexLPBlurCross(A_mat,b,c,l,u,eps);
    [GurobiCostBlurCross,GurobiOFBlurCross,GurobiTimeBlurCross(k),~,GurobiBasis(k,:)] = GurobiLPBlurCross(A_mat,b,c,l,u,eps(k),num2str(roundn(eps(k),-5))+".log",1000);
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







































