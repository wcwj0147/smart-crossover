% 测试LP问题的crossover算法
% 2020/8/12
    % LP问题：求 min c^T x, subject to: A_mat x = b, 0 <= x <= u,
    % 其中, x,c,u\in\R^(m*1), b\in\R^(n*1), A_mat\in\R^(n*m)

    
%% 初始化
clear
 
% load('C:\wcwj\Smart Crossover\LP problem\netlib_copt\QAP15.mat');
% A_mat = Problem.A;
% l = Problem.aux.lo;
% u = Problem.aux.hi;
% b = Problem.b;
% c = Problem.aux.c;

cmd = sprintf('read(%s)', 'datt256_lp.mps'); % Read the problem from file
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

eps = 1e-6;

c_true = c;
c = rand(m,1)*100;

%% 求解问题与crossover

% 求解器用NetworkFlow Simplex直接求解LP问题：
% [CplexCostAll,CplexOFAll,CplexTimeAll,CplexBasisAll] = CplexLPAll(A_mat,b,c,l,u);
%[GurobiCostAll,GurobiOFAll,GurobiTimeAll,GurobiIterAll,GurobiBasisAll] = GurobiLPAll(A_mat,b,c,l,u);

% 求解器用内点法求解LP问题（含crossover）：
% [MosekCostBlurCross,MosekOFBlurCross,MosekTimeBlurCross] = MosekLPBlurCross(A_mat,b,c,l,u,iterlim);
% [CplexCostBlurCross,CplexOFBlurCross,CplexTimeBlurCross] = CplexLPBlurCross(A_mat,b,c,l,u,eps);
[GurobiCostBlurCross,GurobiOFBlurCross,GurobiTimeBlurCross,GurobiBasis] = GurobiLPBlurCross(A_mat,b,c,l,u,eps);

% 求解器用内点法求解LP问题：
% [MosekCostBlur,MosekOFBlur,MosekTimeBlur] = MosekLPBlur(A_mat,b,c,u,iterlim);
[GurobiCostBlur,GurobiOFBlur,GurobiTimeBlur] = GurobiLPBlur(A_mat,b,c,l,u,eps);
% [CplexCostBlur,CplexOFBlur,CplexTimeBlur] = CplexLPBlur(A_mat,b,c,l,u,eps);

X = GurobiOFBlur{1};
% queue = SmartSort_LP2(A_mat,b,c,u,X);
% basind = find(GurobiBasis == 0);
% ratio = zeros(1,floor(log2(m/n)));
% for i = 0:floor(log2(m/n))
%     R = intersect(basind,queue(1:2^i*n)); 
%     ratio(i+1) = length(R)/length(basind);    
% end

[DRCost, DROF, DRTime, DRcount] = DimReduction_GurobiLP4(A_mat,b,c,l,u,X);
% LP1 : 一直column generation，不去人工变量；
% LP2 : 去掉人工变量后直接对原问题热启动求解；
% LP3 : 不加人工变量；
% LP4 : 去人工变量后继续column generation求解；

% MosekCrossTime = MosekTimeBlurCross - MosekTimeBlur
%GurobiCrossTime = GurobiTimeBlurCross - GurobiTimeBlur
%CplexCrossTime = CplexTimeBlurCross - CplexTimeBlur
%MyTime = DRTime

DRCost_true = c_true.'*DROF{1};


%% 当下遇到的问题

% LU分解 补充basis；
% 加上扰动避免退化；
% basis会自己变少？？？这个问题应该可以解决！！！








































