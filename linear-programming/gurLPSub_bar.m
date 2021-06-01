function [realCost, realOF, iter, time, vbasis, cbasis]=gurLPSub_bar(A_mat,b,c,l,u,gap,sense,T,S)

N = 1;
n = size(A_mat,1);
m = size(A_mat,2);

aaa = A_mat(:,T);
A_sub = aaa(S,:);

realCost = zeros(N,1);
realOF = cell(N,1);
iter = zeros(N,1);
time = zeros(N,1);

for testtime = 1:N
    
    c_sub = c(T);
    l_sub = l(T);
    u_sub = u(T);
    b_sub = b(S);
    d_sub = size(c_sub,1);
    sense_sub = sense(S);
    
    model.A = A_sub;
    model.rhs = b_sub;
    model.obj = c_sub;
    model.lb = l_sub;
    model.ub = u_sub;
    model.sense = sense_sub;
    params.Method = 2;
    params.BarConvTol = gap;
    params.simplexpricing = 1;
    params.Sifting = 0;
    params.OutputFlag = 1;
    params.presolve = 0;

    
%     Indrow = find(model.cbasis==-1);
%     Indcol = find(model.vbasis==0);
%     % A_inv = inv(A_true(Indrow,Indcol));
%     x = model.A(Indrow,Indcol)\model.rhs;
%     primalinf = norm(model.A(:,Indcol)*x-model.rhs);
%     A_true = full(A_true)
%     A_sel = full(A_true(Indrow,Indcol))
    
    
    tic;
    result = gurobi(model,params);
    time(testtime) = toc;
    
    realCost(testtime) = result.objval;
    realOF{int16(testtime)} = reshape(result.x,d_sub,1);
    
    iter(testtime) = result.itercount;
    vbasis = -ones(m,1);
    vbasis(T) = result.vbasis;
    cbasis = -ones(n,1);
    cbasis(S) = result.cbasis;
end

end

% d1=3;
% d2=5;
% A_true = [kron(speye(d1),ones(1,d2));kron(ones(1,d2),speye(d1))];
% full(A_true)
