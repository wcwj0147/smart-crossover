function [realCost, realOF, iter, time, basis]=GurobiLPSub(A_mat,b,c,l,u,basis,T)

N = 1;
n = size(A_mat,1);
m = size(A_mat,2);

A_sub = A_mat(:,T);

realCost = zeros(N,1);
realOF = cell(N,1);
iter = zeros(N,1);
time = zeros(N,1);

for testtime = 1:N
    
    c_sub = c(T);
    l_sub = l(T);
    u_sub = u(T);
    d_sub = size(c_sub,1);
    bas = basis;
    bas_sub = basis(T);
    
    T_c = setdiff(1:m,T); 
    Indup = find(bas == -2);
    Indup_rest = intersect(T_c, Indup);
    
    model.A = A_sub;
    model.rhs = b - A_mat(:,Indup_rest)*u(Indup_rest);
    model.obj = c_sub;
    model.lb = l_sub;
    model.ub = u_sub;
    model.sense = '=';
    params.Method = 1;
    params.simplexpricing = 1;
    params.Sifting = 0;
    params.OutputFlag = 1;
    model.vbasis = bas_sub;
    cbasis = -ones(n,1);
    model.cbasis = cbasis;
    
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
    basis(T) = result.vbasis;
end

end

% d1=3;
% d2=5;
% A_true = [kron(speye(d1),ones(1,d2));kron(ones(1,d2),speye(d1))];
% full(A_true)
