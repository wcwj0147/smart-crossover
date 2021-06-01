function [realCost, realOF, time, iter, basis]=GurobiLPAll(A_mat,b,c,l,u)

N = size(b,2);
n = size(b,1);
m = size(c,1);
realCost = zeros(N,1);
realOF = cell(N,1);
iter = zeros(N,1);
time = zeros(N,1);

for testtime = 1:N
    model.A = A_mat;
    model .rhs = b;
    model .obj = c;
    model .lb  = l;
    model .ub  = u;
    model.sense = '=';
    params.Method = 1;
    params.presolve = 0;
    params.simplexpricing = 1;
    params.Sifting=0;
    params.OutputFlag = 1;
    
    tic;
    result = gurobi(model,params);
    time(testtime) = toc;
    
    realCost(testtime) = result.objval;
    realOF{int16(testtime)} = reshape(result.x,m,1);
    
    iter(testtime) = result.itercount;
    
    basis = result.vbasis;
    
end

end

% d1=3;
% d2=5;
% A_true = [kron(speye(d1),ones(1,d2));kron(ones(1,d2),speye(d1))];
% full(A_true)
