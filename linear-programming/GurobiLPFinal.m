function [realCost, realOF, time]=GurobiLPFinal(A_mat,b,c,l,u,vbasis,cbasis)

N = 1;
n = size(A_mat,1);
m = size(A_mat,2);
realCost = zeros(N,1);
realOF = cell(N,1);
iter = zeros(N,1);
time = zeros(N,1);

for testtime = 1:N
    model.A = A_mat;
    model.rhs = b;
    model.obj = c;
    model.sense = '=';
    model.lb = l;
    model.ub = u;
    params.Method = 1;
    
    params.simplexpricing = 1;
    params.Sifting = 0;
    params.OutputFlag = 1;
    model.vbasis = vbasis;
    model.cbasis = cbasis;
    
%     params.presolve = 0;
    
%       Indrow = find(model.cbasis==-1);
%       Indcol = find(model.vbasis==0);
%       %A_inv = inv(A_true(Indrow,Indcol));
%       x = model.A(Indrow,Indcol)\model.rhs;
%       primalinf = norm(model.A(:,Indcol)*x-model.rhs);
%     A_true = full(A_true)
%     A_sel = full(A_true(Indrow,Indcol))
    
    
    tic;
    result = gurobi(model,params);
    time(testtime) = toc;
    
    realCost(testtime) = result.objval;
    realOF{int16(testtime)} = result.x;
    
    iter(testtime) = result.itercount;
end

end

% d1=3;
% d2=5;
% A_true = [kron(speye(d1),ones(1,d2));kron(ones(1,d2),speye(d1))];
% full(A_true)
