function [realCost, realOT, time, basis]=gurOTsub(a,b,c,T,basis)

N = 1;
d1 = size(a,1);
d2 = size(b,1);

A_true = [-kron(speye(d2),ones(1,d1));kron(ones(1,d2),speye(d1))];
A_sub  = A_true(:,T);

realCost = zeros(N,1);
realOT = cell(N,1);
iter = zeros(N,1);
time = zeros(N,1);

for testtime = 1:N
    
    c_sub = c(T);
    d_sub = size(c_sub,1);
    bas = basis;             
    bas = bas(T);         
    
    Aeq = A_sub(1:end-1,:);
    beq = [-b(:,testtime);a(1:end-1,testtime)];
    
    model.A = Aeq;
    model .rhs = beq;
    model .obj = c_sub;
    model.sense = '=';
    params.Method = 1;
    params.presolve=0;
    params.simplexpricing = 1;
    params.Sifting=0;
    params.OutputFlag = 0;
    
    model.vbasis = reshape(bas,d_sub,1);
    
    tic;
    result = gurobi(model,params);
    time(testtime) = toc;
    
    realCost(testtime) = result.objval;
    xx = zeros(d1,d2);
    xx(T) = result.x;
    realOT{int16(testtime)} = xx;
    
    iter(testtime) = result.itercount;
    
    basis = -ones(d1,d2);
    basis(T) = result.vbasis;
    
end