function [realCost, realOF, time, vbasis, cbasis]=gurLPBlurCross(A_mat,b,c,l,u,gap,sense)

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
    model.sense = sense;
    params.Method = 2;
    params.Sifting = 0;
    params.BarConvTol = gap;
    params.Crossover = -1;
    params.simplexpricing = 1;
    params.OutputFlag = 1;
    params.presolve = 0;
    
    tic;
    result = gurobi(model,params);
    time(testtime) = toc;

    iter(testtime) = result.itercount;
    realCost(testtime) = result.objval;
    realOF{int16(testtime)} = reshape(result.x,m,1);
    
    vbasis = result.vbasis;
    cbasis = result.cbasis;
    
end

end

