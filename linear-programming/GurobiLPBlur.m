function [realCost, realOF, time, iter]=GurobiLPBlur(A_mat,b,c,l,u,gap)

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
    params.Method = 2;
    params.BarConvTol = gap;
    params.Crossover = 0;
    params.OutputFlag = 1;
    params.presolve = 0;
    
    tic;
    result = gurobi(model,params);
    time(testtime) = toc;
    
    realCost(testtime) = result.objval;
    realOF{int16(testtime)} = reshape(result.x,m,1);
    
    iter(testtime) = result.itercount;
    
    
end

end
