function [realCost, realOF, time, iter, basis]=GurobiLPBlurCross(A_mat,b,c,l,u,gap,name,IterLim)

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
    params.Sifting = 0;
    params.BarConvTol = gap;
    params.BarIterLimit = IterLim;
    params.Crossover = -1;
    params.simplexpricing = 1;
    % params.OutputFlag = 1;
    params.LogFile = char(name);
    params.presolve = 0;
    
    tic;
    result = gurobi(model,params);
    time(testtime) = toc;
    
    %
    %             tic;
    %              params.Crossover = 2;
    %     result = gurobi(model,params);
    %     time(testtime) = toc;
    %                 tic;
    %              params.Crossover = 3;
    %     result = gurobi(model,params);
    %     time(testtime) = toc;
    %
    %                 tic;
    %              params.Crossover = 4;
    %     result = gurobi(model,params);
    %     time(testtime) = toc;
    %     realCost(testtime) = result.objval;
    %     realOT(:,:,N) = reshape(result.x,d1,d2);
    
    iter(testtime) = result.itercount;
    realCost(testtime) = result.objval;
    
    basis = reshape(result.vbasis,1,m);
end

end

