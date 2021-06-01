function [realCost, realOT, iter, time]=gurobiTransportBlur(a,b,M,gap)

N = size(a,2);
d1 = size(a,1);
d2 = size(b,1);
c = reshape(M,[],1);
A_true = [-kron(speye(d2),ones(1,d1));kron(ones(1,d2),speye(d1))];
realCost = zeros(N,1);
realOT = zeros(d1,d2,N);
iter = zeros(N,1);
time = zeros(N,1);

for testtime = 1:N
    model.A = A_true;
    model .rhs = [-b(:,testtime);a(:,testtime)];
    model .obj = c;
    model.sense = '=';
    params.Method = 2;
    params.BarConvTol = gap;
    params.Crossover = 0;
    params.OutputFlag = 1;
    
    tic;
    result = gurobi(model,params);
    time(testtime) = toc;
    
    realCost(testtime) = result.objval;
    realOT(:,:,N) = reshape(result.x,d1,d2);
    
    iter(testtime) = result.itercount;
    
    
end

end

% d1=3;
% d2=5;
% A_true = [kron(speye(d1),ones(1,d2));kron(ones(1,d2),speye(d1))];
% full(A_true)
