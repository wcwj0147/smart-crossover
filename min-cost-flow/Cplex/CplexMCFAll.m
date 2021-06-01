function [realCost, realOF, time, basis_new]=CplexMCFAll(A_mat,b,c,l,u)

N = size(b,2);
n = size(b,1);
m = size(c,1);
realCost = zeros(N,1);
realOF = cell(N,1);
iter = zeros(N,1);
time = zeros(N,1);

for testtime = 1:N
    
    % Initialize the CPLEX object
    cplex = Cplex('CplexMCFAll');
    Aeq = A_mat(1:end-1,:);
    beq = b(1:end-1);
    cplex.Model.sense = 'minimize';
    cplex.Model.obj   = c;
    cplex.Model.lb    = l;
    cplex.Model.ub    = u;
    cplex.Model.A = Aeq;
    cplex.Model.rhs = beq;
    cplex.Model.lhs = beq;
    
    cplex.Param.network.display.Cur = 0;
    %cplex.Param.network.pricing.Cur = 0;
    % Help: 'pricing strategy index:?  
    %0 = let cplex select pricing strategy?  
    %1 = partial pricing?  
    %2 = multiple partial pricing (no sorting)?  
    %3 = multiple partial pricing (with sorting)'

    cplex.Param.simplex.display.Cur = 0;
    cplex.Param.solutiontype.Cur=1;
    %Help: 'solution information CPLEX will attempt to compute:?
    %0 = auto?
    %1 = basic solution?
    %2 = primal dual vector pair'
    
    cplex.Param.lpmethod.Cur=3;
    %Help:
    %'method for linear optimization:?
    %0 = automatic?
    %1 = primal simplex?
    %2 = dual simplex?
    %3 = network simplex?
    %4 = barrier?
    %5 = sifting?
    %6 = concurrent optimizers'
    
    tic;
    cplex.solve();
    
    % Write the solution
    %    fprintf ('\nSolution status = %s\n',cplex.Solution.statusstring);
    %    fprintf ('Solution value = %f\n',cplex.Solution.objval);
    %    disp ('Values = ');
    %    disp (cplex.Solution.x');
    %    cplex.writeBasis('myprob.bas')
    
    % cplex.writeBasis('realBasis.bas')
    
    time(testtime) = toc;
    realCost(testtime) = cplex.Solution.objval;
    realOF{int16(testtime)} = reshape(cplex.Solution.x,m,1);
    
    basis_new = cplex.Solution.basis;
    
end

end

% d1=3;
% d2=5;
% A_true = [kron(speye(d1),ones(1,d2));kron(ones(1,d2),speye(d1))];
% full(A_true)
