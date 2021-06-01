function [realCost, realOF, time]=CplexMCFBlur(A_mat,b,c,l,u,eps,IterLim)

N = size(b,2);
n = size(b,1);
m = size(c,1);
realCost = zeros(N,1);
realOF = cell(N,1);
iter = zeros(N,1);
time = zeros(N,1);


for testtime = 1:N
    
    % Initialize the CPLEX object
    cplex = Cplex('CplexMCFBlur');
    Aeq = A_mat;
    beq = b;
    cplex.Model.sense = 'minimize';
    cplex.Model.obj   = c;
    cplex.Model.lb    = l;
    cplex.Model.ub    = u;
    cplex.Model.A = Aeq;
    cplex.Model.rhs = beq;
    cplex.Model.lhs = beq;
    
    cplex.Param.lpmethod.Cur=4;
    %Help:
    %'method for linear optimization:?
    %0 = automatic?
    %1 = primal simplex?
    %2 = dual simplex?
    %3 = network simplex?
    %4 = barrier?
    %5 = sifting?
    %6 = concurrent optimizers'
    
    cplex.Param.solutiontype.Cur = 2;
    %Help: 'solution information CPLEX will attempt to compute:?
    %0 = auto?
    %1 = basic solution?
    %2 = primal dual vector pair'
    
    cplex.Param.barrier.display.Cur = 0;
    %Help: 'barrier display level:?  
    %0 = no display?  
    %1 = display normal information?  
    %2 = display detailed (diagnostic) output'
    
    cplex.Param.barrier.convergetol.Cur = eps;
    %Help: 'tolerance on complementarity for convergence'
    
    cplex.Param.barrier.crossover.Cur = -1;
    %Help: 'barrier crossover choice:?  
    %-1 = no crossover?  
    %0 = automatic?  
    %1 = primal crossover?  
    %2 = dual crossover'
    
    %Cplex.Param.barrier.limits.iteration = IterLim;
    
    tic;
    cplex.solve();
    t = toc;
    % Write the solution
    %    fprintf ('\nSolution status = %s\n',cplex.Solution.statusstring);
    %    fprintf ('Solution value = %f\n',cplex.Solution.objval);
    %    disp ('Values = ');
    %    disp (cplex.Solution.x');
    %    cplex.writeBasis('myprob.bas')
    
    
    time(testtime) = t;
    realCost(testtime) = cplex.Solution.objval;
    realOF{int16(testtime)} = cplex.Solution.x;
    
end

end

