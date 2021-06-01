function [realCost, realOF, time]=CplexLPFinal2(A_mat,b,c,l,u,vbasis,cbasis)

N = size(b,2);
n = size(b,1);
m = size(c,1);
realCost = zeros(N,1);
realOF = cell(N,1);
iter = zeros(N,1);
time = zeros(N,1);


for testtime = 1:N
    
    % Initialize the CPLEX object
    cplex = Cplex('CplexLPFinal');
    Aeq = A_mat;
    beq = b;
    cplex.Model.sense = 'minimize';
    cplex.Model.obj   = c;
    cplex.Model.lb    = l;
    cplex.Model.ub    = u;
    cplex.Model.A = Aeq;
    cplex.Model.rhs = beq;
    cplex.Model.lhs = beq;
    
    cplex.Param.simplex.display.Cur = 100;
    cplex.Param.solutiontype.Cur=1;
    %Help: 'solution information CPLEX will attempt to compute:?
    %0 = auto?
    %1 = basic solution?
    %2 = primal dual vector pair'
    
    cplex.Param.lpmethod.Cur=0;
    %Help:
    %'method for linear optimization:?
    %0 = automatic?
    %1 = primal simplex?
    %2 = dual simplex?
    %3 = network simplex?
    %4 = barrier?
    %5 = sifting?
    %6 = concurrent optimizers'
    
    bas = vbasis;
    Indrow = find(cbasis==-1);
    Indcol = find(bas == 0);
    Indup = find(bas == 1);

    x = zeros(m,1);
    x(Indup) = u(Indup);
    x(Indcol) = Aeq(Indrow,Indcol)\(beq(Indrow)-Aeq(Indrow,Indup)*u(Indup));
    
    tic;
    fileBasis = fopen('crossBasis_final.bas','w+','n','ISO-8859-1');
    fprintf(fileBasis,'* ENCODING=ISO-8859-1\n');
    fprintf(fileBasis,'NAME          CplexLPFinal  Iterations 0  Rows %d  Cols %d\n', n, m);
    for i = 1:length(Indcol)
        fprintf(fileBasis,' XL x%d        c%d          %2.16e\n',Indcol(i),Indrow(i),x(Indcol(i)));
    end
    for i = 1:length(Indup)
        fprintf(fileBasis,' UL x%d\n', Indup(i));
    end
    for i = 1:length(Indrow)
        fprintf(fileBasis,' SR c%d %f\n',Indrow(i),                  0);
    end
    leftcol  = setdiff((1:m),Indcol);
    for i = 1:length(leftcol)
        t = 0;
        if bas(leftcol(i)) == 1
            t = u(leftcol(i));
        end
        fprintf(fileBasis,' SC x%d %f\n',leftcol(i),                t);
        if mod(i,2e5) == 0
            fprintf('loading data %2.1f \n', 100*i/(m-n));
        end
    end
    fprintf(fileBasis,'ENDATA');
    fclose(fileBasis);
    
    cplex.readBasis('crossBasis_final.bas');
    
    tic;
    cplex.solve();
    % Write the solution
    %    fprintf ('\nSolution status = %s\n',cplex.Solution.statusstring);
    %    fprintf ('Solution value = %f\n',cplex.Solution.objval);
    %    disp ('Values = ');
    %    disp (cplex.Solution.x');
    %    cplex.writeBasis('myprob.bas')
    
    tfinalcplex = toc;
    time(testtime) = tfinalcplex;
    realCost(testtime) = cplex.Solution.objval;
    realOF{int16(testtime)} = reshape(cplex.Solution.x,m,1);
    
    cplex.writeBasis('FinalBasis.bas');
    
    %cplex.writeBasis('myprob.bas');
    
    %cplex.Solution
    
end

end

