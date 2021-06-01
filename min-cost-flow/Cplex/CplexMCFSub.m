function [realCost, realOF, time, basis_new]=CplexMCFSub(A_mat,b,c,l,u,basis,T)
% 只求�?个子问题

N = 1;
n = size(A_mat,1);
m = size(A_mat,2);

A_sub = A_mat(:,T);

realCost = zeros(N,1);
realOF = cell(N,1);
iter = zeros(N,1);
time = zeros(N,1);

for testtime = 1:N
    
    c_sub = c(T);
    l_sub = l(T);
    u_sub = u(T);
    d_sub = size(c_sub,1);
    bas = basis; 
    
    T_c = setdiff(1:m,T); 
    Indup = find(bas == -2);
    Indup_rest = intersect(T_c, Indup);
    
    % Initialize the CPLEX object
    cplex = Cplex('CplexLPSub');
    Aeq = A_sub(1:end-1,:);
    beq = b - A_mat(:,Indup_rest)*u(Indup_rest);
    cplex.Model.sense = 'minimize';
    cplex.Model.obj   = c_sub;
    cplex.Model.lb    = l_sub;
    cplex.Model.ub    = u_sub;
    cplex.Model.A = Aeq;
    cplex.Model.rhs = beq(1:end-1);
    cplex.Model.lhs = beq(1:end-1);
    
    cplex.Param.network.display.Cur = 10;
    cplex.Param.simplex.display.Cur = 0;
    cplex.Param.solutiontype.Cur = 1;
    %Help: 'solution information CPLEX will attempt to compute:?
    %0 = auto?
    %1 = basic solution?
    %2 = primal dual vector pair'
    
    cplex.Param.lpmethod.Cur = 3;
    %Help:
    %'method for linear optimization:?
    %0 = automatic?
    %1 = primal simplex?
    %2 = dual simplex?
    %3 = network simplex?
    %4 = barrier?
    %5 = sifting?
    %6 = concurrent optimizers'
    
%     if FLAG == 0

        cbasis = -ones(n,1);
        Indrow = find(cbasis==-1);
        Indcol = find(bas == 0);
        subbas = bas(T);
        Indcol_sub = find(subbas == 0);
        Indup_sub = find(subbas == -2);
        
        x = zeros(d_sub,1);
        x(Indup_sub) = u_sub(Indup_sub);
        x(Indcol_sub) = A_sub(Indrow,Indcol_sub)\(beq-A_sub(Indrow,Indup_sub)*u_sub(Indup_sub));
        
        % tic;
        fileBasis = fopen('crossBasis.bas','w+','n','ISO-8859-1');
        fprintf(fileBasis,'* ENCODING=ISO-8859-1\n');
        fprintf(fileBasis,'NAME          CplexLPSub  Iterations 0  Rows %d  Cols %d\n', n-1, d_sub);
        for i = 1:length(Indcol_sub)
            % fprintf(fileBasis,' XL x%d        c%d          %2.16e\n',Indcol_sub(i),Indrow(i),x(Indcol_sub(i)));
            fprintf(fileBasis,' XL x%d        c%d\n',Indcol_sub(i),Indrow(i));
        end
        for i = 1:length(Indup_sub)
            fprintf(fileBasis,' UL x%d\n',Indup_sub(i));
        end
        for i = 1:n-1
            % fprintf(fileBasis,' SR c%d %f\n',Indrow(i),0);
            fprintf(fileBasis,' SR c%d\n',Indrow(i));
        end
        leftcol  = setdiff((1:d_sub),Indcol_sub);
        for i = 1:(d_sub-length(Indcol_sub))
            t = 0;
            if subbas(leftcol(i)) == -2
                t = u_sub(leftcol(i));
            end
            % fprintf(fileBasis,' SC x%d %f\n',leftcol(i), t);
            fprintf(fileBasis,' SC x%d\n',leftcol(i));
            if mod(i,2e5) == 0
                %fprintf('loading data %2.1f \n', 100*i/(d_sub-n));
            end
        end
        fprintf(fileBasis,'ENDATA');
        fclose(fileBasis);
        % twritebasis = toc

        cplex.readBasis('crossBasis.bas');
        
%     else 
%         cplex.readBasis('crossBasis_1.bas');
%     end
        
     
    tic;
    cplex.solve();
    % Write the solution
    %    fprintf ('\nSolution status = %s\n',cplex.Solution.statusstring);
    %    fprintf ('Solution value = %f\n',cplex.Solution.objval);
    %    disp ('Values = ');
    %    disp (cplex.Solution.x');
    %    cplex.writeBasis('myprob.bas')
    tcrosscplex = toc;
    
    time(testtime) = tcrosscplex;
    realCost(testtime) = cplex.Solution.objval;
    realOF{int16(testtime)} = reshape(cplex.Solution.x,[],d_sub);
    
    % cplex.writeBasis('crossBasis_1.bas');
    
%     fileID = fopen('crossBasis_1.bas');
%     Bas = textscan(fileID,'%s %s %*[^\n]','headerlines',2);
%     basind = erase(Bas{2}(1:n), "x");
%     basind = double(string(basind));
%     indnan = find(isnan(basind));
%     basind(indnan) = m-length(indnan)+1:m;
%     basis_new = -ones(size(c,1),1);
%     basis_new(T(basind)) = 0;
    
    Bas = cplex.Solution.basis; 
    basis_new = -ones(m,1);
    basis_new(T) = Bas.colstat;
    basis_new(basis_new == 0) = -1;
    basis_new(basis_new >= 2) = -1;
    basis_new(basis_new == 1) = 0;
    basis_new(Indup_rest) = -2;
    xx = -ones(m,1);
    xx(T) = realOF{1};
    basis_new((xx==u)&(basis_new==-1)) = -2;
        
    %cplex.Solution
    
end

end

 