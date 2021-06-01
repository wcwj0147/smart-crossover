function [realCost, realOT, time, count1, count2] = dimReduction_general(a,b,M,X)
%dimReduction_general 针对Optimal Transport问题，基于降维的crossover方法
%   Input: a:source, b:destination; M:cost;
%          X:an interior point solution

N = size(a,2);
d1 = size(a,1);
d2 = size(b,1);

A_true = [-kron(speye(d2),ones(1,d1));kron(ones(1,d2),speye(d1))];
A_tilde = A_true(1:end-1,1:end);
realCost = zeros(N,1);
realOT = cell(N,1);
time = zeros(N,1);
count1 = zeros(N,1);
count2 = zeros(N,1);
K = d1*d2*max(max(M));
beta = 2;

for testtime = 1:N
    
    tt = 0;
    tic;
    
    % Solve another OT problem with artificial nodes and arcs.
    % Initialization - construct an extended problem OT_1:
    a_1 = [a(:,testtime) ; sum(b(:,testtime))];
    b_1 = [b(:,testtime) ; sum(a(:,testtime))];
    M_1 = [M(:,:,testtime) , K*ones(d1,1) ; K*ones(1,d2) , 1];
    c_1 = reshape(M_1,[],1);
    bas_1 = [-ones(d1,d2) , zeros(d1,1) ; zeros(1,d2) , 0];
    tree_1 = reshape(bas_1,[],1);
    T_1 = find(tree_1 == 0);
    T_0 = T_1;
    
    t_0 = toc;
    
    % Find a sequence:
    XX = X(:,:,testtime);
    
    aa = sum(XX,2);
    bb = sum(XX,1)';
    
    Yab = XX./aa./bb';
    Ya = XX./aa;
    Yb = XX./bb';
    
    Yab  = reshape(Yab, [], 1);
    Ya = reshape(Ya, [], 1);
    Yb  = reshape(Yb, [], 1);
    thhold2 = 1.99*mean(Ya(1:97:end));
    thhold3 = 1.99*mean(Yb(1:97:end));
    I1 = find(Yab);
    I2 = find(Ya>thhold2);
    I3 = find(Yb>thhold3);
    [~,II1] = sort(Yab(I1) ,'descend');
    [~,II2] = sort(Ya(I2) ,'descend');
    [~,II3] = sort(Yb(I3) ,'descend');
    I1 = I1(II1);
    I2 = I2(II2);
    I3 = I3(II3);
    minlength = min([length(I1),length(I2),length(I3),int32((d1+d2)/2)])-1;
    if minlength >= 1
        I = reshape([I1(1:minlength)';I2(1:minlength)';I3(1:minlength)'],[],1);
        queue = [I;I1(minlength+1:end)];
    else
        Y = Yab;
        Y = reshape(Y, [], 1);
        [~,queue] = sort(Y, 'descend');
    end
    
    t_sort = toc;
    
    trans = reshape(1:(d1+1)*(d2+1),d1+1,d2+1);
    trans = trans(1:d1,1:d2);
    queue1 = reshape(trans(queue),[],1);
    
    left = 1; right = 1;
    k = d1+d2;
    flag = 1;
    count1(testtime) = 0;
    
    tt1 = toc;
    while flag > 1e-10
        right = min([k,length(queue1)]);
        T_1 = union(T_1,queue1(left:right));
        
        disp(['run subproblem ', num2str(count1(testtime)), ' ...'])
        [subCost,subsol,subTime,bas_1] = gurOTsub(a_1,b_1,c_1,T_1,bas_1);
        count1(testtime) = count1(testtime) + 1;
        
        tic;
        sol = subsol{1};
        flag = max([sol(d1+1,1:d2)' ; sol(1:d1,d2+1)]);
        
        k = floor(beta*k);
        left = right + 1;
        tt3 = toc;
        
        tt = tt + tt3 + subTime;
    end
    
    tic;
    T1 = setdiff(T_1,T_0);
    
    trans = [reshape(1:d1*d2,d1,d2) , zeros(d1,1) ; zeros(1,d2) , 0];
    T = trans(T1);
    
    bas = bas_1(1:d1,1:d2);
    tree = reshape(bas,[],1);
    B = A_tilde(:,tree==0);
    c = reshape(M(:,:,testtime),[],1);
    c_B = c(tree == 0);
    
    p = B'\c_B;
    c_reduced = (c'-p'*A_tilde)';
    
    count2(testtime) = 0;
    
    tt2 = toc;
    while 1
        if isempty(find(c_reduced < -1e-6, 1)) || (right >= d1*d2)
            subCost = subCost - 1;
            sol = sol(1:d1,1:d2);
            break
        end
        
        tic;
        temp = find(c_reduced(queue(1:end)) < -1e-6);
        if ~isempty(temp)
            right = min([ max([k,temp(1)]) , length(queue) ]);
        else
            right = min([ k , length(queue) ]);
        end
        T = union(T,queue(left:right));
        tt3 = toc;
        
        if left > right
            k = floor(beta*k);
            continue;
        end
        
        disp(['run subproblem ', num2str(count1(testtime)+count2(testtime)), ' ...'])
        [subCost,subsol,subTime,bas] = gurOTsub(a(:,testtime),b(:,testtime),c,T,bas);
        count2(testtime) = count2(testtime) + 1;
        
        tic;
        sol = subsol{1};
        B = A_tilde(:,sol  > 0);
        c_B = c(sol > 0);
        p = B'\c_B;
        c_reduced = (c'-p'*A_tilde)';
        
        % Verify optimality condition:
        if isempty(find(c_reduced < -1e-6, 1)) || (right == d1*d2)
            tt4 = toc;
            tt = tt + tt3 + tt4 + subTime;
            break
        end
        
        left = right + 1;
        k = floor(beta*k);
        
        tt4 = toc;
        tt = tt + tt3 + tt4 + subTime;
    end
    
    tdimReduction_general = tt + tt1 + tt2
    time(testtime) = tdimReduction_general;
    realCost(testtime) = subCost;
    realOT{int16(testtime)} = sol;
    
end

end

