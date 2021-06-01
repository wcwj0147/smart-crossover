function [realCost, realOT, time, count] = dimReduction_warm(a,b,M,basis,queue)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明

N = size(a,2);
d1 = size(a,1);
d2 = size(b,1);

A_true = [-kron(speye(d2),ones(1,d1));kron(ones(1,d2),speye(d1))];
A_tilde = A_true(1:end-1,1:end);
realCost = zeros(N,1);
realOT = cell(N,1);
iter = zeros(N,1);
time = zeros(N,1);
count = zeros(N,1);
beta = 2;

for testtime = 1:N
    
    tt = 0;
    tic;
    % initialization:
    c = reshape(M(:,:,testtime),[],1);
    bas = basis(:,:,testtime);
    tree = reshape(bas,[],1);
    B = A_tilde(:,tree==0);
    c_B = c(tree == 0);
    
    % solve reduced costs:
    p = B'\c_B;
    c_reduced = (c'-p'*A_tilde)'; 
    
    T = find(tree == 0);
    
%     % find a sequence:
%     XX = X(:,:,testtime);
%     
%     aa = sum(XX,2);           
%     bb = sum(XX,1)';          
% 
%     Yab = XX./aa./bb';        
%     Ya = XX./aa;              
%     Yb = XX./bb';             
% 
%     Yab  = reshape(Yab, [], 1);           
%     Ya = reshape(Ya, [], 1);
%     Yb  = reshape(Yb, [], 1);
%     thhold1 = 1.9*mean(Yab(1:97:end));
%     thhold2 = 1.99*mean(Ya(1:97:end));
%     thhold3 = 1.99*mean(Yb(1:97:end));
%     I1 = find(Yab);                       
%     I2 = find(Ya>thhold2>thhold1);
%     I3 = find(Yb>thhold3);
%     [~,II1] = sort(Yab(I1) ,'descend');
%     [~,II2] = sort(Ya(I2) ,'descend');
%     [~,II3] = sort(Yb(I3) ,'descend');
%     I1 = I1(II1);
%     I2 = I2(II2);
%     I3 = I3(II3);
%     minlength = min([length(I1),length(I2),length(I3),int32((d1+d2)/2)])-1;
%     if minlength >= 1
%         I = reshape([I1(1:minlength)';I2(1:minlength)';I3(1:minlength)'],[],1);
%         queue = [I;I1(minlength+1:end)];
%     else
%         Y = Yab;
%         Y = reshape(Y, [], 1);
%         [~,queue] = sort(Y, 'descend');
%     end
    
    left = 1; right = 1;
    k = d1+d2;
    count(testtime) = 0;
    
    tt1 = toc;
    
    while 1 
        tic;
        temp = find(c_reduced(queue(left:end)) < -1e-6);
        if ~isempty(temp)
            right = min([ max([k,temp(1)]) , length(queue) ]);
        else
            right = min([ k , length(queue) ]);
        end
        T = union(T,queue(left:right));  
        tt2 = toc; 
        tt = tt + tt2;
        
        if left > right
            k = floor(beta*k);
            continue;
        end
        
        disp(['run subproblem ', num2str(count(testtime)), ' ...'])
        [subCost,subsol,subTime,bas] = gurOTsub(a(:,testtime),b(:,testtime),c,T,bas);
        count(testtime) = count(testtime) + 1;
        
        tic;
        
        sol = subsol{1};
        B = A_tilde(:,sol > 0);
        c_B = c(sol > 0);
        p = B'\c_B;
        c_reduced = (c'-p'*A_tilde)';

        % Verify optimality condition:
        if isempty(find(c_reduced < -1e-6, 1)) || (right == d1*d2)
            tt3 = toc;
            tt = tt + tt3 + subTime;
            break
        end
        
        left = min([right+1,length(queue)]);
        k = floor(beta*k);   
        
        tt3 = toc;
        tt = tt + tt3 + subTime;
    end
    
    tdimReduction_warm = tt1 + tt
    time(testtime) = tdimReduction_warm;
    realCost(testtime) = subCost;
    realOT{int16(testtime)} = sol;
    
end

end

