function [realCost, realOF, time, count1] = DimReduction_GurobiLP3(A_mat,b,c,l,u,X)
%% Smart Sort Algorithm
tic;
N = 1;
n = size(b,1);
m = size(c,1);

Indup = find(X>u/2);
x_hat = X;
x_hat(Indup) = u(Indup)-X(Indup);
x_hat((X<0)|(X>u)) = 0;

% 构建新问�?: min c_bat ^T x, s.t. A_bar x = b_bar, 0 <= x <= u_bar;
A_plus  = bsxfun(@max, spalloc(n,m,1),  A_mat);
A_minus = bsxfun(@max, spalloc(n,m,1), -A_mat);
a_sum = (1/2*ones(1,n)*(A_plus + A_minus)).';
% A_bar = A_mat;
% A_bar(:,X<=u/2) =  A_mat(:,X<=u/2)./a_sum(X<=u/2);
% A_bar(:,X>=u/2) = -A_mat(:,X>=u/2)./a_sum(X>=u/2);
[row,col,a] = find(A_mat);
val = a./a_sum(col);
A_bar = sparse(row,col,val,n,m);
A_bar(:,X>u/2) = -A_bar(:,X>u/2);
% c_bar = c;
% c_bar(X<=u/2) =  c(X<=u/2)./a_sum(X<=u/2)';
% c_bar(X>=u/2) = -c(X>=u/2)./a_sum(X>=u/2)';
x_bar = a_sum.*x_hat;
% b_bar = A_bar*x_bar;
% u_bar = a_sum'.*u;
A_barplus  = bsxfun(@max, spalloc(n,m,1),  A_bar);
A_barminus = bsxfun(@max, spalloc(n,m,1), -A_bar);

% 求f_i（源点的流出量或汇点的流入量）：
f_1 = A_barplus*x_bar;
f_2 = A_barminus*x_bar;
f = bsxfun(@max, f_1, f_2);
f_inv = 1./f;

% 求排序依赖的指标r_{ij}（�?�虑流量占比）：
[row,col,a] = find(A_bar);
val = f_inv(row).*x_bar(col);
r = sparse(row,col,val.*a,n,m);

% 对r排序，得到边的序列queue�?
% r_1 = max(r.*sign(r));
% [~,queue] = sort(r_1,'descend');

r_1 = max(r.*sign(r));
[~,queue1] = sort(r_1,'descend');
[~,queue2] = sort(x_bar,'descend');
queue = reshape([queue1;queue2'],2*m,1);
queue = unique(queue,'stable');

t_sort = toc

%% Dimension Reduction Algorithm
realCost = zeros(N,1);
realOF = cell(N,1);
time = zeros(N,1);
count1 = zeros(N,1);

beta = 2;
t_2 = 0; t_3 = 0;

for testtime = 1:N

    tic;

    left = 1;
    k = floor(2*n);
    %k = max(2,floor(m/(3*n)))*n;
    flag = true;
    count1(testtime) = 0;
    
    right = min([k,length(queue)]);
    T = queue(left:right);
    bas = -ones(m,1);
    bas(Indup) = -2;
    t_1 = toc;
    
    [~,~,~,time_first,bas] = GurobiLPSub(A_mat,b,c,l,u,bas,T);
    t_1 = t_1 + time_first;
    
    while flag
        
        if left > length(queue)
            disp(' ');
            fprintf(' ##### Dimension reduction algorithm fails! #####');
            disp(' ');
            break;
        end
        
        tic;
        right = min([k,length(queue)]);
        T = union(T,queue(left:right));
        t_2 = t_2 + toc;
        
        [~,subOF,~,subTime,bas] = GurobiLPSub(A_mat,b,c,l,u,bas,T);
        count1(testtime) = count1(testtime) + 1;
        
        tic;
        % 解决基的�?化问题：
%         B = A_mat_1(:,bas_1==0);
%         s = size(B,2);
%         [L,U,P] = lu(B);
%         if s < n
%             row1 = s+1:n;
%             col1 = 1:n-s;
%             val1 = ones(1,n-s);
%             L = [L , sparse(row1,col1,val1,n,n-s)];
%             U = [[U ; zeros(n-s,s)] , sparse(row1,col1,val1,n,n-s)];
%             B = P'*L*U;
%             [row2,~,~] = find(B(:,s+1:n));
%             bas_1(row2 + m) = 0;
%         end    
        
        B = A_mat(:,bas==0);
        sol = zeros(m,1);
        sol(T) = subOF{1};
        sol(bas==-2) = u(bas==-2);
        c_B = c(bas==0);
%         p1 = U'\c_B; p2 = L'\p1; p = P\p2;
        p = B'\c_B;
        r = c - A_mat'*p;   % reduced cost
        
        if min([r(bas==-1);-r(bas==-2)])>-1e-6
            flag = false;
        end 
        
        r(bas==-2) = -r(bas==-2);
        temp = r(queue);
        loc = find(temp<0 , 1);
        k = max(floor(beta*k), loc);
        left = right + 1;
        t_3 = t_3 + subTime + toc;
    end
    
    % [~, MyOF, t_f] = GurobiLPFinal(A_mat,b,c,l,u,bas);

    tdimReduction = t_1 + t_2 + t_3
    time(testtime) = tdimReduction + t_sort;
    realCost(testtime) = c'*sol;
    realOF{int16(testtime)} = sol;
    
end

end

