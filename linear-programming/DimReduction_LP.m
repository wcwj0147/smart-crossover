function [realCost, realOF, time, count1] = DimReduction_LP(A_mat,b,c,l,u,X)
%% Smart Sort Algorithm
tic;
N = 1;
n = size(b,1);
m = size(c,1);

Indup = find(X>u/2);
x_hat = X;
x_hat(Indup) = u(Indup)-X(Indup);
x_hat((X<0)|(X>u)) = 0;

% 构建新问题: min c_bat ^T x, s.t. A_bar x = b_bar, 0 <= x <= u_bar;
A_plus  = bsxfun(@max, spalloc(n,m,1),  A_mat);
A_minus = bsxfun(@max, spalloc(n,m,1), -A_mat);
a_sum = 1/2*ones(1,n)*(A_plus + A_minus);
A_bar = A_mat;
A_bar(:,X<=u/2) =  A_mat(:,X<=u/2)./a_sum(X<=u/2);
A_bar(:,X>=u/2) = -A_mat(:,X>=u/2)./a_sum(X>=u/2);
c_bar = c;
c_bar(X<=u/2) =  c(X<=u/2)./a_sum(X<=u/2)';
c_bar(X>=u/2) = -c(X>=u/2)./a_sum(X>=u/2)';
x_bar = a_sum'.*x_hat;
b_bar = A_bar*x_bar;
u_bar = a_sum'.*u;
A_barplus  = bsxfun(@max, spalloc(n,m,1),  A_bar);
A_barminus = bsxfun(@max, spalloc(n,m,1), -A_bar);

% 求f_i（源点的流出量或汇点的流入量）：
f_1 = A_barplus*x_bar;
f_2 = A_barminus*x_bar;
f = bsxfun(@max, f_1, f_2);
f_inv = 1./f;

% 求排序依赖的指标r_{ij}（考虑流量占比）：
[row,col] = find(A_bar);
val = f_inv(row).*x_bar(col);
a = A_bar(A_bar~=0);
r = sparse(row,col,val.*a,n,m);

% 对r排序，得到边的序列queue：
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
K = 1e5*m*max(c.*sign(c));
beta = 2;
t_2 = 0; t_3 = 0;

for testtime = 1:N

    tic;
    % 构造新问题LP_1：
    b_1 = b;
    b_true = b - A_mat(:,Indup)*u(Indup);
    sig = sign(b_true); sig(sig == 0) = 1;
    c_1 = [c ; K*ones(n,1).*sig];
    ll = zeros(n,1);  uu = Inf*ones(n,1);
    ll(sig<0) = -Inf; uu(sig<0) = 0;
    l_1 = [l ; ll];
    u_1 = [u ; uu];
    
    bas_1 = [-ones(m,1) ; zeros(n,1)];
    bas_1(Indup) = 1;
    T_1 = find(bas_1 == 0);

    A_mat_1 = [A_mat, speye(n)];
    
    left = 1;
    k = max(2,floor(m/(2*n)))*n;
    flag = true;
    count1(testtime) = 0;
    t_1 = toc;
    
    while flag
        
        if left > length(queue)
            disp(' ');
            fprintf(' ##### Dimension reduction algorithm fails! #####');
            disp(' ');
            break;
        end
        
        tic;
        right = min([k,length(queue)]);
        T_1 = union(T_1,queue(left:right));
        t_2 = t_2 + toc;
        
        [~,subOF,subTime,bas_1] = CplexLPSub(A_mat_1,b_1,c_1,l_1,u_1,bas_1,T_1);
        count1(testtime) = count1(testtime) + 1;
        
        tic;
        % 解决基的退化问题：
        B = A_mat_1(:,bas_1==0);
        s = size(B,2);
        [L,U,P] = lu(B);
        if s < n
            row1 = s+1:n;
            col1 = 1:n-s;
            val1 = ones(1,n-s);
            L = [L , sparse(row1,col1,val1,n,n-s)];
            U = [[U ; zeros(n-s,s)] , sparse(row1,col1,val1,n,n-s)];
            B = P'*L*U;
            [row2,~,~] = find(B(:,s+1:n));
            bas_1(row2 + m) = 0;
        end    
        
        sol = zeros(m+n,1);
        sol(T_1) = subOF{1};
        sol(bas_1==1) = u_1(bas_1==1);
        c_B = c_1(bas_1==0);
        p1 = U'\c_B; p2 = L'\p1; p = P\p2;
        r = c_1 - A_mat_1'*p;   % reduced cost

        if (max(sol(m+1:m+n).*sign(sol(m+1:m+n)))<1e-10/K) && (min([r(bas_1==-1);-r(bas_1==1)])>-1e-6)
            flag = false;
        end    
        k = floor(beta*k);
        left = right + 1;
        t_3 = t_3 + subTime + toc;
    end
    
    % 将找到的基放回原问题里：
    % [MyCost, MyOF, t_f] = CplexLPFinal(A_mat_1,b_1,c_1,l_1,u_1,bas_1);

    tdimReduction = t_1 + t_2 + t_3
    time(testtime) = tdimReduction + t_sort;
    Indup = find(bas_1==1); 
    T_rest = setdiff(1:m+n, T_1);
    Indup_rest = intersect(Indup,T_rest);
    realCost(testtime) = c_1'*sol;
    realOF{int16(testtime)} = sol;
    
end

end

