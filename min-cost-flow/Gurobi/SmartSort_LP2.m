function queue = SmartSort_LP2(A_mat,b,c,u,X)
%% Smart Sort Algorithm
tic;
N = 1;
n = size(b,1);
m = size(c,1);

x_hat = X;
x_hat(X>u/2) = u(X>u/2)-X(X>u/2);
x_hat((X<0)|(X>u)) = 0;

% 构建新问�?: min c_bat ^T x, s.t. A_bar x = b_bar, 0 <= x <= u_bar;
% A_plus  = bsxfun(@max, spalloc(n,m,1),  A_mat);
% A_minus = bsxfun(@max, spalloc(n,m,1), -A_mat);
% a_sum = 1/2*ones(1,n)*(A_plus + A_minus);
% A_bar = A_mat;
% A_bar(:,X<=u/2) =  A_mat(:,X<=u/2)./a_sum(X<=u/2);
% A_bar(:,X>=u/2) = -A_mat(:,X>=u/2)./a_sum(X>=u/2);
A_bar(:,X<=u/2) =  A_mat(:,X<=u/2);
A_bar(:,X>=u/2) = -A_mat(:,X>=u/2);
% c_bar = c;
% c_bar(X<=u/2) =  c(X<=u/2)./a_sum(X<=u/2)';
% c_bar(X>=u/2) = -c(X>=u/2)./a_sum(X>=u/2)';
% x_bar = a_sum'.*x_hat;
x_bar = x_hat;
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
[row,col] = find(A_bar);
val = f_inv(row).*x_bar(col);
a = A_bar(A_bar~=0);
r = sparse(row,col,val.*a,n,m);

% 对r排序，得到边的序列queue�?
r_1 = max(r.*sign(r));

[~,queue1] = sort(r_1,'descend');
[~,queue2] = sort(x_bar,'descend');
queue = reshape([queue1;queue2'],2*m,1);
queue = unique(queue,'stable');

t_sort = toc


end

