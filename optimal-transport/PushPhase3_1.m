function [Bas] = PushPhase3_1(basis,a,b,M)

% PushPhase: push an infeasible basic solution into an FBS.
%   

d1 = size(a,1);
d2 = size(b,1);

c = reshape(M,[],1);
A_true = [-kron(speye(d2),ones(1,d1));kron(ones(1,d2),speye(d1))];
b_true = [-b;a];

tree = reshape(basis,[],1);     
A_tilde = A_true(1:end-1,1:end);
b_tilde = b_true(1:end-1);
B = A_tilde(:,tree==0);          % basis matrix
x_B = B\b_tilde;                 % basic solution (basic part)
x_tree = basis;                  % turn the basic solution into a matrix form
x_tree(basis == 0) = x_B;
x_tree(basis == -1) = 0;       
c_B = c(tree == 0);


iter2 = 0;
aim = find(x_tree < 0);        % find converse flows
for i = 1:length(aim)          % deal with the converse flows one by one
    Ind = aim(i);
    
    % consider the ith converse flow: x_{I1,J1}
    J1 = ceil(Ind/d1);
    I1 = mod(Ind,d1); 
    if I1 == 0
        I1 = d1;
    end
    
    [~,J2] = max(x_tree(I1,:));
    [~,I2] = max(x_tree(:,J1));   
    while x_tree(I1,J1) < 0 
        
        assert((x_tree(I2,J1) > 0) && (x_tree(I1,J2) > 0));
        assert(x_tree(I2,J2) == 0);
        
        % "Irrigation" - try to make the converse flow positive
        [theta,flag] = min([-x_tree(I1,J1) x_tree(I1,J2) x_tree(I2,J1)],[],'linear');
        x_tree(I1,J1) = x_tree(I1,J1) + theta;
        x_tree(I2,J1) = x_tree(I2,J1) - theta;
        x_tree(I1,J2) = x_tree(I1,J2) - theta;
        x_tree(I2,J2) = x_tree(I2,J2) + theta;        
        
        if flag == 2
            [~,J2] = max(x_tree(I1,:));
        elseif flag == 3
            [~,I2] = max(x_tree(:,J1));
        end
        
        iter2 = iter2 + 1;
    end   
end

Bas = x_tree;
Bas(x_tree == 0) = -1;
Bas(x_tree ~= 0) = 0;

% % Test basis:
% assert(length(find(Bas==0)) == d1+d2-1);
% 
% % Test feasibility:
% tree_new = reshape(Bas,[],1);     
% B_new = A_tilde(:,tree_new==0);                          
% x_newtree = Bas;                  
% x_newtree(Bas == 0) = B_new\b_tilde;
% x_newtree(Bas == -1) = 0;       
% converse_flow = find(x_newtree < 0, 1);
% assert(isempty(converse_flow));


disp(['num of converse flow: ', num2str(length(aim))]);
disp(['iternum of pushphase3: ', num2str(iter2)]);

end

 