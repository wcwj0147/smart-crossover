%% A module getting standard form LP from general ones.
%
%    A general form LP model:
%             min      c^T x
%             s.t.     A x (sense) b
%                      l <= x <= u
%
%    sense: '=' or '<' or '>'.
%
%    A standard form LP model:
%             min      c^T x
%             s.t.     A x = b
%                   0 <= x <= u


%% Main procedure: get standard LP for each model.
lp_data_path = fullfile(pwd, 'all');
lp_files = dir(fullfile(lp_data_path, '*.mps'));

f = waitbar(0, 'Starting');
K = length(lp_files);
for k = 19:K
    waitbar(k/K, f, sprintf('Progress: %d %%, %s', floor(k/K*100), lp_files(k).name));
    m = gurobi_read(fullfile(lp_data_path, lp_files(k).name));
    results = gurobi(gurobi_relax(m));
    if isfield(m, 'modelname')
        m_std.modelname = m.modelname;
    end
    assert(~isfield(m, 'modelsense'), "Model is to maximize.")
    [A, b, c, u, cost_change] = to_standard_form(m.A, m.rhs, m.obj, m.lb, m.ub, m.sense);
    m_std.A     = A;
    m_std.rhs   = b;
    m_std.obj   = c;
    m_std.ub    = u;
    m_std.lb    = zeros(length(c), 1);
    m_std.sense = '=';
    params.Method = 2;
    results_std = gurobi(m_std, params);
    assert(abs(results.objval-results_std.objval-cost_change) < 1e-6, "Transfermation failed!")
    gurobi_write(m_std, fullfile(pwd, 'standard', lp_files(k).name))
end
waitbar(1, f, 'Done!');
close(f)


%% Main function: transfer a general form LP to standard.
function [A_std, b_std, c_std, u_std, cost_change] = to_standard_form(A, b, c, l, u, sense)

% Copy input matrices
A_std = A;
b_std = b;
c_std = c;
u_std = u;

[m, n] = size(A_std);

% Identify '<' and '>' constraints
le_constr_ind = find(sense == '<');
le_num = length(le_constr_ind);
ge_constr_ind = find(sense == '>');
ge_num = length(ge_constr_ind);

% Add slack variables to '<' constraints and surplus variables to '>' constraints
i = [le_constr_ind; ge_constr_ind];
j = 1:le_num + ge_num;
v = [ones(le_num, 1); -ones(ge_num, 1)];
slack_surplus = sparse(i, j, v, m, le_num + ge_num);
A_std = [A_std, slack_surplus];

% Update the upper bounds and objective function for the added slack and surplus variables
u_std = [u_std; inf(le_num + ge_num, 1)];
c_std = [c_std; zeros(le_num + ge_num, 1)];

% Identify variables with l = -inf and u = inf, and l = -inf and u < inf
free_vars = (l == -inf) & (u == inf);
down_free_vars = (l == -inf) & (u < inf);
up_free_vars = (l > -inf) & (u == inf);
bounded_vars = (l > -inf) & (u < inf);

% For free variables
if any(free_vars)
    num_free_vars = sum(free_vars);
    A_std = [A_std, -A_std(:, free_vars)];
    u_std = [u_std; inf(num_free_vars, 1)];
    u_std(free_vars) = inf;
    c_std = [c_std; -c_std(free_vars)];
end

% Replace variables with l = -inf and u < inf with u-x
if any(down_free_vars)
    A_std(:, down_free_vars) = -A_std(:, down_free_vars);
    b_std = b_std - A_std(:, down_free_vars) * u(down_free_vars);
    c_std(down_free_vars) = -c_std(down_free_vars);
    u_std(down_free_vars) = inf;
end

% Replace variables with l > -inf with x-l
if any(bounded_vars)
    b_std = b_std - A_std(:, bounded_vars) * l(bounded_vars);
    u_std(bounded_vars) = u_std(bounded_vars) - l(bounded_vars);
end
if any(up_free_vars)
    b_std = b_std - A_std(:, up_free_vars) * l(up_free_vars);
end

cost_change = 0;
if any(bounded_vars | up_free_vars)
    cost_change = cost_change + c_std(bounded_vars | up_free_vars)' * l(bounded_vars | up_free_vars);
end
if any(down_free_vars)
    cost_change = cost_change + c_std(down_free_vars)' * u(down_free_vars);
end

end
