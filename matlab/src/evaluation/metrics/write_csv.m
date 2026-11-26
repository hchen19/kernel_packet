function write_csv(rows, path, part)
% Rows (struct array) of one experiment part into the experiment's single table at path: a part column, the union of
% the columns; earlier rows of the same part are replaced. Printed too.
T = convertvars(struct2table(rows, 'AsArray', true), @iscellstr, 'string');
T = addvars(T, repmat(string(part), height(T), 1), 'Before', 1, 'NewVariableNames', 'part');
if isfile(path)
    old = readtable(path, 'TextType', 'string', 'Delimiter', ',');
    old = old(old.part ~= part, :);
    [old, T] = deal(add_missing(old, T), add_missing(T, old));
    T = [old; T(:, old.Properties.VariableNames)];
end
writetable(T, path);
disp(T(T.part == part, :));
end

function A = add_missing(A, B)
% columns of B missing in A, as NaN or missing strings
for v = setdiff(B.Properties.VariableNames, A.Properties.VariableNames, 'stable')
    if isnumeric(B.(v{1})), A.(v{1}) = nan(height(A), 1); else, A.(v{1}) = strings(height(A), 1) + missing; end
end
end
