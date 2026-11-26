function rows = read_csv(path, part)
% Rows of one part of the experiment's table as a struct array
T = readtable(path, 'TextType', 'string', 'Delimiter', ',');
T = T(T.part == string(part), :);
rows = table2struct(T);
end
