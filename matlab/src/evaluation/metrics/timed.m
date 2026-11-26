function [t, varargout] = timed(fn, repeat)
% Seconds and outputs of fn(): the mean of `repeat` calls after a first one (repeat = 0: the first call)
if nargin < 2, repeat = 1; end
varargout = cell(1, max(nargout - 1, 0));
tic;
[varargout{:}] = fn();
t = toc;
if repeat
    tic;
    for r = 1:repeat, [varargout{:}] = fn(); end
    t = toc / repeat;
end
end
