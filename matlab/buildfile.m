function plan = buildfile
% Build plan of the MATLAB front-end (MATLAB >= R2023b), from matlab/:
%   buildtool mex     compile the MEX gateway kp_mex from the C++ core (cpp/) into build/
%   buildtool test    the tests against dense Gaussian process algebra (after mex)
plan = buildplan(localfunctions);
plan("test").Dependencies = "mex";
plan.DefaultTasks = "test";
end

function mexTask(~)
% Compile kp_mex from the C++ core into build/ (MATLAB >= R2023b, resources/mpackage.json)
assert(~isMATLABReleaseOlderThan("R2023b"), "kernel_packets needs MATLAB R2023b or newer");
addpath(genpath(fullfile(fileparts(mfilename('fullpath')), 'src')));
kp_build;
end

function testTask(~)
% Run matlab/tests from the repository root: matlab/ as the current folder is the root of the package of
% resources/mpackage.json on the path without being installed, which MATLAB warns about
here = fileparts(mfilename('fullpath'));
old = cd(fileparts(here));
back = onCleanup(@() cd(old));
assertSuccess(runtests(fullfile(here, 'tests')));
end
