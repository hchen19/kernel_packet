function kp_build
% Compile kp_mex from the C++ core (cpp/) into build/ (when missing or older than a source) and add build/ to the path
here = fileparts(mfilename('fullpath'));
root = fileparts(fileparts(fileparts(fileparts(here))));
out = fullfile(root, 'build');
cpp = [dir(fullfile(root, 'cpp', '*.cpp')); dir(fullfile(root, 'cpp', 'include', '**', '*.hpp')); dir(fullfile(root, 'cpp', 'toolchain.txt'))];
src = [{fullfile(here, 'kp_mex.cpp')}, fullfile({cpp.folder}, {cpp.name})];
bin = dir(fullfile(out, ['kp_mex.' mexext]));
if isempty(bin) || bin.datenum < max(cellfun(@(f) getfield(dir(f), 'datenum'), src))
    % compiler and flags of cpp/toolchain.txt (pinned for identical binaries on every machine)
    tc = splitlines(string(fileread(fullfile(root, 'cpp', 'toolchain.txt'))));
    cxx = split(tc(1));
    local_cxx = fullfile(root, '.gcc', 'env', 'bin', cxx(1));  % the pinned g++ in .gcc/ (scripts/setup_env.sh)
    if isfile(local_cxx), cxx(1) = local_cxx; end
    [~, found] = system(cxx(1) + " -dumpfullversion");
    assert(strtrim(found) == cxx(2) || ~isempty(getenv('KP_ALLOW_ANY_CXX')), ...
        "cpp/toolchain.txt pins %s %s, found %s", cxx(1), cxx(2), strtrim(found));
    if ~isfolder(out), mkdir(out); end
    mex('-R2018a', '-O', ['-I' fullfile(root, 'cpp', 'include')], "CXX=" + cxx(1), "CXXFLAGS=$CXXFLAGS " + tc(2), ...
        'LDFLAGS=$LDFLAGS -pthread', '-outdir', out, src{endsWith(src, '.cpp')});
end
addpath(out);
end
