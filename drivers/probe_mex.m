function probe_mex(gvRoot, logFile, mode, label)
% Phase 0 probe: do GraphVar 2.03a's bundled Linux MEX binaries load and run
% on this MATLAB release? Synthetic inputs only.
%
% Crash and hang tolerant: each item logs START before it runs and RESULT
% after. A watchdog kills MATLAB if one call takes over 150 s. The workflow
% calls 'resume' again in a fresh session until the log ends.
src = fullfile(gvRoot, 'src');
addpath(genpath(src));

items = probe_items(src);
switch mode
    case 'list'
        fid = fopen(logFile, 'w');
        % smoke tests first, then the bare load checks
        order = [find(cellfun(@(t) strcmp(t.kind, 'smoke'), items)), find(cellfun(@(t) strcmp(t.kind, 'load'), items))];
        for i = order
            fprintf(fid, '%s\n', items{i}.label);
        end
        fclose(fid);
    case 'env'
        env = struct('release', version(), 'computer', computer(), ...
                     'maxNumCompThreads', maxNumCompThreads(), ...
                     'blas', version('-blas'), 'lapack', version('-lapack'));
        names = {'tdot', 'tlstsq', 'tpinv', 'trandlstsq'};
        for i = 1:numel(names)
            env.precedence.(names{i}) = strrep(which(names{i}, '-all'), [src filesep], '');
        end
        write_line(logFile, struct('kind', 'RESULT', 'item', 'env', 'data', env));
    case 'resume'
        % Run the items in list order. An item that has a START line but no
        % RESULT line crashed or hung the previous MATLAB session.
        order = [find(cellfun(@(t) strcmp(t.kind, 'smoke'), items)), find(cellfun(@(t) strcmp(t.kind, 'load'), items))];
        [done, started] = read_log(logFile);
        if ~isKey(done, 'env')
            env = struct('release', version(), 'computer', computer(), ...
                         'maxNumCompThreads', maxNumCompThreads(), ...
                         'blas', version('-blas'), 'lapack', version('-lapack'));
            names = {'tdot', 'tlstsq', 'tpinv', 'trandlstsq'};
            for i = 1:numel(names)
                env.precedence.(names{i}) = strrep(which(names{i}, '-all'), [src filesep], '');
            end
            write_line(logFile, struct('kind', 'RESULT', 'item', 'env', 'data', env));
        end
        pid = feature('getpid');
        for i = order
            it = items{i};
            if isKey(done, it.label)
                continue
            end
            if isKey(started, it.label)
                write_line(logFile, struct('kind', 'RESULT', 'item', it.label, ...
                    'data', struct('status', 'CRASHED_OR_HUNG', 'note', 'MATLAB session ended during this call (crash, or killed by the 150 s watchdog)')));
                continue
            end
            write_line(logFile, struct('kind', 'START', 'item', it.label));
            [~, w] = system(sprintf('(sleep 150 && kill -9 %d) > /dev/null 2>&1 & echo $!', pid));
            r = run_item(it);
            system(sprintf('kill %s > /dev/null 2>&1', strtrim(w)));
            write_line(logFile, struct('kind', 'RESULT', 'item', it.label, 'data', r));
        end
        write_line(logFile, struct('kind', 'END', 'item', ''));
end
end

function items = probe_items(src)
items = {};
mexFiles = dir(fullfile(src, '**', ['*.' mexext]));
for i = 1:numel(mexFiles)
    [~, name] = fileparts(mexFiles(i).name);
    rel = strrep(fullfile(mexFiles(i).folder, mexFiles(i).name), [src filesep], '');
    items{end+1} = struct('label', ['load:' rel], 'kind', 'load', ...
        'name', name, 'folder', mexFiles(i).folder, 'fn', []); %#ok<AGROW>
end
rng(20261126, 'twister');
X = randn(40, 6);
y = X * [1; -0.5; 0; 0.25; 0; 2] + 0.1 * randn(40, 1);
lab = double(y > median(y)) + 1;
W = rand(12); W = (W + W') / 2; W(1:13:end) = 0;
S = cov(X);
B = W - sum(W, 2) * sum(W, 1) / sum(W(:));
smoke = {
    'smoke:glasso', 2, @() glasso(size(S, 2), S, 0, repmat(0.1, size(S, 2)), 0, 0, 0, 1, 1e-4, 1e4, zeros(size(S)), zeros(size(S)))
    'smoke:glmnet_gaussian', 1, @() glmnetPredict(glmnet(X, y, 'gaussian', struct('alpha', 0.5, 'nlambda', 1, 'lambda', 0.01)), [], 0.01, 'coefficients')
    'smoke:glmnet_binomial', 1, @() glmnetPredict(glmnet(X, lab, 'binomial', struct('alpha', 0.5, 'nlambda', 1, 'lambda', 0.01)), X, 0.01, 'response', 0)
    'smoke:libsvm_csvc', 1, @() svmrun(lab, X, '-s 0 -t 0 -c 1 -b 0 -q', '-q -b 0')
    'smoke:libsvm_csvc_prob', 1, @() svmrun(lab, X, '-s 0 -t 0 -c 1 -b 1 -q', '-q -b 1')
    'smoke:libsvm_nusvr', 1, @() svmrun(y, X, '-s 4 -t 0 -n 0.5 -b 0 -q', '-q -b 0')
    'smoke:c_null_model_und_sign', 1, @() c_null_model_und_sign(W, 5)
    'smoke:genlouvain', 1, @() genlouvain(B, 10000, 0)
    'smoke:tdot', 1, @() tdot(X', X)
    'smoke:tlstsq', 2, @() tlstsq(X, y)
    'smoke:tpinv', 1, @() tpinv(X' * X)
    'smoke:SLEP_LeastR', 1, @() LeastR(X, y, 0.1, struct('rFlag', 1, 'mFlag', 0, 'lFlag', 0, 'init', 2, 'tFlag', 5, 'maxIter', 1000))
    };
for i = 1:size(smoke, 1)
    items{end+1} = struct('label', smoke{i, 1}, 'kind', 'smoke', 'name', '', ...
        'folder', '', 'fn', smoke{i, 3}, 'nout', smoke{i, 2}); %#ok<AGROW>
end
end

function r = run_item(it)
r = struct('status', '', 'identifier', '', 'message', '', 'resolved', '', 'outputs', {{}});
if strcmp(it.kind, 'load')
    r.resolved = which(it.name);
    prev = cd(it.folder);
    try
        feval(it.name);
        r.status = 'LOADED_RAN';
    catch err
        r.identifier = err.identifier;
        r.message = err.message;
        if contains(err.identifier, 'invalidMEXFile', 'IgnoreCase', true) ...
                || contains(err.message, 'Invalid MEX-file') ...
                || contains(err.message, 'shared object') || contains(err.message, 'GLIBC')
            r.status = 'LOAD_FAILED';
        else
            r.status = 'LOADED_ERRORED';
        end
    end
    cd(prev);
    return
end
try
    out = cell(1, it.nout);
    [out{:}] = it.fn();
    for k = 1:it.nout
        v = out{k};
        if isnumeric(v) || islogical(v)
            v = double(v);
            r.outputs{end+1} = struct('size', size(v), 'hex', {cellstr(num2hex(v(:)))'});
        end
    end
    r.status = 'OK';
catch err
    r.status = 'ERROR';
    r.identifier = err.identifier;
    r.message = err.message;
end
end

function p = svmrun(lab, X, trainOpts, predictOpts)
model = svmtrain(lab, X, trainOpts);
[~, ~, p] = svmpredict(lab, X, model, predictOpts);
end

function [done, started] = read_log(logFile)
done = containers.Map();
started = containers.Map();
if ~isfile(logFile)
    return
end
lines = readlines(logFile);
for i = 1:numel(lines)
    if strlength(lines(i)) == 0
        continue
    end
    e = jsondecode(lines(i));
    if strcmp(e.kind, 'START')
        started(e.item) = true;
    elseif strcmp(e.kind, 'RESULT')
        done(e.item) = true;
    end
end
end

function write_line(logFile, s)
fid = fopen(logFile, 'a');
fprintf(fid, '%s\n', jsonencode(s));
fclose(fid);
end
