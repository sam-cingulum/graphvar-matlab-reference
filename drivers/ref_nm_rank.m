function ref_nm_rank(bctDir, gvDir, inDir, outDir)
% Signed null-model rank-failure probe on genuine MATLAB R2026b.
% Inputs: inDir/nm_rank.json (tools/gen_nm_rank.py). Writes outDir/nm_rank.jsonl.
% BCT 2019-03-03 takes precedence over GraphVar 2.03a's older ext/BCT copy.
% Each call resets rng(seed, 'twister'), runs
%   [W0,R] = null_model_und_sign(W, bin_swaps, wei_freq)
% and immediately after success or a caught exception records after = rand(1,4).
% Successful calls record W0 and both strength-correlation outputs R; caught calls
% record the exception's exact identifier and message. Every numeric output is
% emitted as size plus column-major binary64 words.
p = jsondecode(fileread(fullfile(inDir, 'nm_rank.json')));
if ~isfolder(outDir), mkdir(outDir); end
out = fullfile(outDir, 'nm_rank.jsonl');
if isfile(out), delete(out); end

src = fullfile(gvDir, 'src');
addpath(fullfile(src, 'ext', 'Network Community Toolbox'));
addpath(fullfile(src, 'gui', 'helper'));
addpath(fullfile(src, 'ext', 'BCT'), '-end');
addpath(bctDir);

rec = newrec('nm-rank-env', 'env');
rec.version = version;
rec.notes.which_null_model_und_sign = clean(which('null_model_und_sign'), bctDir, gvDir);
rec.notes.which_randmio_und_signed = clean(which('randmio_und_signed'), bctDir, gvDir);
rec.notes.nargin_randperm = nargin('randperm');
emit(out, rec);

G = struct();
for g = cellify(p.graphs)
    g = g{1};
    G.(key(g.name)) = reshape(hex2num(cellstr(g.W)), g.n, g.n);
end

for c = cellify(p.calls)
    c = c{1};
    rec = newrec(c.case, 'null_model_und_sign');
    rec.graph = c.graph;
    rec.seed = c.seed;
    rec.bin_swaps = c.bin_swaps;
    rec.wei_freq = c.wei_freq;
    W = G.(key(c.graph));
    rng(c.seed, 'twister');
    try
        [W0, R] = null_model_und_sign(W, c.bin_swaps, c.wei_freq);
        rec.outputs.W0 = W0;
        rec.outputs.R = R;
        rec.notes.completed = 1;
    catch err
        rec.error = struct('identifier', err.identifier, 'message', err.message);
        rec.notes.completed = 0;
    end
    % This must be the next random operation after either return or catch.
    rec.outputs.after = rand(1, 4);
    emit(out, rec);
end
end

function s = clean(w, bctDir, gvDir)
s = strrep(strrep(w, bctDir, '<bct>'), gvDir, '<gv>');
end

function k = key(name)
k = ['g_' strrep(strrep(name, '-', '_'), '.', 'p')];
end

function c = cellify(s)
if iscell(s), c = s(:)'; else, c = num2cell(s(:)'); end
end

function rec = newrec(caseId, part)
rec = struct('case', caseId, 'part', part, 'outputs', struct(), 'notes', struct());
end

function emit(out, rec)
f = fieldnames(rec.outputs);
for i = 1:numel(f)
    v = rec.outputs.(f{i});
    if ischar(v) || isstruct(v) || iscell(v), continue, end
    v = double(v);
    if isempty(v)
        rec.outputs.(f{i}) = struct('size', size(v), 'hex', {{}});
    else
        rec.outputs.(f{i}) = struct('size', size(v), 'hex', {cellstr(num2hex(v(:)))'});
    end
end
fid = fopen(out, 'a');
fprintf(fid, '%s\n', jsonencode(rec));
fclose(fid);
end
