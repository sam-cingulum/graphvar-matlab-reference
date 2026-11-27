function ref_nm(bctDir, gvDir, inDir, outDir)
% Null-model battery. Inputs: inDir/nm.json (tools/gen_nm.py). BCT 2019-03-03 functions come from bctDir;
% GraphVar 2.03a's own helpers (smallworldness_bu/wu, clusterMean_*, charpath_*) and its
% small_world_propensity (Network Community Toolbox copy) come from gvDir. Writes outDir/nm.jsonl.
% Record kinds (field "part"):
%   env         exist() of the functions small_world_propensity needs, nargin('randperm')
%   rand        rng(seed, 'twister'); rand(1, count)
%   randperm_k  rng(seed, 'twister'); p = randperm(n, k); then after = rand(1, 4)
%   randperm    rng(seed, 'twister'); p = randperm(n); then after = rand(1, 4)
%   randi       rng(seed, 'twister'); count scalar calls randi([lo, hi]); then the vector form
%               randi([lo, hi], 1, count) on a fresh rng(seed); after = rand(1, 4) following each
%   call        rng(seed, 'twister'); the function on graph W; every output; after = rand(1, 4)
%               small_world_propensity also records its intermediates (see swp_parts)
%               smallworldness: RW = nr calls of randmio_und(W, iter) on one stream, then
%               smallworldness_bu (binary W) or smallworldness_wu (weighted W) of W and RW
p = jsondecode(fileread(fullfile(inDir, 'nm.json')));
if ~isfolder(outDir), mkdir(outDir); end
out = fullfile(outDir, 'nm.jsonl');
if isfile(out), delete(out); end

% GraphVar helpers first, then BCT 2019 on top, so BCT functions resolve to 2019.
src = fullfile(gvDir, 'src');
addpath(fullfile(src, 'ext', 'Network Community Toolbox'));
addpath(fullfile(src, 'gui', 'helper'));
addpath(bctDir);
% small_world_propensity's local functions, one file each, so their intermediates can be recorded
swpDir = fullfile(outDir, 'swp_local');
split_local(fullfile(src, 'ext', 'Network Community Toolbox', 'small_world_propensity.m'), swpDir);
addpath(swpDir, '-end');

rec = newrec('nm-env', 'env');
for f = {'graphallshortestpaths', 'nanmean', 'randmio_und', 'randomizer_bin_und', 'smallworldness_bu', ...
         'small_world_propensity', 'regular_matrix_generator', 'randmio_dir_signed'}
    rec.outputs.(['exist_' f{1}]) = exist(f{1}); %#ok<EXIST>
    w = which(f{1});
    rec.outputs.(['which_' f{1}]) = strrep(strrep(w, bctDir, '<bct>'), gvDir, '<gv>');
end
rec.outputs.nargin_randperm = nargin('randperm');
emit(out, rec);

for c = cellify(p.rng.rand)
    c = c{1};
    rec = newrec(sprintf('nm-rand-s%d', c.seed), 'rand');
    rng(c.seed, 'twister');
    rec.outputs.u = rand(1, c.count);
    emit(out, rec);
end
for c = cellify(p.rng.randperm_k)
    c = c{1};
    rec = newrec(sprintf('nm-randperm-n%d-k%d-s%d', c.n, c.k, c.seed), 'randperm_k');
    rec.n = c.n; rec.k = c.k; rec.seed = c.seed;
    rng(c.seed, 'twister');
    rec.outputs.p = randperm(c.n, c.k);
    rec.outputs.after = rand(1, 4);
    emit(out, rec);
end
for c = cellify(p.rng.randperm)
    c = c{1};
    rec = newrec(sprintf('nm-randperm-n%d-s%d', c.n, c.seed), 'randperm');
    rec.n = c.n; rec.seed = c.seed;
    rng(c.seed, 'twister');
    rec.outputs.p = randperm(c.n);
    rec.outputs.after = rand(1, 4);
    emit(out, rec);
end
for c = cellify(p.rng.randi)
    c = c{1};
    rec = newrec(sprintf('nm-randi-%d-%d-s%d', c.lo, c.hi, c.seed), 'randi');
    rec.lo = c.lo; rec.hi = c.hi; rec.seed = c.seed;
    rng(c.seed, 'twister');
    v = zeros(1, c.count);
    for t = 1:c.count, v(t) = randi([c.lo, c.hi]); end
    rec.outputs.scalar = v;
    rec.outputs.after_scalar = rand(1, 4);
    rng(c.seed, 'twister');
    rec.outputs.vector = randi([c.lo, c.hi], 1, c.count);
    rec.outputs.after_vector = rand(1, 4);
    emit(out, rec);
end

G = struct();
for g = cellify(p.graphs)
    g = g{1};
    W = reshape(hex2num(cellstr(g.W)), g.n, g.n);
    G.(strrep(strrep(g.name, '-', '_'), '.', 'p')) = W;
end
for c = cellify(p.calls)
    c = c{1};
    rec = newrec(c.case, 'call');
    rec.fn = c.fn; rec.graph = c.graph; rec.seed = c.seed;
    args = c.args;
    if isempty(args), args = {}; elseif ~iscell(args), args = num2cell(args(:)'); end
    rec.args = args;
    W = G.(strrep(strrep(c.graph, '-', '_'), '.', 'p'));
    try
        rng(c.seed, 'twister');
        switch c.fn
            case {'randmio_und', 'randmio_und_connected', 'randmio_dir', 'randmio_dir_connected', ...
                  'randmio_und_signed', 'randmio_dir_signed'}
                [R, eff] = feval(c.fn, W, args{:});
                rec.outputs.R = R; rec.outputs.eff = eff;
            case 'randomizer_bin_und'
                rec.outputs.R = randomizer_bin_und(W, args{:});
            case {'null_model_und_sign', 'null_model_dir_sign'}
                [W0, Rc] = feval(c.fn, W, args{:});
                rec.outputs.W0 = W0; rec.outputs.R = Rc;
            case 'small_world_propensity'
                [SWP, dC, dL] = small_world_propensity(W, args{:});
                rec.outputs.SWP = SWP; rec.outputs.delta_C = dC; rec.outputs.delta_L = dL;
                rec.outputs.after = rand(1, 4);
                rng(c.seed, 'twister');
                rec.parts = swp_parts(W, args{:});
            case 'smallworldness'
                RW = cell(1, args{1});
                for t = 1:args{1}, RW{t} = randmio_und(W, args{2}); end
                if all(W(:) == 0 | W(:) == 1)
                    rec.outputs.S = smallworldness_bu(W, RW); rec.which = 'bu';
                else
                    rec.outputs.S = smallworldness_wu(W, RW); rec.which = 'wu';
                end
                for t = 1:args{1}, rec.outputs.(sprintf('RW%d', t)) = RW{t}; end
        end
        if ~isfield(rec.outputs, 'after'), rec.outputs.after = rand(1, 4); end
    catch err
        rec.error = struct('identifier', err.identifier, 'message', err.message);
    end
    emit(out, rec);
end
end

function s = swp_parts(A, varargin)
% small_world_propensity's main body up to its outputs, with every intermediate recorded. Same calls in the
% same order as GraphVar's small_world_propensity.m, so the stream is consumed identically.
if isempty(varargin), varargin{1} = 'O'; end
s = struct();
bin_matrix = 0;
if strcmp(varargin{1}, 'bin') == 1
    bin_matrix = 1;
    A = A > 0;
end
symcheck = abs(A - A');
if sum(sum(symcheck)) > 0
    W = symm_matrix(A, bin_matrix);
else
    W = A;
end
n = length(W);
numb_connections = length(find(W > 0));
avg_rad_eff = ceil(numb_connections / n / 2);
s.avg_rad_eff = avg_rad_eff;
W_reg = regular_matrix_generator(W, avg_rad_eff);
s.W_reg = hexs(W_reg);
s.after_reg = hexs(rand_peek());
W_rand = randomize_matrix(W);
s.W_rand = hexs(W_rand);
s.reg_path = hexs(avg_path_matrix(1 ./ W_reg));
s.rand_path = hexs(avg_path_matrix(1 ./ W_rand));
s.net_path = hexs(avg_path_matrix(1 ./ W));
s.reg_clus = hexs(avg_clus_matrix(W_reg, varargin{1}));
s.rand_clus = hexs(avg_clus_matrix(W_rand, varargin{1}));
s.net_clus = hexs(avg_clus_matrix(W, varargin{1}));
s.C_net = hexs(clustering_coef_matrix(W, varargin{1}));
end

function u = rand_peek()
% the next rand word without consuming it
st = rng;
u = rand;
rng(st);
end

function h = hexs(v)
v = double(v);
h = struct('size', size(v), 'hex', {cellstr(num2hex(v(:)))'});
end

function split_local(file, dir)
% Write each local function of file (all but the first) to dir/<name>.m
if ~isfolder(dir), mkdir(dir); end
txt = fileread(file);
lines = regexp(txt, '\r?\n', 'split');
starts = find(~cellfun(@isempty, regexp(lines, '^\s*function\s', 'once')));
for k = 2:numel(starts)
    a = starts(k);
    if k < numel(starts), b = starts(k + 1) - 1; else, b = numel(lines); end
    tok = regexp(lines{a}, 'function\s+(?:\[[^\]]*\]\s*=\s*|\w+\s*=\s*)?(\w+)', 'tokens', 'once');
    fid = fopen(fullfile(dir, [tok{1} '.m']), 'w');
    fprintf(fid, '%s\n', lines{a:b});
    fclose(fid);
end
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
    if ischar(v) || isstruct(v), continue, end
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
