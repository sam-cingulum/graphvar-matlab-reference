function ref_phase1(bctRoot, inDir, outDir, part)
% Phase-1 reference outputs on genuine MATLAB: primitives, network
% construction and the first 42 measures. BCT 2019-03-03 functions are
% called where the spec follows BCT; the corrected maths named in the spec
% is computed here with MATLAB built-ins where the spec fixes a bug.
% Writes JSON-lines records; arrays over 4096 elements go to big-endian
% binary blobs that tools/digest_blobs.py turns into SHA-256 digests.
addpath(bctRoot);
if ~isfolder(fullfile(outDir, 'blobs')), mkdir(fullfile(outDir, 'blobs')); end
mats = jsondecode(fileread(fullfile(inDir, 'matrices.json')));
switch part
    case 'prim'
        run_prim(inDir, outDir);
    case 'construct'
        cases = jsondecode(fileread(fullfile(inDir, 'construct.json')));
        out = fullfile(outDir, 'construct.jsonl');
        for k = 1:numel(cases)
            c = cases(k);
            rec = struct('case', c.case, 'outputs', struct(), 'error', [], 'warnings', {{}});
            try
                [W, neg] = construct_case(c, mats);
                rec.outputs.W = W;
                rec.outputs.negatives_present = double(neg);
            catch err
                rec.error = struct('identifier', err.identifier, 'message', err.message);
            end
            emit(out, outDir, rec);
        end
    case 'measures'
        cases = jsondecode(fileread(fullfile(inDir, 'construct.json')));
        byId = containers.Map({cases.case}, num2cell(1:numel(cases)));
        mcases = jsondecode(fileread(fullfile(inDir, 'measures1.json')));
        out = fullfile(outDir, 'measures1.jsonl');
        if ~iscell(mcases), mcases = num2cell(mcases); end
        for k = 1:numel(mcases)
            mc = mcases{k};
            if isfield(mc.graph, 'construct') && ~isempty(mc.graph.construct)
                W = construct_case(cases(byId(mc.graph.construct)), mats);
            else
                W = fromhex(mats.(mc.graph.matrix));
            end
            ids = cellstr(mc.measures);
            rec = struct('case', mc.case, 'outputs', struct(), 'error', [], 'warnings', {{}}, 'notes', struct());
            for q = 1:numel(ids)
                try
                    [val, note] = measure(ids{q}, W);
                    rec.outputs.(ids{q}) = val;
                    if ~isempty(note), rec.notes.(ids{q}) = note; end
                catch err
                    rec.outputs.(ids{q}) = struct('error', err.identifier, 'message', err.message);
                end
            end
            emit(out, outDir, rec);
        end
end
end

% ---------------------------------------------------------------- inputs
function X = fromhex(rec)
sz = rec.size(:)';
if isempty(rec.hex)
    X = zeros(sz);
    return
end
X = reshape(hex2num(cellstr(rec.hex)), sz);
end

function [W, neg] = construct_case(c, mats)
R = fromhex(mats.(c.matrix));
P = [];
if ~isempty(c.pvalues), P = fromhex(mats.(c.pvalues)); end
if size(R, 1) ~= size(R, 2)
    error('GV:construct:notSquare', 'Matrix is not square');
end
bad = find(~isfinite(R), 1);
if ~isempty(bad)
    [i, j] = ind2sub(size(R), bad);
    error('GV:construct:nonFinite', 'Connectivity matrix contains NaN or Inf (subject %s, entry (%d,%d))', c.matrix, i, j);
end
if strcmp(c.type, 'significance') && (isempty(P) || ~isequal(size(P), size(R)) || any(~isfinite(P(:))))
    error('GV:construct:pvalues', 'P-value matrix missing, wrong size or not finite (subject %s)', c.matrix);
end
if isempty(c.mask)
    keep = true(1, size(R, 1));
else
    keep = logical(c.mask(:)');
end
if nnz(keep) < 2
    error('GV:construct:regions', 'Select at least two regions');
end
R = R(keep, keep);
if ~isempty(P), P = P(keep, keep); end
R(R == 0) = 0;                      % every zero becomes +0
n = size(R, 1);
switch c.type
    case 'none'
        W = R; W(1:n+1:end) = 0;
    case 'absolute'
        W = threshold_absolute(R, str2double(c.value));
    case 'significance'
        W = R; W(P > str2double(c.value)) = 0; W(1:n+1:end) = 0;
    case 'proportional'
        W = threshold_proportional(R, str2double(c.value));
end
if strcmp(c.mode, 'binary')
    W(W ~= 0) = 1;
    neg = false;
else
    switch c.neg_rule
        case 'absolute', W = abs(W);
        case 'zero', W(W < 0) = 0;
    end
    neg = any(W(:) < 0);
end
end

% ---------------------------------------------------------------- measures
function [v, note] = measure(id, W)
note = '';
n = size(W, 1);
switch id
    case {'M01', 'M34'}, v = degrees_und(W);
    case {'M02', 'M35'}, [~, ~, v] = degrees_dir(W);
    case {'M03', 'M36'}, v = density_und(W);
    case {'M04', 'M37'}, v = density_dir(W);
    case 'M05'
        v = bfs_distance(W);
        note = struct('bct2019_distance_bin_equal', double(isequaln(v, distance_bin(W))));
    case 'M06', v = charpath(bfs_distance(W), 0, 0);
    case 'M07'
        D = bfs_distance(W);
        v = zeros(n, 1);
        for i = 1:n
            r = D(i, [1:i-1, i+1:n]);
            f = r(isfinite(r));
            v(i) = sum(f) / numel(f);
        end
    case 'M08', [~, ~, v] = charpath(bfs_distance(W), 0, 0);
    case 'M09', [~, ~, ~, v] = charpath(bfs_distance(W), 0, 0);
    case 'M10', [~, ~, ~, ~, v] = charpath(bfs_distance(W), 0, 0);
    case 'M11'
        e = inv_distance(W);
        v = sum(e(:)) ./ (n^2 - n);
        note = struct('bct2019_efficiency_bin_equal', double(isequaln(v, efficiency_bin(W))));
    case 'M12'
        v = local_eff_2019_bfs(W);
        note = struct('bct2019_efficiency_bin_local_equal', double(isequaln(v, efficiency_bin(W, 1))));
    case 'M13', v = clustering_coef_bu(W);
    case 'M14', v = mean(clustering_coef_bu(W));
    case 'M15', v = clustering_coef_bd(W);
    case 'M16', v = mean(clustering_coef_bd(W));
    case 'M17', v = transitivity_bu(W);
    case 'M18', v = transitivity_bd(W);
    case 'M19', v = assortativity_bin(W, 0);
    case 'M20', v = assortativity_bin(W, 1);
    case 'M21', v = assortativity_bin(W, 2);
    case 'M22', v = assortativity_bin(W, 3);
    case 'M23', v = assortativity_bin(W, 4);
    case 'M24'
        A = W ~= 0;
        if ~any(any(tril(A, -1)))
            A = A | A.';
        end
        v = double(max(conncomp(digraph(A), 'Type', 'strong')));
    case 'M25', v = kcoreness_centrality_bu(W);
    case 'M26', v = kcoreness_centrality_bd(W);
    case 'M27', v = edge_nei_overlap_bu(W);
    case 'M28', v = edge_nei_overlap_bd(W);
    case 'M29', [M, ~, ~] = matching_ind(W); v = M + M.';
    case 'M30', [~, M, ~] = matching_ind(W); v = M + M.';
    case 'M31', [~, ~, M] = matching_ind(W); v = M + M.';
    case 'M32', v = rich_club_all(W, sum(W));
    case 'M33', [~, ~, deg] = degrees_dir(W); v = rich_club_all(W, deg);
    case 'M38', [~, ~, v] = strengths_dir(W);
    case 'M39', [v, ~, ~, ~] = strengths_und_sign(W);
    case 'M40', [~, v, ~, ~] = strengths_und_sign(W);
    case 'M41', [~, ~, v, ~] = strengths_und_sign(W);
    case 'M42', [~, ~, ~, v] = strengths_und_sign(W);
    otherwise, error('GV:ref:unknownMeasure', 'unknown measure %s', id);
end
end

function D = bfs_distance(W)
% Shortest path lengths by breadth-first search (spec M05).
D = distances(digraph(W ~= 0), 'Method', 'unweighted');
end

function e = inv_distance(W)
e = 1 ./ bfs_distance(W);
e(1:size(W, 1)+1:end) = 0;
end

function E = local_eff_2019_bfs(A)
% BCT 2019 local efficiency (spec M12) with BFS inverse distances.
n = length(A);
E = zeros(n, 1);
for u = 1:n
    V = find(A(u, :) | A(:, u).');
    sa = A(u, V) + A(V, u).';
    e = inv_distance(A(V, V));
    se = e + e.';
    numer = sum(sum((sa.' * sa) .* se)) / 2;
    if numer ~= 0
        denom = sum(sa).^2 - sum(sa.^2);
        E(u) = numer / denom;
    end
end
end

function R = rich_club_all(W, deg)
% Rich-club coefficient for every level k = 1..n (spec M32/M33).
n = size(W, 1);
R = zeros(1, n);
for k = 1:n
    keep = deg > k;
    sub = W(keep, keep);
    Nk = size(sub, 2);
    Ek = sum(sub(:));
    R(k) = Ek / (Nk * (Nk - 1));
end
end

% ---------------------------------------------------------------- primitives
function run_prim(inDir, outDir)
p = jsondecode(fileread(fullfile(inDir, 'prim.json')));
out = fullfile(outDir, 'prim.jsonl');
for k = 1:numel(p.sum1d)
    v = fromhex(p.sum1d(k));
    rec = struct('case', sprintf('prim-sum1d-%02d', k), 'outputs', struct( ...
        'row', sum(v), 'col', sum(v.'), 'mean', mean(v)), 'error', [], 'warnings', {{}});
    emit(out, outDir, rec);
end
for k = 1:numel(p.matsum)
    X = fromhex(p.matsum(k));
    rec = struct('case', sprintf('prim-matsum-%02d', k), 'outputs', struct( ...
        'dim1', sum(X, 1), 'dim2', sum(X, 2), 'all', sum(X(:)), 'sumsum', sum(sum(X)), ...
        'mean1', mean(X, 1), 'mean2', mean(X, 2)), 'error', [], 'warnings', {{}});
    emit(out, outDir, rec);
end
for k = 1:numel(p.mean)
    v = fromhex(p.mean(k));
    emit(out, outDir, struct('case', sprintf('prim-mean-%02d', k), 'outputs', struct('mean', mean(v), 'sum_over_n', sum(v) / numel(v)), 'error', [], 'warnings', {{}}));
end
for k = 1:numel(p.maxmin)
    X = fromhex(p.maxmin(k));
    [mx, ix] = max(X, [], 2); [mn, in] = min(X, [], 2);
    emit(out, outDir, struct('case', sprintf('prim-maxmin-%02d', k), 'outputs', struct( ...
        'max2', mx, 'imax2', ix, 'min2', mn, 'imin2', in, 'max1', max(X, [], 1), 'min1', min(X, [], 1)), 'error', [], 'warnings', {{}}));
end
x = fromhex(p.square);
sq = zeros(size(x));
for i = 1:numel(x), sq(i) = x(i)^2; end
emit(out, outDir, struct('case', 'prim-square', 'outputs', struct('scalar_pow2', sq, 'elem_pow2', x.^2, 'times', x .* x), 'error', [], 'warnings', {{}}));
emit(out, outDir, struct('case', 'prim-round', 'outputs', struct('round', round(fromhex(p.round))), 'error', [], 'warnings', {{}}));
for k = 1:numel(p.sortrows)
    M = fromhex(p.sortrows(k));
    emit(out, outDir, struct('case', sprintf('prim-sortrows-%02d', k), 'outputs', struct('sorted', sortrows(M, -2)), 'error', [], 'warnings', {{}}));
end
Z = [-0, -1, 2; -0, -0, 3; 4, -0, -5];
emit(out, outDir, struct('case', 'prim-nan-words', 'outputs', struct( ...
    'zero_div_zero', 0 / 0, 'inf_times_zero', Inf * 0, 'inf_minus_inf', Inf - Inf, 'nan_const', NaN, ...
    'zero_div_negzero', 0 / -0, 'one_div_negzero', 1 / -0, 'triu', triu(Z), 'triu1', triu(Z, 1), 'tril', tril(Z), ...
    'zero_assign', assign_zero(Z)), 'error', [], 'warnings', {{}}));
end

function Z = assign_zero(Z)
Z(Z == 0) = 0;
end

% ---------------------------------------------------------------- output
function emit(out, outDir, rec)
rec.outputs = encode(rec.outputs, outDir, rec.case, '');
fid = fopen(out, 'a');
fprintf(fid, '%s\n', jsonencode(rec));
fclose(fid);
end

function s = encode(s, outDir, caseId, prefix)
if ~isstruct(s), return, end
f = fieldnames(s);
for i = 1:numel(f)
    v = s.(f{i});
    if isstruct(v)
        s.(f{i}) = encode(v, outDir, caseId, [prefix f{i} '.']);
    elseif isnumeric(v) || islogical(v)
        v = double(v);
        if numel(v) > 4096
            name = sprintf('%s__%s%s.bin', caseId, prefix, f{i});
            fid = fopen(fullfile(outDir, 'blobs', name), 'w', 'ieee-be');
            fwrite(fid, v(:), 'double');
            fclose(fid);
            s.(f{i}) = struct('size', size(v), 'blob', name);
        elseif isempty(v)
            s.(f{i}) = struct('size', size(v), 'hex', {{}});
        else
            s.(f{i}) = struct('size', size(v), 'hex', {cellstr(num2hex(v(:)))'});
        end
    end
end
end
