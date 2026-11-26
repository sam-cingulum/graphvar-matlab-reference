function ref_phase1(bctRoot, inDir, outDir, part)
% Phase-1 reference outputs on genuine MATLAB: primitives, network
% construction and the first 42 measures. BCT 2019-03-03 functions are
% called where the spec follows BCT; the corrected maths named in the spec
% is computed here with MATLAB built-ins where the spec fixes a bug.
% Writes JSON-lines records; arrays over 4096 elements go to big-endian
% binary blobs that tools/digest_blobs.py turns into SHA-256 digests.
addpath(bctRoot);
if ~isfolder(fullfile(outDir, 'blobs')), mkdir(fullfile(outDir, 'blobs')); end
if ~startsWith(part, 'p103') && ~startsWith(part, 'p2'), mats = jsondecode(fileread(fullfile(inDir, 'matrices.json'))); end
switch part
    case 'p103prim'
        run_p103_prim(inDir, outDir);
    case 'p103measures'
        run_p103_measures(inDir, outDir);
    case 'p2construct'
        % P1-01 v2 additions: rank = magnitude. Matrices come from phase 1.
        mats = jsondecode(fileread(fullfile(inDir, '..', 'phase1', 'matrices.json')));
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
    case 'p2measures'
        run_p103_measures(fullfile(inDir, '..', 'p103'), outDir, fullfile(inDir, 'measures.json'), inDir);
    case 'p2bmeasures'
        run_p103_measures(fullfile(inDir, '..', 'p103'), outDir, fullfile(inDir, 'bmeasures.json'), inDir);
    case 'p2cmeasures'
        run_p103_measures(fullfile(inDir, '..', 'p103'), outDir, fullfile(inDir, 'cmeasures.json'), inDir);
    case 'prim'
        run_prim(inDir, outDir);
    case 'g1measures'
        % Firstblind G1: every graph key resolves inside inDir; BCT comparisons off.
        setappdata(0, 'GV_NOBCT', true);
        run_g1_measures(inDir, outDir, mats);
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
        if isfield(c, 'rank') && strcmp(c.rank, 'magnitude')
            W = threshold_magnitude(R, str2double(c.value));
        else
            W = threshold_proportional(R, str2double(c.value));
        end
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

function W = threshold_magnitude(W, p)
% Spec P1-01 v2 step B with rank = magnitude: BCT 2019's steps, ranked by |w|.
n = size(W, 1);
W(1:n+1:end) = 0;
if max(max(abs(W - W.'))) < 1e-10
    W = triu(W); ud = 2;
else
    ud = 1;
end
ind = find(W);
E = sortrows([ind abs(W(ind))], -2);
en = round((n^2 - n) * p / ud);
W(E(en+1:end, 1)) = 0;
if ud == 2
    W = W + W.';
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
        note = struct('bct2019_distance_bin_equal', bct_same(@() distance_bin(W), v));
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
        note = struct('bct2019_efficiency_bin_equal', bct_same(@() efficiency_bin(W), v));
    case 'M12'
        v = local_eff_2019_bfs(W);
        note = struct('bct2019_efficiency_bin_local_equal', bct_same(@() efficiency_bin(W, 1), v));
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
    case 'M26'
        % Spec P1-02 M26 (v4): a node counts in the k-core while its TOTAL degree
        % there is > 0. BCT 2019 counts in-degree only (a bug: its own kn uses total).
        n = size(W, 1); v = zeros(1, n);
        for k = 1:n
            Ck = kcore_bd(W, k);
            ss = (sum(Ck, 1) + sum(Ck, 2).') > 0;
            v(ss) = k;
        end
        note = struct('bct2019_equal', bct_equal(@() kcoreness_centrality_bd(W), v));
    case 'M27', v = edge_nei_overlap_bu(W);
    case 'M28', v = edge_nei_overlap_bd(W);
    case 'M29', [M, ~, ~] = matching_ind(W); v = M + M.';
    case 'M30', [~, M, ~] = matching_ind(W); v = M + M.';
    case 'M31', [~, ~, M] = matching_ind(W); v = M + M.';
    case 'M32', v = rich_club_all(W, sum(W), n);
    case 'M33', [~, ~, deg] = degrees_dir(W); v = rich_club_all(W, deg, 2 * n - 2);
    case 'M38', [~, ~, v] = strengths_dir(W);
    case 'M39', [v, ~, ~, ~] = strengths_und_sign(W);
    case 'M40', [~, v, ~, ~] = strengths_und_sign(W);
    case 'M41', [~, ~, v, ~] = strengths_und_sign(W);
    case 'M42', [~, ~, ~, v] = strengths_und_sign(W);
    otherwise, [v, note] = measure_p103(id, W);
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

function R = rich_club_all(W, deg, K)
% Rich-club coefficient for every level k = 1..K (spec M32/M33).
R = zeros(1, K);
for k = 1:K
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

% ================================================================ G1
function run_g1_measures(inDir, outDir, mats)
% Measures for a firstblind kit: {"construct": id} uses inDir/construct.json
% (its matrix and pvalues names point into inDir/matrices.json), {"matrix": name}
% uses inDir/matrices.json. Notes are left empty.
cases = [];
if isfile(fullfile(inDir, 'construct.json')), cases = jsondecode(fileread(fullfile(inDir, 'construct.json'))); end
if isempty(cases), byId = containers.Map(); else, byId = containers.Map({cases.case}, num2cell(1:numel(cases))); end
mcases = jsondecode(fileread(fullfile(inDir, 'measures.json')));
if ~iscell(mcases), mcases = num2cell(mcases); end
out = fullfile(outDir, 'measures.jsonl');
for k = 1:numel(mcases)
    mc = mcases{k};
    g = mc.graph;
    rec = struct('case', mc.case, 'outputs', struct(), 'error', [], 'warnings', {{}}, 'notes', struct());
    try
        if isfield(g, 'construct') && ~isempty(g.construct)
            W = construct_case(cases(byId(g.construct)), mats);
        else
            W = fromhex(mats.(g.matrix));
        end
    catch err
        rec.error = struct('identifier', err.identifier, 'message', err.message);
        emit(out, outDir, rec);
        continue
    end
    ids = cellstr(mc.measures);
    for q = 1:numel(ids)
        try
            [val, ~] = measure(ids{q}, W);
            rec.outputs.(ids{q}) = val;
        catch err
            rec.outputs.(ids{q}) = struct('error', err.identifier, 'message', err.message);
        end
    end
    emit(out, outDir, rec);
end
end

% ================================================================ P1-03
function run_p103_measures(inDir, outDir, measFile, p2Dir)
p1Dir = fullfile(inDir, '..', 'phase1');
p1mats = jsondecode(fileread(fullfile(p1Dir, 'matrices.json')));
p1cases = jsondecode(fileread(fullfile(p1Dir, 'construct.json')));
byId = containers.Map({p1cases.case}, num2cell(1:numel(p1cases)));
mats = jsondecode(fileread(fullfile(inDir, 'matrices.json')));
if nargin < 3, measFile = fullfile(inDir, 'measures.json'); end
if nargin >= 4
    p2mats = jsondecode(fileread(fullfile(p2Dir, 'matrices.json')));
    if isfile(fullfile(p2Dir, 'construct.json'))
        p2cases = jsondecode(fileread(fullfile(p2Dir, 'construct.json')));
        p2byId = containers.Map({p2cases.case}, num2cell(1:numel(p2cases)));
    end
end
mcases = jsondecode(fileread(measFile));
if ~iscell(mcases), mcases = num2cell(mcases); end
out = fullfile(outDir, 'measures.jsonl');
for k = 1:numel(mcases)
    mc = mcases{k};
    g = mc.graph;
    rec = struct('case', mc.case, 'outputs', struct(), 'error', [], 'warnings', {{}}, 'notes', struct());
    try
    if isfield(g, 'construct') && ~isempty(g.construct)
        W = construct_case(p1cases(byId(g.construct)), p1mats);
    elseif isfield(g, 'phase1_matrix') && ~isempty(g.phase1_matrix)
        W = fromhex(p1mats.(g.phase1_matrix));
    elseif isfield(g, 'p2_construct') && ~isempty(g.p2_construct)
        W = construct_case(p2cases(p2byId(g.p2_construct)), p1mats);
    elseif isfield(g, 'p2_matrix') && ~isempty(g.p2_matrix)
        W = fromhex(p2mats.(g.p2_matrix));
    else
        W = fromhex(mats.(g.matrix));
    end
    catch err
        rec.error = struct('identifier', err.identifier, 'message', err.message);
        emit(out, outDir, rec);
        continue
    end
    ids = cellstr(mc.measures);
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

function [v, note] = measure_p103(id, W)
note = '';
n = size(W, 1);
lbl = p103_label(id);
switch id
    case 'M43'
        check_w(W, lbl, true); L = to_lengths(W);
        v = dist_w(W);
        note = struct('bct2019_equal', bct_equal(@() distance_wei(L), v));
    case 'M44'
        check_w(W, lbl, true); v = charpath(dist_w(W), 0, 0);
    case 'M45'
        check_w(W, lbl, true); D = dist_w(W);
        v = zeros(n, 1);
        for i = 1:n
            r = D(i, isfinite(D(i, :)));
            v(i) = sum(r) / (numel(r) - 1);
        end
    case 'M46'
        check_w(W, lbl, true); [~, ~, v] = charpath(dist_w(W), 0, 0);
    case 'M47'
        check_w(W, lbl, true); [~, ~, ~, v] = charpath(dist_w(W), 0, 0);
    case 'M48'
        check_w(W, lbl, true); [~, ~, ~, ~, v] = charpath(dist_w(W), 0, 0);
    case 'M49'
        check_w(W, lbl, true); v = glob_eff_w(W);
        note = struct('bct2019_equal', bct_equal(@() efficiency_wei(W), v));
    case 'M50'
        check_w(W, lbl, true); v = loc_eff_w(W);
        note = struct('bct2019_equal', bct_equal(@() efficiency_wei(W, 1), v));
    case {'M51', 'M52'}
        check_w(W, lbl, true); v = glob_eff_w(W) - density_und(W);
    case 'M53', v = assortativity_wei(W, 0);
    case 'M54', v = assortativity_wei(W, 1);
    case 'M55', v = assortativity_wei(W, 2);
    case 'M56', v = assortativity_wei(W, 3);
    case 'M57', v = assortativity_wei(W, 4);
    case 'M58'
        check_w(W, lbl, false); deg = sum(W ~= 0);
        v = rich_club_w(W, deg, n);
        note = struct('bct2019_equal_where_defined', rc_note(@(kl) rich_club_wu(W, kl), v, deg));
    case 'M59'
        check_w(W, lbl, false); deg = sum(W ~= 0) + sum(W.' ~= 0);
        v = rich_club_w(W, deg, 2 * n - 2);
        note = struct('bct2019_equal_where_defined', rc_note(@(kl) rich_club_wd(W, kl), v, deg));
    case 'M60'
        check_w(W, lbl, true); L = to_lengths(W);
        [v, ~] = sp_betweenness(L);
        note = struct('bct2019_equal', bct_equal(@() betweenness_wei(L), v));
    case 'M61'
        check_w(W, lbl, true); L = to_lengths(W);
        [~, v] = sp_betweenness(L);
        note = struct('bct2019_equal', bct_equal(@() edge_betweenness_wei(L), v));
    case {'M62', 'M63'}
        e = inv_distance(W);
        v = sum(e(:)) ./ (n^2 - n) - density_und(W);
    case 'M64'
        [~, v] = sp_betweenness(double(W ~= 0));
        note = struct('bct2019_equal', bct_equal(@() edge_betweenness_bin(W), v));
    case {'M65', 'M66', 'M67', 'M68', 'M69', 'M70', 'M71', 'M72'}
        check_w(W, lbl, false);
        Wm = W; gvW = [];
        if any(strcmp(id, {'M67', 'M68', 'M71', 'M72'}))
            gvW = W ./ max(abs(W(:)));          % GraphVar's normalisation, kept for the note
            m = max(abs(W(:)));
            if m ~= 0, Wm = W ./ m; end
        end
        dirItem = any(strcmp(id, {'M66', 'M68', 'M70', 'M72'}));
        if dirItem, c = clust_wd(Wm); bf = @clustering_coef_wd; else, c = clust_wu(Wm); bf = @clustering_coef_wu; end
        globalItem = any(strcmp(id, {'M65', 'M66', 'M67', 'M68'}));
        if globalItem, v = mean(c); else, v = c; end
        if isempty(gvW)
            if globalItem
                note = struct('bct2019_equal', bct_equal(@() mean(bf(W)), v));
            else
                note = struct('bct2019_equal', bct_equal(@() bf(W), v));
            end
        else
            if globalItem
                note = struct('graphvar_formula_equal', bct_equal(@() mean(bf(gvW)), v));
            else
                note = struct('graphvar_formula_equal', bct_equal(@() bf(gvW), v));
            end
        end
    case 'M73'
        check_w(W, lbl, false);
        K = sum(W ~= 0, 2);
        cyc3 = diag((W .^ (1/3)) ^ 3);
        den = sum(K .* (K - 1));
        if den == 0, v = NaN; else, v = sum(cyc3) ./ den; end
        note = struct('bct2019_equal', bct_equal(@() transitivity_wu(W), v));
    case 'M74'
        check_w(W, lbl, false);
        A = W ~= 0;
        S = W .^ (1/3) + (W.') .^ (1/3);
        K = sum(A + A.', 2);
        cyc3 = diag(S ^ 3) / 2;
        CYC3 = K .* (K - 1) - 2 * diag(double(A) ^ 2);
        den = sum(CYC3);
        if den == 0, v = NaN; else, v = sum(cyc3) ./ den; end
        Ko = K; Ko(cyc3 == 0) = inf;           % GraphVar 2.03a's bundled copy
        CYo = Ko .* (Ko - 1) - 2 * diag(double(A) ^ 2);
        note = struct('bct2019_equal', bct_equal(@() transitivity_wd(W), v), ...
            'graphvar_old_equal', double(isequaln(sum(cyc3) ./ sum(CYo), v)));
    case {'M76', 'M77'}
        % P2-03: BCT 2019 path_transitivity with the 'inv' transform, after the
        % P1-03 input checks (negative weights refused; lengths must be finite).
        check_w(W, lbl, true);
        v = path_transitivity(W, 'inv');
    case 'M75'
        G = double(W ~= 0);
        v = betweenness_bin_fixed(G);
        ov = walk_overflow(G);
        if ov, note = struct('bct2019_overflow', 1, 'bct2019_equal', -2);
        else, note = struct('bct2019_overflow', 0, 'bct2019_equal', bct_equal(@() betweenness_bin(G), v)); end
    otherwise, error('GV:ref:unknownMeasure', 'unknown measure %s', id);
end
end

function c = clust_wu(W)
% Spec P2-01 M69.
n = size(W, 1);
K = sum(W ~= 0, 2);
cyc3 = diag((W .^ (1/3)) ^ 3);
c = zeros(n, 1);
ok = cyc3 ~= 0 & K >= 2;
c(ok) = cyc3(ok) ./ (K(ok) .* (K(ok) - 1));
end

function c = clust_wd(W)
% Spec P2-01 M70 (BCT 2019 clustering_coef_wd).
n = size(W, 1);
A = W ~= 0;
S = W .^ (1/3) + (W.') .^ (1/3);
K = sum(A + A.', 2);
cyc3 = diag(S ^ 3) / 2;
CYC3 = K .* (K - 1) - 2 * diag(double(A) ^ 2);
c = zeros(n, 1);
ok = cyc3 ~= 0;
c(ok) = cyc3(ok) ./ CYC3(ok);
end

function BC = betweenness_bin_fixed(G)
% P2-02: BCT 2019 betweenness_bin with one change: each step multiplies only
% the newly reached shortest-path counts by G, so walk counts never overflow.
n = length(G);
I = eye(n) ~= 0;
d = 1;
NSPd = G;
NSP = NSPd; NSP(I) = 1;
L = NSPd; L(I) = 1;
while find(NSPd, 1)
    d = d + 1;
    NPd = NSPd * G;
    NSPd = NPd .* (L == 0);
    NSP = NSP + NSPd;
    L = L + d .* (NSPd ~= 0);
end
L(~L) = inf; L(I) = 0;
NSP(~NSP) = 1;
Gt = G.';
DP = zeros(n);
diam = d - 1;
for d = diam:-1:2
    DPd1 = (((L == d) .* (1 + DP) ./ NSP) * Gt) .* ((L == (d - 1)) .* NSP);
    DP = DP + DPd1;
end
BC = sum(DP, 1);
end

function ov = walk_overflow(G)
% 1 if BCT 2019's all-walk counts reach a non-finite value before every pair
% is reached (BCT's loop then never ends), else 0.
n = length(G);
I = eye(n) ~= 0;
NPd = G; NSPd = G; L = G; L(I) = 1; d = 1; ov = 0;
while find(NSPd, 1)
    d = d + 1;
    NPd = NPd * G;
    if any(~isfinite(NPd(:))), ov = 1; return; end
    NSPd = NPd .* (L == 0);
    L = L + d .* (NSPd ~= 0);
end
end

function s = p103_label(id)
labels = struct('M43', 'Distance - UND/DIR', 'M44', 'Characteristic path global - UND/DIR', ...
    'M45', 'Characteristic Path local - UND/DIR', 'M46', 'Eccentricity - UND/DIR', ...
    'M47', 'Graph radius - UND/DIR', 'M48', 'Graph diameter - UND/DIR', 'M49', 'Efficiency global - UND', ...
    'M50', 'Efficiency local - UND', 'M51', 'Cost efficiency global (Relative or SICE thresholding) - UND', ...
    'M52', 'Cost efficiency global (Absolute or Significance thresholding) - UND', ...
    'M58', 'Rich club coefficient - UND', 'M59', 'Rich club coefficient - DIR', ...
    'M60', 'Betweenness centrality', 'M61', 'Edge betweenness centrality - UND/DIR', ...
    'M65', 'Cluster coefficient global - UND', 'M66', 'Cluster coefficient global - DIR', ...
    'M67', 'Cluster coefficient global (normalized input weight) - UND', ...
    'M68', 'Cluster coefficient global (normalized input weight) - DIR', ...
    'M69', 'Clustering coefficient local - UND', 'M70', 'Clustering coefficient local - DIR', ...
    'M71', 'Clustering coefficient local (normalized input weight) - UND', ...
    'M72', 'Clustering coefficient local (normalized input weight) - DIR', ...
    'M73', 'Transitivity - UND', 'M74', 'Transitivity - DIR', ...
    'M76', 'Path transitivity - UND/DIR', 'M77', 'Path transitivity - UND/DIR');
if isfield(labels, id), s = labels.(id); else, s = id; end
end

function check_w(W, lbl, lengths)
bad = find(W < 0, 1);
if ~isempty(bad)
    [i, j] = ind2sub(size(W), bad);
    error('GV:measure:negativeWeights', '%s needs non-negative weights (entry (%d,%d) is negative); choose the absolute or zero negative-weight rule', lbl, i, j);
end
if lengths
    bad = find(W ~= 0 & isinf(1 ./ W), 1);
    if ~isempty(bad)
        [i, j] = ind2sub(size(W), bad);
        error('GV:measure:lengthOverflow', '%s: the weight at entry (%d,%d) is too small to convert to a length', lbl, i, j);
    end
end
end

function L = to_lengths(W)
L = W;
E = W ~= 0;
L(E) = 1 ./ W(E);
end

function eq = bct_same(f, v)
% isequaln comparison with a BCT 2019 function (NaN equal to NaN); -3 when BCT is off.
if isappdata(0, 'GV_NOBCT'), eq = -3; return, end
eq = double(isequaln(v, f()));
end

function eq = bct_equal(f, v)
% 1 if the BCT 2019 function gives the same words, 0 if not, -1 if it errors.
% Words are compared bit for bit (signed zeros and NaN payloads included).
% -3 when BCT comparisons are off (firstblind runs).
if isappdata(0, 'GV_NOBCT'), eq = -3; return, end
try
    r = f();
catch
    eq = -1;
    return
end
r = double(r); v = double(v);
eq = double(isequal(size(r), size(v)) && isequal(typecast(r(:), 'uint64'), typecast(v(:), 'uint64')));
end

function eq = rc_note(f, v, deg)
% Compare with BCT 2019 on the levels where BCT does not return its NaN for
% "no node below level k" (spec P1-03 section 5).
if isappdata(0, 'GV_NOBCT'), eq = -3; return, end
kl = max(deg);
if kl < 1, eq = -1; return, end
try
    Rb = f(kl);
catch
    eq = -1; return
end
lv = find(arrayfun(@(k) any(deg < k), 1:kl));
eq = double(isequaln(Rb(lv), v(lv)));
end

function D = dist_w(W)
% Distances on lengths 1/W, cached for the graph of the current case.
persistent lastW lastD
if isempty(lastW) || ~isequal(size(lastW), size(W)) || ~isequal(typecast(lastW(:), 'uint64'), typecast(W(:), 'uint64'))
    lastW = W;
    lastD = sp_dist(to_lengths(W));
end
D = lastD;
end

function D = sp_dist(L)
% Spec P1-03 section 1.2: Dijkstra with batches of open nodes only.
n = size(L, 1);
D = inf(n);
for u = 1:n
    d = inf(1, n); d(u) = 0;
    open = true(1, n);
    B = u;
    while true
        open(B) = false;
        for v = B
            T = find(L(v, :) ~= 0 & open);
            d(T) = min(d(T), d(v) + L(v, T));   % one addition per candidate; equal values are the same word
        end
        if ~any(open), break, end
        m = min(d(open));
        if isinf(m), break, end
        B = find(open & d == m);
    end
    D(u, :) = d;
end
end

function [BC, EBC] = sp_betweenness(L)
% Spec P1-03 section 7.1.
n = size(L, 1);
BC = zeros(n, 1);
EBC = zeros(n);
for u = 1:n
    d = inf(1, n); d(u) = 0;
    sg = zeros(1, n); sg(u) = 1;
    P = false(n);
    Q = zeros(1, n); q = n;
    open = true(1, n);
    B = u;
    while true
        open(B) = false;
        for v = B
            Q(q) = v; q = q - 1;
            for w = find(L(v, :) ~= 0 & open)
                c = d(v) + L(v, w);
                if c < d(w)
                    d(w) = c; sg(w) = sg(v); P(w, :) = false; P(w, v) = true;
                elseif c == d(w)
                    sg(w) = sg(w) + sg(v); P(w, v) = true;
                end
            end
        end
        if ~any(open), break, end
        m = min(d(open));
        if isinf(m), Q(1:q) = find(open); break, end
        B = find(open & d == m);
    end
    dl = zeros(n, 1);
    for t = 1:n-1
        w = Q(t);
        BC(w) = BC(w) + dl(w);
        for v = find(P(w, :))
            x = (1 + dl(w)) .* sg(v) ./ sg(w);
            dl(v) = dl(v) + x;
            EBC(v, w) = EBC(v, w) + x;
        end
    end
end
end

function E = glob_eff_w(W)
n = size(W, 1);
I = 1 ./ dist_w(W);
I(1:n+1:end) = 0;
E = sum(I(:)) ./ (n^2 - n);
end

function E = loc_eff_w(W)
% Spec P1-03 M50.
n = size(W, 1);
e = 1 / 3;
C = W .^ e;
A = W > 0;
L = to_lengths(W);
E = zeros(n, 1);
for u = 1:n
    V = find(A(u, :) | A(:, u).');
    k = numel(V);
    sw = C(u, V) + C(V, u).';
    I = 1 ./ sp_dist(L(V, V));
    I(1:k+1:end) = 0;
    se = I .^ e + I.' .^ e;
    X = (sw.' * sw) .* se;
    num = sum(sum(X)) / 2;
    if num ~= 0
        sa = double(A(u, V)) + double(A(V, u).');
        E(u) = num / (sum(sa)^2 - sum(sa.^2));
    end
end
end

function R = rich_club_w(W, deg, K)
% Spec P1-03 section 5.
ranked = sort(W(:), 'descend');
R = zeros(1, K);
for k = 1:K
    keep = deg >= k;
    S = W(keep, keep);
    a = sum(S(:));
    Ek = nnz(S);
    b = sum(ranked(1:Ek));
    R(k) = a / b;
end
end

function run_p103_prim(inDir, outDir)
p = jsondecode(fileread(fullfile(inDir, 'prim.json')));
out = fullfile(outDir, 'prim.jsonl');
e = 1 / 3;
x = fromhex(p.pow);
N = numel(x);
sc = zeros(1, N);
for i = 1:N, sc(i) = x(i) ^ e; end
vc = x .^ e;
M = reshape(x, 379, []);
mc = M .^ e;
T = reshape(x(mod(0:379*379-1, N) + 1), 379, 379);
tc = T .^ e;
small = true;
for kk = [1 2 5 44 100 170]
    S = reshape(x(1:kk*kk), kk, kk);
    small = small && isequal(typecast(S(:) .^ e, 'uint64'), typecast(sc(1:kk*kk).', 'uint64'));
end
rec = struct('case', 'p103-pow', 'outputs', struct('e', e), 'error', [], 'warnings', {{}}, 'notes', struct( ...
    'vector_equals_scalar', double(isequal(typecast(vc, 'uint64'), typecast(sc, 'uint64'))), ...
    'matrix_equals_scalar', double(isequal(typecast(mc(:).', 'uint64'), typecast(sc, 'uint64'))), ...
    'big_matrix_equals_scalar', double(isequal(typecast(tc(:).', 'uint64'), typecast(sc(mod(0:379*379-1, N) + 1), 'uint64'))), ...
    'small_matrices_equal_scalar', double(small)));
for c = 1:ceil(N / 4000)
    idx = (c-1)*4000+1 : min(c*4000, N);
    rec.outputs.(sprintf('scalar_%02d', c)) = sc(idx);
end
emit(out, outDir, rec);
for k = 1:numel(p.outer)
    s = fromhex(p.outer(k));
    O = s.' * s;
    E2 = s.' .* s;
    emit(out, outDir, struct('case', sprintf('p103-outer-%02d', k), 'outputs', struct('outer', O), 'error', [], 'warnings', {{}}, ...
        'notes', struct('equals_elementwise', double(isequal(typecast(O(:), 'uint64'), typecast(E2(:), 'uint64'))))));
end
for k = 1:numel(p.sum1d)
    v = fromhex(p.sum1d(k));
    emit(out, outDir, struct('case', sprintf('p103-sum1d-%02d', k), 'outputs', struct('row', sum(v), 'col', sum(v.')), 'error', [], 'warnings', {{}}));
end
for k = 1:numel(p.rowsum)
    X = fromhex(p.rowsum(k));
    emit(out, outDir, struct('case', sprintf('p103-rowsum-%02d', k), 'outputs', struct('dim2', sum(X, 2), 'dim1', sum(X, 1)), 'error', [], 'warnings', {{}}));
end
end
