function probe_eig3(inDir, outDir)
% Eig probe 3: MATLAB's blocked dsytrd (n > 200), dormtr's back-transform, dstedc's dgemm('N','N'),
% MKL dsteqr on split/tie tridiagonals, and dsyevd's scaling bounds. Inputs: inDir/eig.json
% (tools/gen_eig.py) and inDir/eig3.json (tools/gen_eig3.py). Writes outDir/eig3.jsonl, arrays over
% 4096 words as big-endian blobs in outDir/blobs (FPRev R arrays as big-endian uint16, "type":"uint16";
% 65535 = the masked pair did not both enter that output).
% Record kinds (field "part"):
%   ilaenv   MKL block sizes: DORMQL ispec 1-3, DORMTR, DSYTRD
%   blocked  dsytrd('U') vs the reference DSYTRD loop over MKL dlatrd/dsyr2k/dsytd2; panel snapshots
%   latrd    MKL dlatrd outputs for nb in latrd_nb, against the reference DLATRD over MKL kernels
%   ormtr    which DORMTR composition gives eig's V (blocked dlarft/dlarfb, unblocked dorm2l, ...)
%   steqr    MKL dsteqr('I') on targeted tridiagonals
%   scale    eig on matrices at and beyond dsyevd's scaling bounds
%   fprev    FPRev trees: dgemm NN, dgemv T/N, dtrmm, dsyr2k (columns deduplicated: R(:, map(pos)))
%   kern     random-input kernel records (inputs stored in the record) and signed-zero records
% notes: 1 same words, 0 different, -1 error.
if ~isfolder(fullfile(outDir, 'blobs')), mkdir(fullfile(outDir, 'blobs')); end
mexDir = fullfile(outDir, 'mex');
if ~isfolder(mexDir), mkdir(mexDir); end
for f = {'kpeig.c', 'kpeig2.c', 'kpeig3.c'}
    mex('-R2018a', '-silent', '-outdir', mexDir, fullfile('probes', 'eig', f{1}), '-lmwlapack', '-lmwblas');
end
addpath(mexDir);
fprintf('%s\n%s\n', version('-blas'), version('-lapack'));
out = fullfile(outDir, 'eig3.jsonl');
if isfile(out), delete(out); end
p3 = jsondecode(fileread(fullfile(inDir, 'eig3.json')));
p = jsondecode(fileread(fullfile(inDir, 'eig.json')));
cases = p.cases;
if ~iscell(cases), cases = num2cell(cases); end
bl = p3.blocked;
if ~iscell(bl), bl = num2cell(bl); end

% ---- ilaenv --------------------------------------------------------------
ns = 2:400;
rec = newrec('eig3-ilaenv', 'ilaenv');
rec.outputs.n = ns;
for ispec = 1:3
    rec.outputs.(sprintf('ormql_%d', ispec)) = arrayfun(@(n) ila(ispec, 'DORMQL', 'LN', n - 1, n, n - 1, -1), ns);
    rec.outputs.(sprintf('sytrd_%d', ispec)) = arrayfun(@(n) ila(ispec, 'DSYTRD', 'U', n, -1, -1, -1), ns);
end
rec.outputs.ormtr_1 = arrayfun(@(n) ila(1, 'DORMTR', 'LUN', n, n, -1, -1), ns);
rec.outputs.ormtr_2 = arrayfun(@(n) ila(2, 'DORMTR', 'LUN', n, n, -1, -1), ns);
rec.outputs.ormtr_3 = arrayfun(@(n) ila(3, 'DORMTR', 'LUN', n, n, -1, -1), ns);
emit(out, outDir, rec);

% ---- the symmetric matrices: eig.json + the extra blocked sizes ------------
mats = {};
for k = 1:numel(cases)
    c = cases{k};
    A = fromhex(c.A);
    if ishermitian(A), mats{end + 1} = struct('case', c.case, 'kind', c.kind, 'n', c.n, 'A', A, 'extra', false); end %#ok<AGROW>
end
for k = 1:numel(bl)
    mats{end + 1} = struct('case', bl{k}.case, 'kind', 'gauss', 'n', bl{k}.n, 'A', fromhex(bl{k}.A), 'extra', true); %#ok<AGROW>
end
gaussA = containers.Map('KeyType', 'double', 'ValueType', 'any');
for k = 1:numel(mats), if strcmp(mats{k}.kind, 'gauss'), gaussA(mats{k}.n) = mats{k}.A; end, end

% ---- blocked dsytrd --------------------------------------------------------
for k = 1:numel(mats)
    m = mats{k};
    if m.n <= 200, continue, end
    rec = newrec(['eig3-blocked-' regexprep(m.case, '^[^-]*-', '')], 'blocked');
    rec.kind = m.kind; rec.n = m.n;
    try
        [d, e, tau, Ar] = kpeig('sytrd', m.A, 'U');
        wantSnap = strcmp(m.kind, 'gauss') && any(m.n == p3.snap_n(:)');
        [d2, e2, tau2, Ar2, SA, SW] = kpeig3('sytrd_composed', m.A, double(wantSnap));
        rec.notes.composed_d = same(d2, d); rec.notes.composed_e = same(e2, e);
        rec.notes.composed_tau = same(tau2, tau); rec.notes.composed_Ar = same(Ar2, Ar);
        if m.extra
            rec.outputs.trd_d = d; rec.outputs.trd_e = e; rec.outputs.trd_tau = tau; rec.outputs.trd_Ar = Ar;
            [V, D] = eig(m.A);
            rec.outputs.V = V; rec.outputs.D = diag(D); rec.outputs.d = eig(m.A);
            [w, Z] = kpeig('stedc', d, e);
            rec.outputs.stedc_w = w; rec.outputs.stedc_Z = Z;
            rec.notes.stedc_w_equals_D = same(w, diag(D));
        end
        if wantSnap, rec.outputs.snapA = SA; rec.outputs.snapW = SW; end
    catch err
        rec.error = struct('identifier', err.identifier, 'message', err.message);
    end
    emit(out, outDir, rec);
end

% ---- dlatrd ------------------------------------------------------------------
for n = p3.latrd_n(:)'
    A = gaussA(n);
    for nb = p3.latrd_nb(:)'
        if nb >= n, continue, end
        rec = newrec(sprintf('eig3-latrd-n%03d-nb%02d', n, nb), 'latrd');
        rec.n = n; rec.nb = nb;
        try
            [Arl, el, taul, Wl] = kpeig2('latrd', A, nb, 'U');
            rec.outputs.Ar_cols = Arl(:, n - nb + 1:n);
            rec.outputs.e = el; rec.outputs.tau = taul; rec.outputs.W = Wl;
            for v = 0:1
                [Ac, ec, tc, Wc] = kpeig3('latrd_composed', A, nb, v);
                rec.notes.(sprintf('composed%d', v)) = double(same(Ac, Arl) && same(ec, el) && same(tc, taul) && same(Wc, Wl));
            end
        catch err
            rec.error = struct('identifier', err.identifier, 'message', err.message);
        end
        emit(out, outDir, rec);
    end
end

% ---- dormtr --------------------------------------------------------------------
for k = 1:numel(mats)
    m = mats{k};
    if m.n < 2, continue, end
    rec = newrec(['eig3-ormtr-' regexprep(m.case, '^[^-]*-', '')], 'ormtr');
    rec.kind = m.kind; rec.n = m.n;
    try
        [V, ~] = eig(m.A);
        [d, e, tau, Ar] = kpeig('sytrd', m.A, 'U');
        [~, Z] = kpeig('stedc', d, e);
        rec.notes.mkl_ormtr = same(kpeig2('ormtr', Ar, tau, Z, 'U'), V);
        rec.notes.c_ilaenv = same(kpeig3('ormtr_composed', Ar, tau, Z, 0, -1, 0), V);
        rec.notes.c_ilaenv_cfb = same(kpeig3('ormtr_composed', Ar, tau, Z, 0, -1, 1), V);
        rec.notes.c_unblocked_mkl2l = same(kpeig3('ormtr_composed', Ar, tau, Z, 1e6, -1, 0), V);
        rec.notes.c_unblocked_larf = same(kpeig3('ormtr_composed', Ar, tau, Z, 1e6, 0, 0), V);
        rec.notes.c_unblocked_larf1l = same(kpeig3('ormtr_composed', Ar, tau, Z, 1e6, 1, 0), V);
        for nb = [4 8 16 32]
            rec.notes.(sprintf('c_nb%d', nb)) = same(kpeig3('ormtr_composed', Ar, tau, Z, nb, -1, 0), V);
        end
        if strcmp(m.kind, 'gauss') && any(m.n == p3.ormtr_snap_n(:)')
            [~, Ts, Cs] = kpeig3('ormtr_composed', Ar, tau, Z, 0, -1, 0);
            rec.outputs.Ts = Ts; rec.outputs.Cs = Cs;
        end
    catch err
        rec.error = struct('identifier', err.identifier, 'message', err.message);
    end
    emit(out, outDir, rec);
end

% ---- dsteqr ------------------------------------------------------------------------
st = p3.steqr;
if ~iscell(st), st = num2cell(st); end
for k = 1:numel(st)
    c = st{k};
    rec = newrec(c.case, 'steqr');
    rec.kind = c.kind;
    try
        d = fromhex(c.d); e = fromhex(c.e);
        [w, Z] = kpeig3('steqr', d, e);
        rec.outputs.w = w; rec.outputs.Z = Z;
        [w2, Z2] = kpeig('stedc', d, e);
        rec.notes.stedc_equals_steqr = double(same(w2, w) && same(Z2, Z));
    catch err
        rec.error = struct('identifier', err.identifier, 'message', err.message);
    end
    emit(out, outDir, rec);
end

% ---- scaling bounds --------------------------------------------------------------------
sc = p3.scale;
if ~iscell(sc), sc = num2cell(sc); end
for k = 1:numel(sc)
    c = sc{k};
    rec = newrec(c.case, 'scale');
    rec.kind = c.kind;
    try
        A = fromhex(c.A);
        [V, D] = eig(A);
        rec.outputs.V = V; rec.outputs.D = diag(D); rec.outputs.d = eig(A);
    catch err
        rec.error = struct('identifier', err.identifier, 'message', err.message);
    end
    emit(out, outDir, rec);
end

% ---- FPRev trees -------------------------------------------------------------------------
fg = cellify(p3.fprev_gemm);
for k = 1:numel(fg)
    g = fg{k};
    rec = newrec(sprintf('eig3-fprev-gemmNN-m%03d-n%03d-L%03d', g.m, g.n, g.L), 'fprev');
    rec.op = 'gemmNN'; rec.m = g.m; rec.n = g.n; rec.L = g.L;
    try
        R = kpeig3('fprev_gemm', 'N', 'N', g.m, g.n, g.L, g.rows(:)', g.cols(:)');
        rec.outputs.rows = g.rows(:)'; rec.outputs.cols = g.cols(:)';
        [rec.outputs.R, rec.outputs.map] = dedup(R);
    catch err
        rec.error = struct('identifier', err.identifier, 'message', err.message);
    end
    emit(out, outDir, rec);
end
fv = cellify(p3.fprev_gemv_t);
for k = 1:numel(fv)
    g = fv{k};
    rec = newrec(sprintf('eig3-fprev-gemvT-m%03d-n%02d', g.m, g.n), 'fprev');
    rec.op = 'gemvT'; rec.m = g.m; rec.n = g.n;
    try
        R = kpeig3('fprev_gemv', 'T', g.m, g.n, 1:g.n);
        [rec.outputs.R, rec.outputs.map] = dedup(R);
    catch err
        rec.error = struct('identifier', err.identifier, 'message', err.message);
    end
    emit(out, outDir, rec);
end
fv = cellify(p3.fprev_gemv_n);
for k = 1:numel(fv)
    g = fv{k};
    rec = newrec(sprintf('eig3-fprev-gemvN-m%03d-n%02d', g.m, g.n), 'fprev');
    rec.op = 'gemvN'; rec.m = g.m; rec.n = g.n;
    try
        rows = unique([1:min(g.m, 14), max(1, g.m - 3):g.m]);
        R = kpeig3('fprev_gemv', 'N', g.m, g.n, rows);
        rec.outputs.rows = rows;
        [rec.outputs.R, rec.outputs.map] = dedup(R);
    catch err
        rec.error = struct('identifier', err.identifier, 'message', err.message);
    end
    emit(out, outDir, rec);
end
ft = cellify(p3.fprev_trmm);
for k = 1:numel(ft)
    g = ft{k};
    rec = newrec(sprintf('eig3-fprev-trmmR%s%s%s-m%03d-k%02d', g.uplo, g.trans, g.diag, g.m, g.k), 'fprev');
    rec.op = ['trmmR' g.uplo g.trans g.diag]; rec.m = g.m; rec.k = g.k;
    try
        rows = unique([1:min(g.m, 13), g.m]);
        R = kpeig3('fprev_trmm', g.uplo, g.trans, g.diag, g.m, g.k, rows);
        rec.outputs.rows = rows;
        [rec.outputs.R, rec.outputs.map] = dedup(R);
    catch err
        rec.error = struct('identifier', err.identifier, 'message', err.message);
    end
    emit(out, outDir, rec);
end
fs = cellify(p3.fprev_syr2k);
for k = 1:numel(fs)
    g = fs{k};
    rec = newrec(sprintf('eig3-fprev-syr2k-n%03d-k%02d-d%03d', g.n, g.k, g.d), 'fprev');
    rec.op = 'syr2k'; rec.n = g.n; rec.k = g.k; rec.d = g.d;
    try
        R = kpeig3('fprev_syr2k', g.n, g.k, g.d);
        [rec.outputs.R, rec.outputs.map] = dedup(R);
    catch err
        rec.error = struct('identifier', err.identifier, 'message', err.message);
    end
    emit(out, outDir, rec);
end

% ---- random-input kernel records -------------------------------------------------------------
kr = cellify(p3.kern);
gn = cellify(p3.gemm_nn);
for k = 1:numel(gn)
    g = gn{k};
    kr{end + 1} = struct('kind', 'gemmNN', 'm', g.m, 'n', g.n, 'k', g.k, 'seed', 16000 + k); %#ok<AGROW>
end
for k = 1:numel(kr)
    g = kr{k};
    rec = newrec(sprintf('eig3-kern-%03d-%s', k, g.kind), 'kern');
    rec.kind = g.kind;
    try
        rng(g.seed, 'twister');
        switch g.kind
            case 'gemm'
                rec.ta = g.ta; rec.tb = g.tb;
                if g.ta == 'T', A = randn(g.k, g.m); else, A = randn(g.m, g.k); end
                if g.tb == 'T', B = randn(g.n, g.k); else, B = randn(g.k, g.n); end
                C = randn(g.m, g.n);
                if g.ta == 'T', alpha = 1; else, alpha = -1; end
                rec.outputs.A = A; rec.outputs.B = B; rec.outputs.C = C; rec.outputs.alpha = alpha; rec.outputs.beta = 1;
                rec.outputs.out = kpeig3('gemm', g.ta, g.tb, alpha, A, B, 1, C);
            case 'gemmNN'
                A = randn(g.m, g.k); B = randn(g.k, g.n);
                rec.outputs.A = A; rec.outputs.B = B;
                rec.outputs.out = kpeig3('gemm', 'N', 'N', 1, A, B, 0, zeros(g.m, g.n));
            case 'trmm'
                rec.uplo = g.uplo; rec.trans = g.trans; rec.diag = g.diag;
                A = randn(g.k, g.k); B = randn(g.m, g.k);
                rec.outputs.A = A; rec.outputs.B = B;
                rec.outputs.out = kpeig3('trmm', 'R', g.uplo, g.trans, g.diag, 1, A, B);
            case 'gemvT'
                A = randn(g.m, g.n); x = randn(g.m, 1);
                rec.outputs.A = A; rec.outputs.x = x;
                rec.outputs.out = kpeig3('gemv', 'T', 1, A, x, 0, zeros(g.n, 1), 1);
            case 'gemvNs'
                A = randn(g.m, g.n); s = g.m + 3; xs = randn(s * g.n, 1); y = randn(g.m, 1);
                rec.outputs.A = A; rec.outputs.xs = xs; rec.outputs.incx = s; rec.outputs.y = y;
                rec.outputs.out = kpeig3('gemv', 'N', -1, A, xs, 1, y, s);
            case 'ger'
                A = randn(g.m, g.n); x = randn(g.m, 1); y = randn(g.n, 1); alpha = -rand;
                rec.outputs.A = A; rec.outputs.x = x; rec.outputs.y = y; rec.outputs.alpha = alpha;
                rec.outputs.out = kpeig3('ger', alpha, x, y, A);
            case 'scal'
                x = randn(g.m, 1); alpha = randn;
                rec.outputs.x = x; rec.outputs.alpha = alpha;
                rec.outputs.out = kpeig3('scal', alpha, x);
            case 'syr2k'
                A = randn(g.n, g.k); B = randn(g.n, g.k); C = randn(g.n); C = C + C.';
                rec.outputs.A = A; rec.outputs.B = B; rec.outputs.C = C;
                rec.outputs.out = kpeig3('syr2k', -1, A, B, 1, C);
        end
    catch err
        rec.error = struct('identifier', err.identifier, 'message', err.message);
    end
    emit(out, outDir, rec);
end
% signed zeros
rec = newrec('eig3-kern-signedzero', 'kern');
rec.kind = 'signedzero';
try
    A = [1 2 -0 4; 2 5 -0 7; -0 -0 -0 9; 4 7 9 3]; x = [1; -2; 0; 3]; y = [2; 1; 0; -1];
    rec.outputs.syr2_A = A; rec.outputs.syr2_x = x; rec.outputs.syr2_y = y;
    rec.outputs.syr2 = kpeig3('syr2', -1, x, y, A);
    for n = [1 2 3 4 5 8 9 13]
        rec.outputs.(sprintf('symv_n%02d', n)) = kpeig3('symv', 1, ones(n), -zeros(n, 1));
        rec.outputs.(sprintf('dot_n%02d', n)) = kpeig3('dot', ones(n, 1), -zeros(n, 1));
        rec.outputs.(sprintf('gemvT_n%02d', n)) = kpeig3('gemv', 'T', 1, ones(n, 2), -zeros(n, 1), 0, zeros(2, 1), 1);
    end
catch err
    rec.error = struct('identifier', err.identifier, 'message', err.message);
end
emit(out, outDir, rec);
end

function c = cellify(s)
if iscell(s), c = s(:)'; else, c = num2cell(s(:)'); end
end

function rec = newrec(caseId, part)
rec = struct('case', caseId, 'part', part, 'outputs', struct(), 'notes', struct());
end

function v = ila(ispec, name, opts, n1, n2, n3, n4)
try
    v = kpeig2('ilaenv', ispec, name, opts, n1, n2, n3, n4);
catch
    v = NaN;
end
end

function [U, map] = dedup(R)
% unique columns of R in first-appearance order; R(:, j) == U(:, map(j)). A value that is not a
% count (the masked pair did not both enter that output, so +-M survives) is stored as 65535.
R(~(R >= 0 & R < 65535 & R == round(R))) = 65535;
[~, ia, ic] = unique(R.', 'rows', 'stable');
U = uint16(R(:, ia));
if ~isequal(double(U), R(:, ia)), error('kpeig3:fprev', 'FPRev value outside uint16'); end
map = ic(:)';
end

function eq = same(a, b)
a = double(a); b = double(b);
eq = double(isequal(size(a), size(b)) && isequal(typecast(a(:), 'uint64'), typecast(b(:), 'uint64')));
end

function X = fromhex(rec)
sz = rec.size(:)';
h = cellstr(rec.hex);
if isempty(h), X = zeros(sz); return, end
X = reshape(hex2num(h), sz);
end

function emit(out, outDir, rec)
rec.outputs = encode(rec.outputs, outDir, rec.case);
fid = fopen(out, 'a');
fprintf(fid, '%s\n', jsonencode(rec));
fclose(fid);
end

function s = encode(s, outDir, caseId)
f = fieldnames(s);
for i = 1:numel(f)
    v = s.(f{i});
    if isa(v, 'uint16')
        name = sprintf('%s__%s.u16', caseId, f{i});
        fid = fopen(fullfile(outDir, 'blobs', name), 'w', 'ieee-be');
        fwrite(fid, v(:), 'uint16');
        fclose(fid);
        s.(f{i}) = struct('size', size(v), 'blob', name, 'type', 'uint16');
        continue
    end
    v = double(v);
    if numel(v) > 4096
        name = sprintf('%s__%s.bin', caseId, f{i});
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
