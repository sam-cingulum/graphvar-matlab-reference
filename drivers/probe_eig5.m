function probe_eig5(inDir, outDir)
% Eig probe 5: the back-transform and divide-and-conquer stages of [V, D] = eig(A).
% Inputs: inDir/eig5.json (tools/gen_eig5.py). Writes outDir/eig5.jsonl, arrays over 4096 words as
% big-endian blobs in outDir/blobs. Record kinds (field "part"):
%   panel   eig's V against DORMTR('L','U','N') applied to Z in column panels of width P (each panel a
%           separate composition: MKL dlarft + dlarfb per block, nb forced to 32 or from ilaenv), and
%           against MKL dormtr itself on each panel; notes count differing words (0 = same words);
%           colneq_* = differing words per column; Ar, tau, Z, V and the full composition's Ts for some n
%   orm2l   MKL dormtr('L','U','N') on zero-heavy inputs with n <= 47 (unblocked dorm2l), signed zeros
%   stedc   MKL dstedc('I') w and Z, plus MKL dsteqr('I') on the same tridiagonal
%   kern5   random-input words for dgemm 'T','N' / 'N','T' and dtrmm at DLARFB's shapes (inputs stored)
%   larft5  random-input words for MKL dlarft('B','C') (inputs stored)
if ~isfolder(fullfile(outDir, 'blobs')), mkdir(fullfile(outDir, 'blobs')); end
mexDir = fullfile(outDir, 'mex');
if ~isfolder(mexDir), mkdir(mexDir); end
for f = {'kpeig.c', 'kpeig2.c', 'kpeig3.c'}
    mex('-R2018a', '-silent', '-outdir', mexDir, fullfile('probes', 'eig', f{1}), '-lmwlapack', '-lmwblas');
end
addpath(mexDir);
fprintf('%s\n%s\n', version('-blas'), version('-lapack'));
out = fullfile(outDir, 'eig5.jsonl');
if isfile(out), delete(out); end
p5 = jsondecode(fileread(fullfile(inDir, 'eig5.json')));

% ---- dormtr column panels ----------------------------------------------------------------------
keep = [257 400 513];
widths = [128 192 255 256 257 512];
for c = cellify(p5.panel)
    c = c{1};
    rec = newrec(c.case, 'panel');
    rec.n = c.n;
    try
        A = fromhex(c.A); n = c.n;
        [V, D] = eig(A);
        [d, e, tau, Ar] = kpeig('sytrd', A, 'U');
        [w, Z] = kpeig('stedc', d, e);
        rec.notes.w_equals_D = same(w, diag(D));
        rec.notes.neq_mkl_ormtr = neq(kpeig2('ormtr', Ar, tau, Z, 'U'), V);
        [Cf, Ts] = kpeig3('ormtr_composed', Ar, tau, Z, 32, -1, 0);
        rec.notes.neq_full_nb32 = neq(Cf, V);
        rec.outputs.colneq_full_nb32 = colneq(Cf, V);
        for P = widths
            Cp32 = zeros(n); Cpil = zeros(n); Cmk = zeros(n);
            for j0 = 1:P:n
                cols = j0:min(n, j0 + P - 1);
                Cp32(:, cols) = kpeig3('ormtr_composed', Ar, tau, Z(:, cols), 32, -1, 0);
                Cpil(:, cols) = kpeig3('ormtr_composed', Ar, tau, Z(:, cols), 0, -1, 0);
                Cmk(:, cols) = kpeig2('ormtr', Ar, tau, Z(:, cols), 'U');
            end
            rec.notes.(sprintf('neq_p%d_nb32', P)) = neq(Cp32, V);
            rec.notes.(sprintf('neq_p%d_ilaenv', P)) = neq(Cpil, V);
            rec.notes.(sprintf('neq_p%d_mkl', P)) = neq(Cmk, V);
            if P == 256
                rec.outputs.colneq_p256_nb32 = colneq(Cp32, V);
                rec.outputs.colneq_p256_mkl = colneq(Cmk, V);
                % a narrow last panel through the unblocked path (MKL dorm2l)
                j0 = 256 * floor((n - 1) / 256) + 1;
                cols = j0:n;
                Cu = Cp32;
                Cu(:, cols) = kpeig3('ormtr_composed', Ar, tau, Z(:, cols), 1e6, -1, 0);
                rec.notes.neq_p256_lastunblocked = neq(Cu, V);
                rec.outputs.colneq_p256_lastunblocked = colneq(Cu, V);
            end
        end
        % one column at a time (MKL dormtr on n x 1)
        C1 = zeros(n);
        for j = 1:n, C1(:, j) = kpeig2('ormtr', Ar, tau, Z(:, j), 'U'); end
        rec.notes.neq_cols1_mkl = neq(C1, V);
        rec.outputs.colneq_cols1_mkl = colneq(C1, V);
        if any(n == keep)
            rec.outputs.V = V; rec.outputs.Ar = Ar; rec.outputs.tau = tau; rec.outputs.Z = Z; rec.outputs.Ts = Ts;
        end
    catch err
        rec.error = errs(err);
    end
    emit(out, outDir, rec);
end

% ---- dorm2l with signed zeros ----------------------------------------------------------------------
for c = cellify(p5.orm2l)
    c = c{1};
    rec = newrec(c.case, 'orm2l');
    rec.n = c.n;
    try
        A = fromhex(c.A); Z = fromhex(c.Z);
        [d, e, tau, Ar] = kpeig('sytrd', A, 'U');
        rec.outputs.Ar = Ar; rec.outputs.tau = tau; rec.outputs.Z = Z;
        rec.outputs.out = kpeig2('ormtr', Ar, tau, Z, 'U');
    catch err
        rec.error = errs(err);
    end
    emit(out, outDir, rec);
end

% ---- dstedc ------------------------------------------------------------------------------------------
for c = cellify(p5.stedc)
    c = c{1};
    rec = newrec(c.case, 'stedc');
    rec.kind = c.kind;
    try
        d = fromhex(c.d); e = fromhex(c.e);
        [w, Z] = kpeig('stedc', d, e);
        rec.outputs.w = w; rec.outputs.Z = Z;
        [w2, Z2] = kpeig3('steqr', d, e);
        rec.outputs.steqr_w = w2; rec.outputs.steqr_Z = Z2;
    catch err
        rec.error = errs(err);
    end
    emit(out, outDir, rec);
end

% ---- DLARFB kernels -----------------------------------------------------------------------------------
kr = cellify(p5.kern5);
for k = 1:numel(kr)
    g = kr{k};
    rec = newrec(sprintf('eig5-kern-%03d-%s', k, g.kind), 'kern5');
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
            case 'trmm'
                rec.uplo = g.uplo; rec.trans = g.trans; rec.diag = g.diag;
                A = randn(g.k, g.k); B = randn(g.m, g.k);
                rec.outputs.A = A; rec.outputs.B = B;
                rec.outputs.out = kpeig3('trmm', 'R', g.uplo, g.trans, g.diag, 1, A, B);
        end
    catch err
        rec.error = errs(err);
    end
    emit(out, outDir, rec);
end

% ---- dlarft --------------------------------------------------------------------------------------------
lf = cellify(p5.larft5);
for k = 1:numel(lf)
    g = lf{k};
    rec = newrec(sprintf('eig5-larft-%03d-r%03d-k%02d', k, g.rows, g.ib), 'larft5');
    rec.rows = g.rows; rec.ib = g.ib;
    try
        rng(g.seed, 'twister');
        Vb = randn(g.rows, g.ib) / sqrt(g.rows); t = 1 + rand(g.ib, 1);
        rec.outputs.V = Vb; rec.outputs.tau = t;
        rec.outputs.T = kpeig3('larft', Vb, t);
    catch err
        rec.error = errs(err);
    end
    emit(out, outDir, rec);
end
end

function v = colneq(a, b)
a = double(a); b = double(b);
v = sum(reshape(typecast(a(:), 'uint64') ~= typecast(b(:), 'uint64'), size(a)), 1)';
end

function s = errs(err)
s = struct('identifier', err.identifier, 'message', err.message);
end

function c = cellify(s)
if iscell(s), c = s(:)'; else, c = num2cell(s(:)'); end
end

function rec = newrec(caseId, part)
rec = struct('case', caseId, 'part', part, 'outputs', struct(), 'notes', struct());
end

function k = neq(a, b)
a = double(a); b = double(b);
if ~isequal(size(a), size(b)), k = -1; return, end
k = nnz(typecast(a(:), 'uint64') ~= typecast(b(:), 'uint64'));
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
    v = double(s.(f{i}));
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
