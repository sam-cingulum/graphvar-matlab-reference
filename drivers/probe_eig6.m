function probe_eig6(inDir, outDir)
% Eig probe 6: [V, D] = eig(A) for n = 600..1000 with every stage's output, the signed zeros inside
% dlarft's T, and kernels at the large sizes. Inputs: inDir/eig6.json and inDir/eig6/*.bin
% (tools/gen_eig6.py). Writes outDir/eig6.jsonl; arrays over 4096 words as big-endian blobs in
% outDir/blobs (FPRev R arrays as big-endian uint16, "type":"uint16"). Record kinds (field "part"):
%   ilaenv   MKL ilaenv for DSYTRD, DORMQL and DORMTR at n = 2..1000, and DSTEDC's SMLSIZ
%   full     eig's V, D and d = eig(A); MKL dsytrd's d0, e0, tau and Ar; MKL dstedc('I')'s w and Z on d0, e0;
%            notes: whether dstedc's w is D, whether MKL dormtr on Ar, tau, Z gives V, whether dsyevd
%            gives V and D, whether dsterf on d0, e0 gives d (1 = same words, or a count of differing words)
%   larft6   MKL dlarft('B','C') T on zero-heavy V and tau (inputs in eig6.json)
%   gemmtn0  MKL dgemm('T','N') with beta = 0 (out) and with beta = 1 on C = +0 (out_b1), zero-heavy inputs
%   gemmnn6  MKL dgemm('N','N'), alpha 1, beta 0, random words (inputs stored); out with ld = rows, out_ld
%            with ldc = max(m, n)
%   fprev6   FPRev trees of dgemm('N','N') (columns deduplicated: R(:, map(pos)))
%   trmm6    MKL dtrmm side 'R', alpha 1, with B placed at leading dimension ldb (inputs stored)
if ~isfolder(fullfile(outDir, 'blobs')), mkdir(fullfile(outDir, 'blobs')); end
mexDir = fullfile(outDir, 'mex');
if ~isfolder(mexDir), mkdir(mexDir); end
for f = {'kpeig.c', 'kpeig2.c', 'kpeig3.c', 'kpeig4.c'}
    mex('-R2018a', '-silent', '-outdir', mexDir, fullfile('probes', 'eig', f{1}), '-lmwlapack', '-lmwblas');
end
addpath(mexDir);
fprintf('%s\n%s\n', version('-blas'), version('-lapack'));
out = fullfile(outDir, 'eig6.jsonl');
if isfile(out), delete(out); end
binDir = fullfile(inDir, 'eig6');
p6 = jsondecode(fileread(fullfile(inDir, 'eig6.json')));

% ---- ilaenv --------------------------------------------------------------------------------------
ns = 2:1000;
rec = newrec('eig6-ilaenv', 'ilaenv');
rec.outputs.n = ns;
for ispec = 1:3
    rec.outputs.(sprintf('ormql_%d', ispec)) = arrayfun(@(n) ila(ispec, 'DORMQL', 'LN', n - 1, n, n - 1, -1), ns);
    rec.outputs.(sprintf('sytrd_%d', ispec)) = arrayfun(@(n) ila(ispec, 'DSYTRD', 'U', n, -1, -1, -1), ns);
    rec.outputs.(sprintf('ormtr_%d', ispec)) = arrayfun(@(n) ila(ispec, 'DORMTR', 'LUN', n, n, -1, -1), ns);
end
rec.outputs.stedc_9 = ila(9, 'DSTEDC', ' ', 0, 0, 0, 0);
emit(out, outDir, rec);

% ---- full eig with stage outputs ------------------------------------------------------------------
for c = cellify(p6.full)
    c = c{1};
    rec = newrec(c.case, 'full');
    rec.kind = c.kind; rec.n = c.n;
    try
        A = fromarr(c.A, binDir);
        [V, D] = eig(A);
        dv = eig(A);
        [d0, e0, tau, Ar] = kpeig('sytrd', A, 'U');
        [w, Z] = kpeig('stedc', d0, e0);
        rec.notes.w_equals_D = same(w, diag(D));
        rec.notes.neq_mkl_ormtr = neq(kpeig2('ormtr', Ar, tau, Z, 'U'), V);
        [ws, Vs] = kpeig('syevd', A, 'V', 'U');
        rec.notes.syevd_V = same(Vs, V);
        rec.notes.syevd_D = same(ws, diag(D));
        rec.notes.sterf_d = same(kpeig('sterf', d0, e0), dv);
        rec.notes.offdiag_D_zero = double(nnz(D - diag(diag(D))) == 0);
        rec.outputs.V = V; rec.outputs.D = diag(D); rec.outputs.d = dv;
        rec.outputs.d0 = d0; rec.outputs.e0 = e0; rec.outputs.tau = tau; rec.outputs.Ar = Ar;
        rec.outputs.stedc_w = w; rec.outputs.stedc_Z = Z;
    catch err
        rec.error = errs(err);
    end
    emit(out, outDir, rec);
end

% ---- zero-heavy dlarft --------------------------------------------------------------------------------
for c = cellify(p6.larft6)
    c = c{1};
    rec = newrec(c.case, 'larft6');
    try
        Vb = fromarr(c.V, binDir); t = fromarr(c.tau, binDir);
        rec.rows = size(Vb, 1); rec.k = size(Vb, 2);
        rec.outputs.T = kpeig3('larft', Vb, t);
    catch err
        rec.error = errs(err);
    end
    emit(out, outDir, rec);
end

% ---- zero-heavy dgemm('T','N'), beta = 0 --------------------------------------------------------------
for c = cellify(p6.gemmtn0)
    c = c{1};
    rec = newrec(c.case, 'gemmtn0');
    try
        Am = fromarr(c.A, binDir); Bm = fromarr(c.B, binDir);
        rec.M = size(Am, 2); rec.N = size(Bm, 2); rec.K = size(Am, 1);
        rec.outputs.out = kpeig3('gemm', 'T', 'N', 1, Am, Bm, 0, zeros(rec.M, rec.N));
        rec.outputs.out_b1 = kpeig3('gemm', 'T', 'N', 1, Am, Bm, 1, zeros(rec.M, rec.N));
    catch err
        rec.error = errs(err);
    end
    emit(out, outDir, rec);
end

% ---- dgemm('N','N') random words ----------------------------------------------------------------------------
gn = cellify(p6.gemmnn6);
for k = 1:numel(gn)
    g = gn{k};
    rec = newrec(sprintf('eig6-gemmnn-%02d-m%03d-n%04d-k%04d', k, g.m, g.n, g.k), 'gemmnn6');
    rec.m = g.m; rec.n = g.n; rec.k = g.k;
    try
        rng(g.seed, 'twister');
        Am = randn(g.m, g.k); Bm = randn(g.k, g.n);
        C = kpeig3('gemm', 'N', 'N', 1, Am, Bm, 0, zeros(g.m, g.n));
        ldc = max(g.m, g.n);
        Cl = kpeig4('gemm_ld', 'N', 'N', 1, Am, Bm, 0, zeros(g.m, g.n), g.m, g.k, ldc);
        rec.ldc_ld = ldc;
        rec.notes.neq_out_ld = neq(Cl, C);
        rec.outputs.A = Am; rec.outputs.B = Bm; rec.outputs.out = C;
        if rec.notes.neq_out_ld ~= 0, rec.outputs.out_ld = Cl; end
    catch err
        rec.error = errs(err);
    end
    emit(out, outDir, rec);
end

% ---- dgemm('N','N') FPRev ---------------------------------------------------------------------------------
fg = cellify(p6.fprev6);
for k = 1:numel(fg)
    g = fg{k};
    rec = newrec(sprintf('eig6-fprev-gemmNN-m%03d-n%03d-L%04d', g.m, g.n, g.L), 'fprev6');
    rec.op = 'gemmNN'; rec.m = g.m; rec.n = g.n; rec.L = g.L;
    try
        R = kpeig3('fprev_gemm', 'N', 'N', g.m, g.n, g.L, g.rows(:)', g.cols(:)');
        rec.outputs.rows = g.rows(:)'; rec.outputs.cols = g.cols(:)';
        [rec.outputs.R, rec.outputs.map] = dedup(R);
    catch err
        rec.error = errs(err);
    end
    emit(out, outDir, rec);
end

% ---- dtrmm side 'R' at a leading dimension ---------------------------------------------------------------
tr = cellify(p6.trmm6);
for k = 1:numel(tr)
    g = tr{k};
    rec = newrec(sprintf('eig6-trmm-%03d-R%s%s%s-m%d-k%02d-ld%03d', k, g.uplo, g.trans, g.diag, g.m, g.k, g.ldb), 'trmm6');
    rec.uplo = g.uplo; rec.trans = g.trans; rec.diag = g.diag; rec.m = g.m; rec.k = g.k; rec.ldb = g.ldb;
    try
        rng(g.seed, 'twister');
        Am = randn(g.k, g.k); Bm = randn(g.m, g.k);
        rec.outputs.A = Am; rec.outputs.B = Bm;
        rec.outputs.out = kpeig4('trmm_ld', 'R', g.uplo, g.trans, g.diag, 1, Am, Bm, g.ldb);
    catch err
        rec.error = errs(err);
    end
    emit(out, outDir, rec);
end
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
if ~isequal(double(U), R(:, ia)), error('kpeig6:fprev', 'FPRev value outside uint16'); end
map = ic(:)';
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

function X = fromarr(rec, binDir)
sz = rec.size(:)';
if isfield(rec, 'bin')
    fid = fopen(fullfile(binDir, rec.bin), 'r', 'ieee-be');
    X = fread(fid, prod(sz), 'double=>double');
    fclose(fid);
    if numel(X) ~= prod(sz), error('kpeig6:input', 'short input file %s', rec.bin); end
    X = reshape(X, sz);
    return
end
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
