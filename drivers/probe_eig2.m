function probe_eig2(inDir, outDir)
% Eig probe 2: the stages and kernels of MATLAB's symmetric [V,D] = eig(A)
% path (MKL dsyevd 'U' = dsytrd + dstedc + dormtr; probe 1, run 45893231372).
% Inputs: inDir/eig.json (tools/gen_eig.py) and inDir/eig2.json
% (tools/gen_eig2.py). Writes outDir/eig2.jsonl, arrays over 4096 words as
% big-endian blobs in outDir/blobs. Record kinds (field "part"):
%   ilaenv   block sizes MKL reports, per n
%   stage    per symmetric eig.json case: dsytrd('U') reflectors, dsytd2
%            against dsytrd, dstedc on MKL's tridiagonal (w, Z), dormtr(Z)
%            against eig's V, the first dlatrd panel
%   fprev    FPRev summation trees of dsymv('U') and ddot
%   kernel   random-input records for dsymv, dsyr2, ddot, dnrm2, daxpy,
%            dlarfg, dgemv, dsyr2k
% notes: 1 same words, 0 different, -1 error.
if ~isfolder(fullfile(outDir, 'blobs')), mkdir(fullfile(outDir, 'blobs')); end
mexDir = fullfile(outDir, 'mex');
if ~isfolder(mexDir), mkdir(mexDir); end
mex('-R2018a', '-silent', '-outdir', mexDir, fullfile('probes', 'eig', 'kpeig.c'), '-lmwlapack', '-lmwblas');
mex('-R2018a', '-silent', '-outdir', mexDir, fullfile('probes', 'eig', 'kpeig2.c'), '-lmwlapack', '-lmwblas');
addpath(mexDir);
fprintf('%s\n%s\n', version('-blas'), version('-lapack'));
out = fullfile(outDir, 'eig2.jsonl');
if isfile(out), delete(out); end
p2 = jsondecode(fileread(fullfile(inDir, 'eig2.json')));

% ---- ilaenv ------------------------------------------------------------
ns = p2.ilaenv_n(:)';
rec = struct('case', 'eig2-ilaenv', 'part', 'ilaenv', 'outputs', struct(), 'notes', struct());
rec.outputs.n = ns;
q = @(varargin) arrayfun(@(n) try_ilaenv(varargin{:}, n), ns);
rec.outputs.sytrd_nb = q(1, 'DSYTRD', 'U');
rec.outputs.sytrd_nbmin = q(2, 'DSYTRD', 'U');
rec.outputs.sytrd_nx = q(3, 'DSYTRD', 'U');
rec.outputs.ormql_nb = arrayfun(@(n) try_ilaenv4(1, 'DORMQL', 'LN', n - 1, n, n - 1, -1), ns);
rec.outputs.ormtr_nb = arrayfun(@(n) try_ilaenv4(1, 'DORMTR', 'LUN', n, n, -1, -1), ns);
rec.outputs.stedc_smlsiz = try_ilaenv4(9, 'DSTEDC', ' ', 0, 0, 0, 0);
emit(out, outDir, rec);

% ---- stages ------------------------------------------------------------
p = jsondecode(fileread(fullfile(inDir, 'eig.json')));
cases = p.cases;
if ~iscell(cases), cases = num2cell(cases); end
gauss = containers.Map('KeyType', 'double', 'ValueType', 'any');
for k = 1:numel(cases)
    c = cases{k};
    A = fromhex(c.A);
    if strcmp(c.kind, 'gauss'), gauss(c.n) = A; end
    if ~ishermitian(A), continue, end
    n = c.n;
    rec = struct('case', ['eig2-stage-' c.case(5:end)], 'part', 'stage', 'kind', c.kind, 'n', n, 'outputs', struct(), 'notes', struct());
    try
        [V, D] = eig(A);
        [d, e, tau, Ar] = kpeig('sytrd', A, 'U');
        rec.outputs.trd_d = d;
        rec.outputs.trd_e = e;
        rec.outputs.trd_tau = tau;
        rec.outputs.trd_Ar = Ar;
        [d2, e2, tau2, Ar2] = kpeig2('sytd2', A, 'U');
        rec.notes.sytd2_d = same(d2, d);
        rec.notes.sytd2_e = same(e2, e);
        rec.notes.sytd2_tau = same(tau2, tau);
        rec.notes.sytd2_Ar = same(Ar2, Ar);
        if ~(rec.notes.sytd2_d && rec.notes.sytd2_e && rec.notes.sytd2_tau && rec.notes.sytd2_Ar)
            rec.outputs.td2_d = d2;
            rec.outputs.td2_e = e2;
            rec.outputs.td2_tau = tau2;
            rec.outputs.td2_Ar = Ar2;
        end
        [w, Z] = kpeig('stedc', d, e);
        rec.outputs.stedc_w = w;
        rec.outputs.stedc_Z = Z;
        rec.notes.stedc_w_equals_D = same(w, diag(D));
        C = kpeig2('ormtr', Ar, tau, Z, 'U');
        rec.notes.ormtr_equals_V = same(C, V);
        if ~rec.notes.ormtr_equals_V, rec.outputs.ormtr_C = C; end
        nb = try_ilaenv(1, 'DSYTRD', 'U', n);
        if strcmp(c.kind, 'gauss') && nb > 1 && nb < n
            [Arl, el, taul, Wl] = kpeig2('latrd', A, nb, 'U');
            rec.outputs.latrd_nb = nb;
            rec.outputs.latrd_Ar = Arl;
            rec.outputs.latrd_e = el;
            rec.outputs.latrd_tau = taul;
            rec.outputs.latrd_W = Wl;
        end
    catch err
        rec.error = struct('identifier', err.identifier, 'message', err.message);
    end
    emit(out, outDir, rec);
end

% ---- FPRev trees ---------------------------------------------------------
for n = p2.fprev_symv_n(:)'
    if n <= 48, rows = 1:n; else, rows = unique([1:4, 8, 16, 32, 33, floor(n / 2), n - 4:n]); end
    rec = struct('case', sprintf('eig2-fprev-symv-n%03d', n), 'part', 'fprev', 'n', n, 'outputs', struct(), 'notes', struct());
    try
        rec.outputs.rows = rows;
        rec.outputs.R = kpeig2('fprev_symv', n, rows, 'U');
    catch err
        rec.error = struct('identifier', err.identifier, 'message', err.message);
    end
    emit(out, outDir, rec);
end
for n = p2.fprev_dot_n(:)'
    rec = struct('case', sprintf('eig2-fprev-dot-n%03d', n), 'part', 'fprev', 'n', n, 'outputs', struct(), 'notes', struct());
    try
        rec.outputs.r = kpeig2('fprev_dot', n);
    catch err
        rec.error = struct('identifier', err.identifier, 'message', err.message);
    end
    emit(out, outDir, rec);
end

% ---- random-input kernel records -------------------------------------------
kr = p2.kernel;
if ~iscell(kr), kr = num2cell(kr); end
for k = 1:numel(kr)
    r = kr{k};
    n = r.n;
    rec = struct('case', sprintf('eig2-kernel-n%03d', n), 'part', 'kernel', 'n', n, 'outputs', struct(), 'notes', struct());
    try
        A = gauss(n);
        x = fromhex(r.x); y = fromhex(r.y); alpha = fromhex(r.alpha);
        rec.outputs.symv_1 = kpeig2('symv', 1, A, x, 'U');
        rec.outputs.symv_a = kpeig2('symv', alpha, A, x, 'U');
        rec.outputs.syr2_m1 = kpeig2('syr2', -1, x, y, A, 'U');
        rec.outputs.syr2_a = kpeig2('syr2', alpha, x, y, A, 'U');
        rec.outputs.dot = kpeig2('dot', x, y);
        rec.outputs.nrm2 = kpeig2('nrm2', x);
        rec.outputs.axpy = kpeig2('axpy', alpha, x, y);
        if n >= 2
            [b, v, t] = kpeig2('larfg', x(1), x(2:end));
            rec.outputs.larfg_beta = b;
            rec.outputs.larfg_v = v;
            rec.outputs.larfg_tau = t;
        end
        kk = min(n, 32);
        rec.outputs.gemv_T = kpeig2('gemv', 'T', -1, A(:, 1:kk), x, 1, y(1:kk));
        rec.outputs.gemv_N = kpeig2('gemv', 'N', -1, A(:, 1:kk), x(1:kk), 1, y);
        Vb = A(:, 1:kk); Wb = A(:, n - kk + 1:n);
        rec.outputs.syr2k_kk = kk;
        rec.outputs.syr2k = kpeig2('syr2k', -1, Vb, Wb, 1, A, 'U', 'N');
        rec.notes.dot_equals_mtimes = same(rec.outputs.dot, x.' * y);
    catch err
        rec.error = struct('identifier', err.identifier, 'message', err.message);
    end
    emit(out, outDir, rec);
end
end

function v = try_ilaenv(ispec, name, opts, n)
v = try_ilaenv4(ispec, name, opts, n, -1, -1, -1);
end

function v = try_ilaenv4(ispec, name, opts, n1, n2, n3, n4)
try
    v = kpeig2('ilaenv', ispec, name, opts, n1, n2, n3, n4);
catch
    v = NaN;
end
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
