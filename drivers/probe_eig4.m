function probe_eig4(inDir, outDir, bctDir)
% Eig probe 4: answers to the builder's P2-04 questions (non-finite input, LAPACK non-convergence,
% dlarfg's rescale below 2^-969), the P2-03 walk question, MKL dsteqr stress, and dormtr's strategy
% for the unblocked/blocked crossover (n = 34..47) and n >= 257. Inputs: inDir/eig4.json
% (tools/gen_eig4.py); bctDir = BCT 2019-03-03 (for path_transitivity / distance_wei_floyd).
% Writes outDir/eig4.jsonl, arrays over 4096 words as big-endian blobs in outDir/blobs.
% Record kinds (field "part"):
%   nonfinite  d = eig(A) and [V,D] = eig(A) on NaN/Inf input (outputs, or error identifier + message),
%              plus raw MKL dsyevd('N'/'V','U') through kpeig (its error names LAPACK's info)
%   tiny       d = eig(A), [V,D] = eig(A) and MKL dsytrd('U') where |beta| < 2^-969
%   larfg      MKL dlarfg(alpha, x): beta, v, tau
%   walk       BCT path_transitivity(W, 'inv') (T, or the error), and distance_wei_floyd's hops and Pmat
%   steqr      MKL dsteqr('I')
%   ormtr      eig's V against DORMTR compositions; notes count differing words (0 = same words)
if ~isfolder(fullfile(outDir, 'blobs')), mkdir(fullfile(outDir, 'blobs')); end
mexDir = fullfile(outDir, 'mex');
if ~isfolder(mexDir), mkdir(mexDir); end
for f = {'kpeig.c', 'kpeig2.c', 'kpeig3.c'}
    mex('-R2018a', '-silent', '-outdir', mexDir, fullfile('probes', 'eig', f{1}), '-lmwlapack', '-lmwblas');
end
addpath(mexDir);
fprintf('%s\n%s\n', version('-blas'), version('-lapack'));
out = fullfile(outDir, 'eig4.jsonl');
if isfile(out), delete(out); end
p4 = jsondecode(fileread(fullfile(inDir, 'eig4.json')));

% ---- non-finite input ------------------------------------------------------------------
for c = cellify(p4.nonfinite)
    c = c{1};
    rec = newrec(c.case, 'nonfinite');
    A = fromhex(c.A);
    rec.notes.issymmetric = double(issymmetric(A));
    rec.notes.ishermitian = double(ishermitian(A));
    rec.outputs.A = A;
    lastwarn('', '');
    try
        d = eig(A); rec.outputs.d = d;
    catch err
        rec.d_error = errs(err);
    end
    [wm, wid] = lastwarn(); rec.d_warning = struct('identifier', wid, 'message', wm);
    lastwarn('', '');
    try
        [V, D] = eig(A); rec.outputs.V = V; rec.outputs.D = D;
    catch err
        rec.VD_error = errs(err);
    end
    [wm, wid] = lastwarn(); rec.VD_warning = struct('identifier', wid, 'message', wm);
    try
        [w, ~] = kpeig('syevd', A, 'N', 'U'); rec.outputs.mkl_syevd_N_w = w;
    catch err
        rec.mkl_syevd_N_error = errs(err);
    end
    try
        [w, Z] = kpeig('syevd', A, 'V', 'U'); rec.outputs.mkl_syevd_V_w = w; rec.outputs.mkl_syevd_V_Z = Z;
    catch err
        rec.mkl_syevd_V_error = errs(err);
    end
    emit(out, outDir, rec);
end

% ---- tiny Householder columns --------------------------------------------------------------
for c = cellify(p4.tiny)
    c = c{1};
    rec = newrec(c.case, 'tiny');
    try
        A = fromhex(c.A);
        rec.outputs.d = eig(A);
        [V, D] = eig(A); rec.outputs.V = V; rec.outputs.D = diag(D);
        [d0, e0, tau, Ar] = kpeig('sytrd', A, 'U');
        rec.outputs.trd_U_d = d0; rec.outputs.trd_U_e = e0; rec.outputs.trd_U_tau = tau; rec.outputs.trd_U_Ar = Ar;
    catch err
        rec.error = errs(err);
    end
    emit(out, outDir, rec);
end

% ---- dlarfg kernel ------------------------------------------------------------------------
for c = cellify(p4.larfg)
    c = c{1};
    rec = newrec(c.case, 'larfg');
    try
        [beta, v, tau] = kpeig2('larfg', fromhex(c.alpha), fromhex(c.x));
        rec.outputs.beta = beta; rec.outputs.v = v; rec.outputs.tau = tau;
    catch err
        rec.error = errs(err);
    end
    emit(out, outDir, rec);
end

% ---- P2-03 walk ------------------------------------------------------------------------------
addpath(bctDir);
for c = cellify(p4.walk)
    c = c{1};
    rec = newrec(c.case, 'walk');
    W = fromhex(c.W);
    try
        [~, hops, Pmat] = distance_wei_floyd(W, 'inv');
        rec.outputs.hops = hops; rec.outputs.Pmat = Pmat;
    catch err
        rec.floyd_error = errs(err);
    end
    lastwarn('', '');
    try
        rec.outputs.T = path_transitivity(W, 'inv');
    catch err
        rec.error = errs(err);
    end
    [wm, wid] = lastwarn(); rec.warning = struct('identifier', wid, 'message', wm);
    emit(out, outDir, rec);
end
rmpath(bctDir);

% ---- dsteqr stress ------------------------------------------------------------------------------
for c = cellify(p4.steqr)
    c = c{1};
    rec = newrec(c.case, 'steqr');
    rec.kind = c.kind;
    try
        [w, Z] = kpeig3('steqr', fromhex(c.d), fromhex(c.e));
        rec.outputs.w = w; rec.outputs.Z = Z;
    catch err
        rec.error = errs(err);
    end
    emit(out, outDir, rec);
end

% ---- dormtr strategy ----------------------------------------------------------------------------------
keep = [257 300 379];
for c = cellify(p4.ormtr)
    c = c{1};
    rec = newrec(c.case, 'ormtr');
    rec.n = c.n;
    try
        A = fromhex(c.A); n = c.n;
        [V, D] = eig(A);
        [d, e, tau, Ar] = kpeig('sytrd', A, 'U');
        [w, Z] = kpeig('stedc', d, e);
        rec.notes.w_equals_D = same(w, diag(D));
        rec.notes.neq_mkl_ormtr = neq(kpeig2('ormtr', Ar, tau, Z, 'U'), V);
        rec.notes.neq_c_ilaenv = neq(kpeig3('ormtr_composed', Ar, tau, Z, 0, -1, 0), V);
        rec.notes.neq_c_ilaenv_cfb = neq(kpeig3('ormtr_composed', Ar, tau, Z, 0, -1, 1), V);
        rec.notes.neq_c_unblocked_mkl2l = neq(kpeig3('ormtr_composed', Ar, tau, Z, 1e6, -1, 0), V);
        for nb = [8 16 24 32 48 64 96 128 192 256]
            rec.notes.(sprintf('neq_c_nb%d', nb)) = neq(kpeig3('ormtr_composed', Ar, tau, Z, nb, -1, 0), V);
        end
        Q = kpeig('orgtr', Ar, tau, 'U');
        rec.notes.neq_orgtr_mtimes = neq(Q * Z, V);
        rec.notes.neq_orgtr_gemm = neq(kpeig3('gemm', 'N', 'N', 1, Q, Z, 0, zeros(n)), V);
        if any(n == keep)
            rec.outputs.V = V; rec.outputs.Ar = Ar; rec.outputs.tau = tau; rec.outputs.Z = Z; rec.outputs.Q = Q;
        end
    catch err
        rec.error = errs(err);
    end
    emit(out, outDir, rec);
end
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
