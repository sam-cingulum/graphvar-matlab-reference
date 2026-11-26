function probe_eig(inDir, outDir)
% Eig probe: which LAPACK path MATLAB's eig takes on real symmetric matrices,
% and its stages (tridiagonal reduction, tridiagonal solver, back-transform).
% Inputs from tools/gen_eig.py (inDir/eig.json). Builds probes/eig/kpeig.c
% against MATLAB's own LAPACK, then for every case compares eig's words with
% each candidate routine word for word (notes: 1 same, 0 different, -1 error).
% Writes outDir/eig.jsonl; arrays over 4096 words go to big-endian blobs in
% outDir/blobs (kept: the raw words are needed for modelling).
if ~isfolder(fullfile(outDir, 'blobs')), mkdir(fullfile(outDir, 'blobs')); end
mexDir = fullfile(outDir, 'mex');
if ~isfolder(mexDir), mkdir(mexDir); end
mex('-R2018a', '-silent', '-outdir', mexDir, fullfile('probes', 'eig', 'kpeig.c'), '-lmwlapack', '-lmwblas');
addpath(mexDir);
fprintf('%s\n%s\n', version('-blas'), version('-lapack'));
p = jsondecode(fileread(fullfile(inDir, 'eig.json')));
cases = p.cases;
if ~iscell(cases), cases = num2cell(cases); end
out = fullfile(outDir, 'eig.jsonl');
if isfile(out), delete(out); end
procs = {'syev', 'syevd', 'syevr', 'syevx'};
uplos = {'L', 'U'};
for k = 1:numel(cases)
    c = cases{k};
    A = fromhex(c.A);
    rec = struct('case', c.case, 'kind', c.kind, 'n', c.n, 'outputs', struct(), 'notes', struct());
    rec.notes.ishermitian = double(ishermitian(A));
    try
        [V, D] = eig(A);
        d = eig(A);
    catch err
        rec.error = struct('identifier', err.identifier, 'message', err.message);
        emit(out, outDir, rec);
        continue
    end
    rec.notes.isreal_V = double(isreal(V));
    if ~isreal(V) || ~isreal(D) || ~isreal(d)
        emit(out, outDir, rec);
        continue
    end
    dV = diag(D);
    rec.outputs.V = V;
    rec.outputs.D = dV;
    rec.outputs.d = d;
    rec.notes.d_equals_diagD = same(d, dV);
    for i = 1:numel(procs)
        for j = 1:numel(uplos)
            tag = [procs{i} '_' uplos{j}];
            try
                [w, Z] = kpeig(procs{i}, A, 'V', uplos{j});
                rec.notes.([tag '_V']) = same(Z, V);
                rec.notes.([tag '_Vsign']) = same_up_to_sign(Z, V);
                rec.notes.([tag '_D']) = same(w, dV);
            catch
                rec.notes.([tag '_V']) = -1;
            end
            try
                w = kpeig(procs{i}, A, 'N', uplos{j});
                rec.notes.([tag '_N']) = same(w, d);
            catch
                rec.notes.([tag '_N']) = -1;
            end
        end
    end
    for j = 1:numel(uplos)
        u = uplos{j};
        try
            [dd, e, tau, Ar] = kpeig('sytrd', A, u);
            rec.outputs.(['trd_' u '_d']) = dd;
            rec.outputs.(['trd_' u '_e']) = e;
            rec.outputs.(['trd_' u '_tau']) = tau;
            rec.notes.(['sterf_' u]) = same(kpeig('sterf', dd, e), d);
            [w2, Z2] = kpeig('stedc', dd, e);
            rec.notes.(['stedc_' u '_D']) = same(w2, dV);
            [w3, ~] = kpeig('steqr', dd, e);
            rec.notes.(['steqr_' u '_D']) = same(w3, dV);
            Q = kpeig('orgtr', Ar, tau, u);
            rec.notes.(['Q_times_stedc_' u '_V']) = same(Q * Z2, V);
        catch
            rec.notes.(['sytrd_' u]) = -1;
        end
    end
    emit(out, outDir, rec);
end
end

function eq = same(a, b)
a = double(a); b = double(b);
eq = double(isequal(size(a), size(b)) && isequal(typecast(a(:), 'uint64'), typecast(b(:), 'uint64')));
end

function eq = same_up_to_sign(Z, V)
% 1 if every column of Z equals the same column of V or its negation, word for word.
eq = double(isequal(size(Z), size(V)));
if ~eq, return, end
for q = 1:size(V, 2)
    if ~same(Z(:, q), V(:, q)) && ~same(-Z(:, q), V(:, q)), eq = 0; return, end
end
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
