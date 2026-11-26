function ref_p2(inDir, outDir)
% Phase-2 reference: MATLAB's real matrix multiply (MKL dgemm) word for word.
% Inputs from tools/gen_p2.py (cases/p2/gemm.json). Writes outDir/gemm.jsonl,
% one JSON record per line. No blobs: long outputs are split into fields of
% at most 4000 words (name_01, name_02, ...) so every word is kept.
p = jsondecode(fileread(fullfile(inDir, 'gemm.json')));
if ~isfolder(outDir), mkdir(outDir); end
out = fullfile(outDir, 'gemm.jsonl');
if isfile(out), delete(out); end
M = 2^80;

% --- tree: reduction order of the k products
for t = 1:numel(p.tree)
    k = p.tree(t).k;
    calls = p.tree(t).calls;
    for c = 1:numel(calls)
        pr = reshape(calls(c).pairs, [], 2);
        A = ones(k);
        for i = 1:k
            A(i, pr(i, 1)) = M;
            A(i, pr(i, 2)) = -M;
        end
        C = A * ones(k);
        rec = struct('case', sprintf('p2-tree-k%03d-%03d', k, c), 'k', k, 'outputs', struct(), 'notes', struct());
        rec.outputs.col1 = hexs(C(:, 1));
        same = all(all(typecast(C(:), 'uint64') == repmat(typecast(C(:, 1), 'uint64'), k, 1)));
        rec.notes.columns_equal = double(same);
        rec.notes.integer_valued = double(all(C(:) == round(C(:))));
        if ~same
            [Up, ~, Jp] = unique(C.', 'rows', 'stable');
            rec.notes.column_pattern = Jp(:).';
            for u = 1:size(Up, 1)
                rec.outputs.(sprintf('pattern_%03d', u)) = hexs(Up(u, :).');
            end
        end
        emit(out, rec);
    end
end

% --- fma: are products rounded before they are added?
for f = 1:numel(p.fma)
    k = p.fma(f).k;
    A = fromhex(p.fma(f).A);
    B = fromhex(p.fma(f).B);
    C = A * B;
    rec = struct('case', sprintf('p2-fma-k%03d', k), 'k', k, 'outputs', struct(), 'notes', struct());
    rec.outputs.col1 = hexs(C(:, 1));
    rec.notes.columns_equal = double(isequal(typecast(C(:), 'uint64'), repmat(typecast(C(:, 1), 'uint64'), k, 1)));
    emit(out, rec);
end

% --- zero: signs of zero results
for z = 1:numel(p.zero)
    k = p.zero(z).k;
    A = fromhex(p.zero(z).A);
    C = A * ones(k);
    rec = struct('case', sprintf('p2-zero-k%03d', k), 'k', k, 'outputs', struct(), 'notes', struct());
    rec.outputs.col1 = hexs(C(:, 1));
    rec.notes.columns_equal = double(isequal(typecast(C(:), 'uint64'), repmat(typecast(C(:, 1), 'uint64'), k, 1)));
    emit(out, rec);
end

% --- rand: Gaussian products for checking the model
for r = 1:numel(p.rand)
    n = p.rand(r).n;
    A = fromhex(p.rand(r).A);
    B = fromhex(p.rand(r).B);
    rec = struct('case', sprintf('p2-rand-n%03d', n), 'n', n, 'outputs', struct(), 'notes', struct());
    rec.outputs = chunks(rec.outputs, 'AB', A * B);
    rec.outputs = chunks(rec.outputs, 'AtB', A.' * B);
    rec.outputs = chunks(rec.outputs, 'ABt', A * B.');
    rec.notes.AAt_equals_general = double(isequal(typecast(reshape(A * A.', [], 1), 'uint64'), typecast(reshape(A * (A.' + 0), [], 1), 'uint64')));
    rec.notes.AtA_equals_general = double(isequal(typecast(reshape(A.' * A, [], 1), 'uint64'), typecast(reshape((A.' + 0) * A, [], 1), 'uint64')));
    v = B(:, 1);
    rec.outputs.Av = hexs(A * v);
    AV2 = A * [v v];
    rec.notes.Av_equals_twocol = double(isequal(typecast(A * v, 'uint64'), typecast(AV2(:, 1), 'uint64')));
    emit(out, rec);
end

% --- mpow: A^2, A^3 and symmetric products on non-negative weights
for q = 1:numel(p.mpow)
    n = p.mpow(q).n;
    W = fromhex(p.mpow(q).W);
    C = W .^ (1/3);
    rec = struct('case', sprintf('p2-mpow-n%03d', n), 'n', n, 'outputs', struct(), 'notes', struct());
    P3 = C ^ 3;
    rec.outputs = chunks(rec.outputs, 'C3', P3);
    rec.outputs = chunks(rec.outputs, 'C2', C ^ 2);
    u = @(X) typecast(X(:), 'uint64');
    rec.notes.pow2_equals_CC = double(isequal(u(C ^ 2), u(C * C)));
    rec.notes.pow3_equals_CC_C = double(isequal(u(P3), u((C * C) * C)));
    rec.notes.pow3_equals_C_CC = double(isequal(u(P3), u(C * (C * C))));
    rec.notes.pow3_symmetric = double(isequal(u(P3), u(P3.')));
    D = diag(P3);
    rec.outputs.diagC3 = hexs(D);
    emit(out, rec);
end
end

function s = hexs(v)
v = double(v(:));
s = struct('size', [numel(v) 1], 'hex', {cellstr(num2hex(v))'});
end

function o = chunks(o, name, X)
v = double(X(:));
o.([name '_size']) = size(X);
for c = 1:ceil(numel(v) / 4000)
    idx = (c-1)*4000+1 : min(c*4000, numel(v));
    o.(sprintf('%s_%02d', name, c)) = struct('size', [numel(idx) 1], 'hex', {cellstr(num2hex(v(idx)))'});
end
end

function X = fromhex(rec)
sz = rec.size(:)';
if isempty(rec.hex)
    X = zeros(sz);
    return
end
X = reshape(hex2num(cellstr(rec.hex)), sz);
end

function emit(out, rec)
fid = fopen(out, 'a');
fprintf(fid, '%s\n', jsonencode(rec));
fclose(fid);
end
