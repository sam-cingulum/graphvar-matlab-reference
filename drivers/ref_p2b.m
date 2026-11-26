function ref_p2b(inDir, outDir)
% Phase-2b reference: one-operand Gram products, distinct-operand transposed
% products and short inner products, word for word. Inputs from
% tools/gen_p2b.py (cases/p2b/gram.json). Writes outDir/gram.jsonl.
p = jsondecode(fileread(fullfile(inDir, 'gram.json')));
if ~isfolder(outDir), mkdir(outDir); end
out = fullfile(outDir, 'gram.jsonl');
if isfile(out), delete(out); end
u = @(X) typecast(X(:), 'uint64');

for s = 1:numel(p.square)
    n = p.square(s).n;
    A = fromhex(p.square(s).A);
    B = fromhex(p.square(s).B);
    rec = struct('case', sprintf('p2b-sq-n%03d', n), 'n', n, 'outputs', struct(), 'notes', struct());
    G = A.' * A;
    H = A * A.';
    rec.outputs = chunks(rec.outputs, 'AtA', G);
    rec.outputs = chunks(rec.outputs, 'AAt', H);
    rec.outputs = chunks(rec.outputs, 'AtB', A.' * B);
    rec.outputs = chunks(rec.outputs, 'ABt', A * B.');
    rec.notes.AtA_symmetric = double(isequal(u(G), u(G.')));
    rec.notes.AAt_symmetric = double(isequal(u(H), u(H.')));
    rec.notes.AtA_equals_copy = double(isequal(u(G), u((A.' + 0) * A)));
    emit(out, rec);
end

for s = 1:numel(p.rect)
    r = p.rect(s).r; w = p.rect(s).w;
    X = fromhex(p.rect(s).X);
    Y = fromhex(p.rect(s).Y);
    rec = struct('case', sprintf('p2b-rect-r%03d-w%03d', r, w), 'r', r, 'w', w, 'outputs', struct(), 'notes', struct());
    G = X.' * X;
    rec.outputs = chunks(rec.outputs, 'XtX', G);
    rec.outputs = chunks(rec.outputs, 'XtY', X.' * Y);
    rec.notes.XtX_symmetric = double(isequal(u(G), u(G.')));
    emit(out, rec);
end

for s = 1:numel(p.inner)
    L = p.inner(s).L;
    a = fromhex(p.inner(s).a);
    b = fromhex(p.inner(s).b);
    rec = struct('case', sprintf('p2b-dot-%03d', L), 'L', L, 'outputs', struct(), 'notes', struct());
    rec.outputs.atb = hexs(a.' * b);
    rec.outputs.bta = hexs(b.' * a);
    rec.outputs.dot = hexs(dot(a, b));
    rec.outputs.ata = hexs(a.' * a);
    rec.outputs.sumab = hexs(sum(a .* b));
    rec.outputs.rowcol = hexs(a.' * [b b]);
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
X = reshape(hex2num(cellstr(rec.hex)), sz);
end

function emit(out, rec)
fid = fopen(out, 'a');
fprintf(fid, '%s\n', jsonencode(rec));
fclose(fid);
end
