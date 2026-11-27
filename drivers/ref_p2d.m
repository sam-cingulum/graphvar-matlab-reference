function ref_p2d(inDir, outDir)
% MM7 battery: C = A.'*B with A (K x M) and B (K x N) drawn from a Park-Miller stream (see tools/gen_p2d.py),
% every word of C. Also records whether an explicit transposed copy gives the same words ((A.'+0)*B: note
% copy_same) and, for M = 1 or N = 1, whether the product equals the column-by-column form (note colwise_same:
% each column j as A.'*B(:,j), or each row i as A(:,i).'*B). Writes outDir/tn.jsonl.
p = jsondecode(fileread(fullfile(inDir, 'tn.json')));
if ~isfolder(outDir), mkdir(outDir); end
out = fullfile(outDir, 'tn.jsonl');
if isfile(out), delete(out); end
u = @(X) typecast(X(:), 'uint64');
recs = p.tn;
if ~iscell(recs), recs = num2cell(recs); end
fid = fopen(out, 'a');
for k = 1:numel(recs)
    g = recs{k};
    v = stream(g.seed, g.K * (g.M + g.N));
    A = reshape(v(1:g.K * g.M), g.K, g.M);
    B = reshape(v(g.K * g.M + 1:end), g.K, g.N);
    C = A.' * B;
    rec = struct('case', g.case, 'M', g.M, 'N', g.N, 'K', g.K, 'notes', struct());
    rec.notes.copy_same = double(isequal(u(C), u((A.' + 0) * B)));
    if g.M == 1 || g.N == 1
        D = zeros(g.M, g.N);
        if g.M == 1
            for j = 1:g.N, D(1, j) = A.' * B(:, j); end
        else
            for i = 1:g.M, D(i, 1) = A(:, i).' * B; end
        end
        rec.notes.colwise_same = double(isequal(u(C), u(D)));
    end
    rec.C = cellstr(num2hex(C(:)))';
    fprintf(fid, '%s\n', jsonencode(rec));
end
fclose(fid);
end

function v = stream(seed, count)
v = zeros(count, 1);
s = seed;
for i = 1:count
    s = mod(16807 * s, 2147483647);
    v(i) = (s - 1073741823.5) / 1073741824;
end
end
