function nm_split_local(file, dir, from, to)
% Write each local function of file (all but the first) to dir/<name>.m. With from/to, replace the text
% from by to first (used to swap graphallshortestpaths for a named replacement).
if ~isfolder(dir), mkdir(dir); end
txt = fileread(file);
if nargin >= 4, txt = strrep(txt, from, to); end
lines = regexp(txt, '\r?\n', 'split');
starts = find(~cellfun(@isempty, regexp(lines, '^\s*function\s', 'once')));
for k = 2:numel(starts)
    a = starts(k);
    if k < numel(starts), b = starts(k + 1) - 1; else, b = numel(lines); end
    tok = regexp(lines{a}, 'function\s+(?:\[[^\]]*\]\s*=\s*|\w+\s*=\s*)?(\w+)', 'tokens', 'once');
    fid = fopen(fullfile(dir, [tok{1} '.m']), 'w');
    fprintf(fid, '%s\n', lines{a:b});
    fclose(fid);
end
end
