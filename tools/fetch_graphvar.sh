#!/usr/bin/env bash
# Fetch GraphVar 2.03a (GPL-3) from NITRC at run time and verify it.
# The GraphVar source is never committed to this repository.
set -euo pipefail
dest="${1:?destination folder}"
url="https://www.nitrc.org/frs/download.php/24261/GraphVar_2.03a.zip"
want="81d6a889aa8ec5a4c4604edfd98f2388d71cbba8cc93ce3ca982b3309894fb88"
mkdir -p "$dest"
curl -fsSL --retry 5 --retry-delay 10 -o "$dest/GraphVar_2.03a.zip" "$url"
echo "$want  $dest/GraphVar_2.03a.zip" | sha256sum -c -
unzip -q "$dest/GraphVar_2.03a.zip" -d "$dest"
rm -rf "$dest/__MACOSX"
test -d "$dest/GraphVar_2.03a/src"
echo "GraphVar 2.03a ready at $dest/GraphVar_2.03a"
