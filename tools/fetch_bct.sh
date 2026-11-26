#!/usr/bin/env bash
# Fetch the official Brain Connectivity Toolbox release 2019-03-03 (GPL) from
# the download linked on the BCT website, and verify it. Never committed.
set -euo pipefail
dest="${1:?destination folder}"
url="https://drive.usercontent.google.com/download?id=1DmMvRnferBfGe057O-sZwB5jL4j8w1Hu&export=download&confirm=t"
want="c6468c8cdc1e477064ec94cd62db5af5cfe5b03ea6311be236834eebb936a6ce"
mkdir -p "$dest"
curl -fsSL --retry 5 --retry-delay 10 -o "$dest/2019_03_03_BCT.zip" "$url"
echo "$want  $dest/2019_03_03_BCT.zip" | sha256sum -c -
unzip -q "$dest/2019_03_03_BCT.zip" -d "$dest"
test -f "$dest/2019_03_03_BCT/charpath.m"
echo "BCT 2019-03-03 ready at $dest/2019_03_03_BCT"
