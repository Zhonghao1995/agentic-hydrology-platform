#!/usr/bin/env bash
# Fetch the tiny CAMELS-US 4-basin sample that ships with neuralhydrology's
# tests into data/camels_us_sample/ (git-ignored). Real data, ~no download.
# Run from the repo root.
set -euo pipefail

NH="${NH_REPO:-/tmp/nh_repo}"
if [ ! -d "$NH" ]; then
  git clone --depth 1 https://github.com/neuralhydrology/neuralhydrology "$NH"
fi

SRC="$NH/test/test_data/camels_us"
DST="data/camels_us_sample"
mkdir -p "$DST/basin_mean_forcing"
for f in daymet maurer nldas; do
  cp -R "$SRC/basin_mean_forcing/$f" "$DST/basin_mean_forcing/"
done
cp -R "$SRC/usgs_streamflow" "$DST/"
cp -R "$SRC/camels_attributes_v2.0" "$DST/"
cp "$NH/test/test_data/4_basins_test_set.txt" "$DST/4_basins.txt"

echo "Prepared $DST ($(cat "$DST/4_basins.txt" | tr '\n' ' '))"
