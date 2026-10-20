#!/bin/bash
# Pediatric AML reference (Lambo et al. 2023 Cancer Cell), selective download:
#   GSE235308 = scATAC (peak matrices; the fragment files are skipped, ~96 GB of the 103 GB)
#   GSE235063 = scRNA  (processed matrices; the raw_* barcode/gene/matrix files are skipped)
# Cells are linked across the two series by Linked_scRNA_barcode in the ATAC metadata.
#
# Per-file URLs live under samples/<GSM prefix>nnn/<GSM>/suppl/, NOT under the series
# suppl/ directory: the series path 404s and, without --fail, silently writes 990-byte
# HTML error pages that look like successful downloads.
set -u
BASE=/mnt/10t/holiday/geo_gse235308
for GSE in GSE235308 GSE235063; do
  DIR="$BASE/$GSE"; mkdir -p "$DIR"
  curl -s --max-time 120 "https://ftp.ncbi.nlm.nih.gov/geo/series/${GSE:0:6}nnn/$GSE/suppl/filelist.txt" \
    > "$DIR/_filelist.txt"
  awk -F'\t' '$1=="File"{print $2"\t"$4}' "$DIR/_filelist.txt" \
    | grep -vE "_fragments\.tsv\.gz|_raw_(barcodes|genes|matrix)" > "$DIR/_want.txt"
  n=$(wc -l < "$DIR/_want.txt")
  tot=$(awk -F'\t' '{s+=$2}END{printf "%.1f", s/1073741824}' "$DIR/_want.txt")
  echo "[$GSE] $n files, ${tot} GB"
  i=0; bad=0
  while IFS=$'\t' read -r f sz; do
    i=$((i+1))
    gsm="${f%%_*}"                                  # GSM7498708_AML16_... -> GSM7498708
    url="https://ftp.ncbi.nlm.nih.gov/geo/samples/${gsm:0:7}nnn/$gsm/suppl/$f"
    if [ -f "$DIR/$f" ] && [ "$(stat -c%s "$DIR/$f")" = "$sz" ]; then continue; fi
    if ! curl -s --max-time 1800 --retry 3 --fail -o "$DIR/$f" "$url"; then
      echo "[$GSE] FAIL $f"; bad=$((bad+1)); rm -f "$DIR/$f"; continue
    fi
    got=$(stat -c%s "$DIR/$f" 2>/dev/null || echo 0)
    if [ "$got" != "$sz" ]; then
      echo "[$GSE] SIZE $f want $sz got $got"; bad=$((bad+1))
    fi
    if [ $((i % 25)) -eq 0 ]; then echo "[$GSE] $i/$n  $(du -sh "$DIR" | cut -f1)"; fi
  done < "$DIR/_want.txt"
  echo "[$GSE] done, $(du -sh "$DIR" | cut -f1) on disk, $bad problems"
done
echo ALL_DONE
