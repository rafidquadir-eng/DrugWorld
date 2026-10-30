# D0 access and throughput (plan section 3, item 2)

Measured 2026-10-30 (21:40–22:30 UTC) from this instance (Lambda us-east-1) with curl 8.5.0
`--parallel`, writing to `/dev/null`. Raw outputs: `logs/d0/throughput_idr0119.txt`,
`logs/d0/throughput_idr0064.txt` and the per-request `tp*_out*.txt` files. idr0119 samples are
vehicle-well files only.

## Locations

| set | location | format |
|---|---|---|
| idr0119 | `https://ftp.ebi.ac.uk/pub/databases/IDR/idr0119-gross-cellresponse/` (EBI FTP over HTTPS); file paths from the 34 IDR companion OME-XML files | single-plane uint16 TIFF, 1408 x 1040, one file per channel and time point (1,748,979 files) |
| idr0119 | IDR OME-Zarr bucket (`uk1s3.embassy.ebi.ac.uk/idr/zarr/…/idr0119/`), BioImage Archive | **not available** (empty prefix; no BioStudies record) |
| idr0064 | BioImage Archive `S-BIAD992` (OME-NGFF), S3 `bia-integrator-data/S-BIAD992/<uuid>/<uuid>.zarr` (9 plates) | OME-Zarr v0.4, chunks (1, 1, 1, 512, 512) uint16, blosc-lz4 |
| idr0064 | `https://ftp.ebi.ac.uk/pub/databases/IDR/idr0064-goglia-erkdynamics/20190630-ftp/` | one 2-channel, 121-frame TIFF per well (127 MB), SHA-1 list `SHASUMS` |

## idr0119 (EBI FTP over HTTPS)

| streams | files | MB | seconds | MB/s | failed requests |
|---|---|---|---|---|---|
| 64 | 800 | 85.9 | 4.6 | 18.9 | 756 |
| 128 | 800 | 91.4 | 9.9 | 9.2 | 750 |
| 64 | 800 | 1331.9 | 21.5 | 62.0 | 69 |
| 128 | 800 | 95.5 | 17.2 | 5.6 | 749 |
| 8 | 400 | 731.8 | 91.7 | 8.0 | 0 |
| 16 | 400 | 708.9 | 59.7 | 11.9 | 0 |
| 24 | 400 | 672.4 | 23.2 | 28.9 | 0 |
| 32 | 400 | 650.4 | 17.2 | 37.7 | 42 |

- The failed requests are refused TCP connections ("Failed to connect to ftp.ebi.ac.uk port
  443"): the server limits concurrent connections per client. Each connection carries about
  1 MB/s. At 64 and 128 streams most requests are refused; the one 64-stream run with 69
  failures (62 MB/s) repeated files already requested in an earlier run and is not used for
  the estimate. The server answers HTTP/1.1 only (no HTTP/2 multiplexing); plain FTP (port 21)
  was slower (2.7–5.2 MB/s, most connections refused).
- Mean file size over 2,289 successful downloads: 1.79 MB, so the whole set is about
  1,748,979 x 1.79 MB ≈ **3.13 TB** (the plan's ~3.16 TB).
- **Estimated transfer:** at 24–32 streams with retries, 29–38 MB/s: **23–30 h** for all three
  channels; about **8–10 h** if only the phase channel is needed (channel mapping (i)).
  No workaround of the per-client limit is used (rule: never bypass access controls).

## idr0064

| route | streams | data | seconds | MB/s | errors |
|---|---|---|---|---|---|
| EBI FTP over HTTPS, whole TIFFs | 8 | 8 files, 1015 MB | 129.6 | 7.8 | 0 |
| | 24 | 24 files, 3046 MB | 180.9 | 16.8 | 0 |
| BIA S3 OME-Zarr chunks | 64 | 1,936 chunks, 996 MB | 25.3 | 39.4 | 0 |
| | 128 | the same 1,936 chunks, 996 MB | 9.7 | 102.5 | 0 |

The 128-stream run repeated the 64-stream run's chunks and may be helped by server caching.
Whole set: 450 wells x 127 MB ≈ **57 GB**, i.e. **about 10–25 min** from the BIA S3 bucket
(about 1 h over FTP-HTTPS).
