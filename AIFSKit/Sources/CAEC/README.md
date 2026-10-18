# CAEC — vendored libaec 1.1.7 (decoder only)

ECMWF open data is GRIB2 with CCSDS (AEC) compression (`packingType = grid_ccsds`,
template 5.42). These are the decoder sources of [libaec](https://github.com/MathisRosenhauer/libaec)
v1.1.7, unmodified apart from `include/libaec.h` (version macros filled in) and a
hand-written `config.h`. BSD-2-Clause, see `LICENSE.txt`.
