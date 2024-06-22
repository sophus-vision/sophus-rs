#!/bin/bash

set -x # echo on
set -e # exit on error

cargo publish -p sophus_core
cargo publish -p sophus_lie
cargo publish -p sophus_pyo3
cargo publish -p sophus_image
cargo publish -p sophus_sensor
cargo publish -p sophus_opt
cargo publish -p sophus_viewer
cargo publish -p sophus
