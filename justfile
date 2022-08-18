watch-img:
  qiv --scale_down --watch tmp/out.png

render:
  mkdir -p tmp
  cargo run --release -- -o tmp/out.png

render-all:
  #!/usr/bin/env bash

  set -ueo pipefail

  mkdir -p output
  for (( i=0; i<3; i++)); do
    cargo run --release -- --size 100 --render "$i" --output "output/out_$i.png"
  done
  for (( i=3; i<22; i++)); do
    cargo run --release -- --size 1000 --render "$i" --output "output/out_$i.png"
  done

