watch-img:
  qiv --scale_down --watch output/out.png

render:
  mkdir -p output
  cargo run --release -- -o output/out.png
