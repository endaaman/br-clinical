#!/usr/bin/env zx

const thres = 1
let i = 0

while (i < 100) {
  await $`python main.py gbm --seed ${i} --thres ${thres}`
  i += 1
}
