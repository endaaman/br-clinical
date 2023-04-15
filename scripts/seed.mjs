#!/usr/bin/env zx

let i = 0
while (i < 100) {
  await $`python main.py gbm --seed ${i}`
  i += 1
}
