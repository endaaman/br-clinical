#!/usr/bin/env zx

let i = 0
while (i < 100) {
  await $`mv out20230410_${i}_t0/roc_181E.png tmp/rocs/roc${i}.png`
  i += 1
}
