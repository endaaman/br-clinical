#!/usr/bin/env zx

await $`mkdir -p tmp/rocs_t1`

let i = 0
while (i < 100) {
  await $`mv out20230410_${i}_t1/roc_1E18.png tmp/rocs_t1/roc${i}.png`
  i += 1
}
