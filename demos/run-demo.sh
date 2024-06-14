#!/bin/sh

WHEREAMI="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

perl -I"${WHEREAMI}/../blib/lib/" \
     -I"${WHEREAMI}/../blib/lib/auto/share/dist" \
  "${WHEREAMI}/mandelbrot-marioroy/mandelbrot.pl"
