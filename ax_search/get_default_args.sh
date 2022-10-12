#!/bin/zsh

args=$(grep '\-\-' training.py | cut -d',' -f1 | cut -d'"' -f2 | sed 's/\-\-//g' | sed 's/-/_/g')
values=$(grep '\-\-' training.py | cut -d',' -f3 | cut -d'=' -f2 | sed 's/)//g')
paste -d '=,' <(echo "$args") <(echo "$values") <(echo "")
