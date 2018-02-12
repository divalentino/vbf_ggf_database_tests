#!/bin/bash

message="${1}"

# Do cleanup.
rm *~

#Add new changes.
git add *.sh
git add *.py
git add README.md
git add output

git commit -m "${message}"
git push