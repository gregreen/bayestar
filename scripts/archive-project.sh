#!/bin/bash

export COMMIT_HASH=`git rev-parse HEAD`
git-archive --format=tar --output=$1 ${COMMIT_HASH}
echo ${COMMIT_HASH} > commit.txt
tar --append --file=$1 commit.txt
gzip -3 $1

echo "Project archived to $1.gz"
