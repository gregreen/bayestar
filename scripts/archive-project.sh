#!/bin/bash
#
# Creates a tarball containing the Bayestar repository, along
# with a file called "commit.txt," containing the git hash of
# the current commit.
# 
# Invocation:
#   bash archive-project <output.tar>
# 

export COMMIT_HASH=`git rev-parse HEAD`
git archive --format=tar --output=${1}.tar ${COMMIT_HASH}
echo ${COMMIT_HASH} > commit.txt
tar --append --file=${1}.tar commit.txt
gzip -3 ${1}.tar

echo "Project archived to ${1}.tar.gz"
