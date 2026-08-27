#!/bin/bash
# $1 = source script, $2 = dest ; rewrite only the two absolute cluster paths
S=/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/e1d07839-2376-4458-891b-01d0a9e0433b/scratchpad
sed -e "s#/data/stat-cadd/scat9264/bin/micromamba#$S/vv/bin/micromamba#" \
    -e "s#/data/stat-cadd/scat9264/KIRBy#$S/vv/fakeroot/data/stat-cadd/scat9264/KIRBy#" \
    -e "s#/data/stat-cadd/scat9264/qsar_qm_models#$S/vv/fakeroot/data/stat-cadd/scat9264/qsar_qm_models#" \
    "$1" > "$2"
chmod +x "$2"
