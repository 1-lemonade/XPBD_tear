$ErrorActionPreference = "Stop"
& conda run --no-capture-output -n MPM_couple python -B "$PSScriptRoot/src/tools/u4_joint.py" @args
exit $LASTEXITCODE
