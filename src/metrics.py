# -*- coding: utf-8 -*-
"""[DEPRECATED in DMCA v1.0.4]
FinOps metrics (FTI/λ) are computed by SIDRCE. This module remains as a guard
so accidental in-DMCA usage fails loudly.
"""
class DMCAFinOpsRemoved(Exception):
    pass

def _removed(*_, **__):
    raise DMCAFinOpsRemoved(
        "FTI/λ were removed from DMCA v1.0.4. Export ops telemetry to SIDRCE."
    )

# Preserve public names to trigger explicit failure on old imports
FTIResult = _removed
calc_fti = _removed
lambda_effective = _removed
norm_perf = _removed
norm_cost = _removed
