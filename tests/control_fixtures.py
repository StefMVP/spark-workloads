import copy
import json
from pathlib import Path

from profileplan import ACTIVITY_CLASSES


def authority(operation_id="operation-a", generation=1, resource="gpu:GPU-test-card-0001", state="sealed"):
    snapshot = json.loads((Path(__file__).parent / "fixtures/control-protocol-v1.json").read_text())["snapshot"]
    snapshot = copy.deepcopy(snapshot)
    snapshot["coverage"] = [{"resource": resource, "activity_class": kind, "status": "supported", "reason": ""} for kind in sorted(ACTIVITY_CLASSES)]
    snapshot["self_resource"]["resource_id"] = resource
    snapshot["self_resource"]["gpu_uuid"] = resource.removeprefix("gpu:")
    snapshot["fences"] = [{"operation_id": operation_id, "generation": generation, "resources": [resource], "state": state, "roots": []}]
    return {"state": "fresh", "expected": {key: snapshot[key] for key in ("instance_id", "birth", "revision")},
            "snapshot": snapshot, "binding": snapshot["self_resource"], "observed_at": 10, "resources": [resource]}
