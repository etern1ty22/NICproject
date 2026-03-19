from __future__ import annotations

from nic_vrptw.core.models import ValidationReport, VRPTWInstance

from .loader import fingerprint_instance


def validate_instance(instance: VRPTWInstance) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    fingerprint = fingerprint_instance(instance)

    if instance.vehicle.count <= 0:
        errors.append("vehicle.count must be positive")
    if instance.vehicle.capacity <= 0:
        errors.append("vehicle.capacity must be positive")
    if not instance.customers:
        errors.append("instance must include at least one customer")

    node_ids = list(instance.node_ids)
    if len(node_ids) != len(set(node_ids)):
        errors.append("node_ids must be unique")
    if instance.depot_id not in node_ids:
        errors.append("depot_id must be present in node_ids")

    matrix_size = len(instance.distance_matrix)
    if matrix_size != len(node_ids):
        errors.append("distance matrix row count does not match node count")
    for row in instance.distance_matrix:
        if len(row) != len(node_ids):
            errors.append("distance matrix must be square")
            break

    customer_ids = set()
    for customer in instance.customers:
        if customer.customer_id in customer_ids:
            errors.append(f"duplicate customer id {customer.customer_id}")
        customer_ids.add(customer.customer_id)
        if customer.demand < 0:
            errors.append(f"customer {customer.customer_id} has negative demand")
        if customer.service_time < 0:
            errors.append(f"customer {customer.customer_id} has negative service time")
        if customer.time_window.start > customer.time_window.end:
            errors.append(f"customer {customer.customer_id} has inverted time window")

    is_symmetric = _is_symmetric(instance.distance_matrix)
    if instance.metadata.get("distance_mode") == "euclidean" and not is_symmetric:
        errors.append("euclidean instances must have symmetric distance matrices")
    if not is_symmetric and not instance.metadata.get("asymmetric_allowed", False):
        errors.append("asymmetric matrix is not allowed for this instance")
    if is_symmetric and instance.metadata.get("distance_mode") == "explicit":
        warnings.append("explicit matrix is symmetric; asymmetry support was not exercised")

    return ValidationReport(valid=not errors, errors=tuple(errors), warnings=tuple(warnings), fingerprint=fingerprint)


def _is_symmetric(matrix: tuple[tuple[float, ...], ...], tolerance: float = 1e-9) -> bool:
    size = len(matrix)
    for row in matrix:
        if len(row) != size:
            return False
    for i in range(size):
        for j in range(i + 1, size):
            if abs(matrix[i][j] - matrix[j][i]) > tolerance:
                return False
    return True
