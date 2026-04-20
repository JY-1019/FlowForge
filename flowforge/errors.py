"""Custom exceptions for FlowForge."""


class FlowForgeError(Exception):
    """Base exception for all FlowForge errors."""


class OrderConflictError(FlowForgeError):
    """Raised when two ``@step`` nodes inside a leaf task share the same order number."""

    def __init__(self, task_name: str, order: int) -> None:
        super().__init__(
            f"Task '{task_name}': duplicate order={order}. "
            "Each @step within a leaf task must have a unique order number."
        )
        self.task_name = task_name
        self.order = order


class IOBindingError(FlowForgeError):
    """Raised when consecutive nodes have incompatible I/O schemas."""

    def __init__(self, from_node: str, to_node: str, detail: str = "") -> None:
        msg = f"I/O binding error between '{from_node}' and '{to_node}'"
        if detail:
            msg += f": {detail}"
        super().__init__(msg)
        self.from_node = from_node
        self.to_node = to_node


class BranchOutputMismatchError(FlowForgeError):
    """Raised when branch handlers on a ``@step`` return different output types.

    All handler callables listed in a branching step's ``branches`` dict must
    share the same return-type annotation so that the next step in the chain
    always receives a consistent type.
    """

    def __init__(self, branch_name: str, detail: str = "") -> None:
        msg = (
            f"Branching step '{branch_name}': all handlers must return the "
            "same output schema"
        )
        if detail:
            msg += f": {detail}"
        super().__init__(msg)
        self.branch_name = branch_name


class CycleDetectedError(FlowForgeError):
    """Raised when a cycle is detected in the DAG."""

    def __init__(self, cycle: list[str]) -> None:
        super().__init__(f"Cycle detected in DAG: {' -> '.join(cycle)}")
        self.cycle = cycle


class CompileError(FlowForgeError):
    """General compilation error."""


class ExecutionError(FlowForgeError):
    """Raised when execution of a node fails."""

    def __init__(self, node_id: str, detail: str = "") -> None:
        msg = f"Execution failed for node '{node_id}'"
        if detail:
            msg += f": {detail}"
        super().__init__(msg)
        self.node_id = node_id


class ValidationError(FlowForgeError):
    """Raised when input/output validation fails."""


class DocGenerationError(FlowForgeError):
    """Raised when doc generation fails."""


class PlannerError(FlowForgeError):
    """Raised when the planner fails to produce a valid plan."""
