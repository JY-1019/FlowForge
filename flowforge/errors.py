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
    """Raised when execution of a node fails.

    Carries structured context to simplify debugging in deeply nested
    agent pipelines:

    * ``node_id`` — the DAG node that failed.
    * ``step_input`` — the input that was passed to the failing node.
    * ``partial_output`` — the last successful output before the failure
      (``None`` if the very first step failed).
    * ``trace_path`` — the full path from global → flow → task → step,
      built by the runner when the error propagates up.
    """

    def __init__(
        self,
        node_id: str,
        detail: str = "",
        *,
        step_input: object = None,
        partial_output: object = None,
        trace_path: list[str] | None = None,
    ) -> None:
        msg = f"Execution failed for node '{node_id}'"
        if detail:
            msg += f": {detail}"
        super().__init__(msg)
        self.node_id = node_id
        self.step_input = step_input
        self.partial_output = partial_output
        self.trace_path: list[str] = trace_path or [node_id]


class ApprovalRequired(FlowForgeError):
    """Raised when a step with ``approval=True`` needs human approval.

    The engine catches this, saves a checkpoint, and re-raises so the caller
    can inspect the pending input, approve or reject, and resume::

        try:
            result = await engine.run(data)
        except ApprovalRequired as e:
            print(f"Step {e.node_id} needs approval")
            print(f"Input: {e.step_input}")
            # After approval:
            result = await engine.run(data, resume_from=e.checkpoint)

    Attributes
    ----------
    node_id:
        DAG node ID of the step requesting approval.
    step_input:
        The validated input that would be passed to the step.
    checkpoint:
        A :class:`Checkpoint` snapshot that can be passed to
        ``engine.run(resume_from=...)`` after approval.
    """

    def __init__(
        self,
        node_id: str,
        step_input: object = None,
        checkpoint: object = None,
    ) -> None:
        super().__init__(
            f"Step '{node_id}' requires human approval before execution"
        )
        self.node_id = node_id
        self.step_input = step_input
        self.checkpoint = checkpoint


class ValidationError(FlowForgeError):
    """Raised when input/output validation fails."""


class DocGenerationError(FlowForgeError):
    """Raised when doc generation fails."""


class PlannerError(FlowForgeError):
    """Raised when the planner fails to produce a valid plan."""
