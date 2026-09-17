"""Give pinned Harbor's Terminus agent a concise completion instruction.

This is loaded only in Harbor subprocesses launched by terminal_bench.py.
The stored Harbor job config and vendored task instructions stay untouched.
"""

try:
    from harbor.agents.terminus_2.terminus_2 import Terminus2
except ImportError:
    # The docker-compatible shim may start a Python interpreter without Harbor.
    pass
else:
    _original_init = Terminus2.__init__
    _completion_instruction = (
        "When the task is complete, finish immediately. Keep analysis and plan "
        "brief; return commands as an empty array and task_complete as true. "
        "Do not run optional checks or explain the completed work further.\n\n"
    )

    def _init_with_completion_instruction(self, *args, **kwargs):
        _original_init(self, *args, **kwargs)
        if self._parser_name == "json":
            marker = "Task Description:\n"
            if marker not in self._prompt_template:
                raise RuntimeError("Harbor's Terminus prompt layout changed")
            self._prompt_template = self._prompt_template.replace(
                marker, _completion_instruction + marker, 1
            )

    Terminus2.__init__ = _init_with_completion_instruction
