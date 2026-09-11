from __future__ import annotations

import importlib
from collections import defaultdict, deque

from app.core.module import Module, PipelineStep


class ModuleRegistry:
    def __init__(self, modules: list[Module]) -> None:
        self.modules = modules
        self._by_name = {module.name: module for module in modules}

    @classmethod
    def discover(cls, module_names: list[str]) -> ModuleRegistry:
        loaded = []
        for name in module_names:
            package = importlib.import_module(f"app.modules.{name}")
            loaded.append(package.module)
        registry = cls(loaded)
        registry._validate_dependencies()
        registry.modules = registry._sort_modules()
        return registry

    def _validate_dependencies(self) -> None:
        available = set(self._by_name)
        for module in self.modules:
            missing = set(module.depends_on) - available
            if missing:
                joined = ", ".join(sorted(missing))
                raise ValueError(f"Module {module.name} is missing dependencies: {joined}")

    def _sort_modules(self) -> list[Module]:
        indegree = {module.name: 0 for module in self.modules}
        graph: dict[str, list[str]] = defaultdict(list)
        for module in self.modules:
            for dependency in module.depends_on:
                graph[dependency].append(module.name)
                indegree[module.name] += 1
        queue = deque(sorted(name for name, degree in indegree.items() if degree == 0))
        ordered = []
        while queue:
            name = queue.popleft()
            ordered.append(self._by_name[name])
            for child in sorted(graph[name]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if len(ordered) != len(self.modules):
            raise ValueError("Module dependency cycle detected")
        return ordered

    def pipeline_steps(self) -> list[PipelineStep]:
        return sorted(
            [step for module in self.modules for step in module.pipeline_steps()],
            key=lambda step: step.order,
        )

    async def startup(self, ctx) -> None:
        for module in self.modules:
            await module.on_startup(ctx)

    async def shutdown(self, ctx) -> None:
        for module in reversed(self.modules):
            await module.on_shutdown(ctx)
