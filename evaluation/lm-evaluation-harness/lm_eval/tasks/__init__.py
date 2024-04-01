import logging
from typing import List, Mapping, Optional, Tuple, Type, Union

import lm_eval.api.utils
from lm_eval.api.task import Task
from promptsource.templates import DatasetTemplates

from . import taxi_dev, taxi_test, sib200_dev, sib200_test

logger = logging.getLogger(__name__)


TASK_REGISTRY = {
    # sib200
    "sib200_dev": sib200_dev.Sib200_dev,
    "sib200_test": sib200_test.Sib200_test,
    # taxi
    "taxi_dev": taxi_dev.Taxi_dev,
    "taxi_test": taxi_test.Taxi_test,
}


def list_tasks() -> List[str]:
    """Returns a list of all the available tasks by name."""
    return sorted(list(TASK_REGISTRY))


def get_task(task_name: str, template_name: str, **task_kwargs) -> Task:
    """Returns a task from the registry and instantiates it with the `promptsource`
    template specified by `template_name`.

    Args:
        task_name: Name of the task to load from the task registry.
        template_name: Name of the prompt template from `promptsource` to use
            for this task.
        **task_kwargs: Keyword arguments to pass to the task constructor. See constructor
            args for `lm_eval.api.task.Task`.

    Returns:
        A task instance with formatting specified by `template_name`.
    """
    task_class = _get_task_from_registry(task_name)
    template = get_templates(task_name)[template_name]
    return task_class(prompt_template=template, **task_kwargs)


def get_task_list(
    task_name: str, template_names: List[str], **task_kwargs
) -> List[Task]:
    """Returns a list of the same task but with multiple prompt templates.

    Args:
        task_name: Name of the task to load from the task registry.
        template_names: Name of the prompt template from `promptsource` to use
            for this task.
        **task_kwargs: Keyword arguments to pass to the task constructor. See constructor
            args for `lm_eval.api.task.Task`.

    Returns:
        A list of tasks with the same name but different prompt templates.
    """
    assert template_names, "Must specify at least one template name"
    template_names = sorted(set(template_names))
    return [get_task(task_name, t, **task_kwargs) for t in template_names]


def list_templates(task_name: str) -> List[str]:
    """Returns all template names available in `promptsource` for a given task."""
    templates = get_templates(task_name)
    return sorted(templates.all_template_names)


def get_templates(task_name: str) -> DatasetTemplates:
    """Returns the `promptsource` `DatasetTemplates` for the specified task name."""
    task_class = _get_task_from_registry(task_name)
    return _get_templates_from_task(task_class)


def get_task_list_from_args_string(
    task_name: str,
    template_names: List[str],
    task_args: str,
    additional_config: Optional[Mapping[str, str]] = None,
) -> List[Task]:
    """Returns a list of the same task but with multiple prompt templates, each
    task instantiated with the given kwargs.

    Args:
        task_name: Name of the task to use as found in the task registry.
        template_names: Name of the prompt template from `promptsource` to use
            for this task.
        task_args: A string of comma-separated key=value pairs that will be passed
            to the task constructor. E.g. "data_dir=./datasets,example_separator=\n\n"
        additional_config: An additional dictionary of key=value pairs that will
            be passed to the task constructor.

    Returns:
        A list of `Task` instances.
    """
    kwargs = lm_eval.api.utils.parse_cli_args_string(task_args)
    assert "prompt_template" not in kwargs, (
        "Cannot specify a `prompt_template` object in the `task_args` string. "
        "Only primitive type arguments are allowed."
    )
    additional_config = {} if additional_config is None else additional_config
    additional_args = {k: v for k, v in additional_config.items() if v is not None}
    kwargs.update(additional_args)
    return get_task_list(task_name, template_names, **kwargs)


# Helper functions


def _get_task_from_registry(task_name: str) -> Type[Task]:
    try:
        return TASK_REGISTRY[task_name]
    except KeyError:
        logger.warning(f"Available tasks:\n{list_tasks()}")
        raise KeyError(f"`{task_name}` is missing from the task registry.")


def _get_templates_from_task(task: Union[Task, Type[Task]]) -> DatasetTemplates:
    dataset_name = (
        task.DATASET_PATH
        if task.DATASET_NAME is None
        else f"{task.DATASET_PATH}/{task.DATASET_NAME}"
    )
    return DatasetTemplates(dataset_name)


# TODO(jon-tow): Refactor everything below! These functions are only required
# b/c the task registry is non-uniformly hard-coded.


# TODO(jon-tow): Remove this function after refactoring the task registry to use
# `Task` object __str__ representations for task names as opposed to
# hardcoded string keys.
def get_registry_name_from_task(task: Task) -> str:
    """Returns the task registry name from a Task instance."""
    for name, class_ in TASK_REGISTRY.items():
        if isinstance(task, class_):
            return name
    # This gives a mechanism for non-registered tasks to have a custom name anyways when reporting.
    return type(task).__name__


_TASK_TEMPLATE_KEY_SEP = "+"


def _get_task_template_key(task_name: str, template_name: str) -> str:
    """Returns a `str` key for a task with that prompt template name appended.
    This should be used to uniquely identify a task by its name AND
    its specific prompt template - as a task can have many templates.
    """
    if not template_name:
        # Add `null` prompt template to the key if no template name is specified.
        template_name = "null"
    return f"{task_name}{_TASK_TEMPLATE_KEY_SEP}{template_name}"


def _split_task_template_key(task_template_key: str) -> Tuple[str, str]:
    """Splits a task template key as returned from `_get_task_template_key`
    into it's constituent parts: (task name, template name).
    """
    task_name, template_name = task_template_key.split(_TASK_TEMPLATE_KEY_SEP, 1)
    return task_name, template_name
