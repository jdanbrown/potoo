"""
pytest helpers for atom-hydrogen-extras usage
"""

import sys

import _pytest
from IPython import get_ipython


# TODO Figure out a clean way to jump between tests
#   - pytest_runtest_setup works great for working within one test, but we'll need to incorporate
#     pytest_runtest_teardown to cleanly jump from one test to another without polluting the env
#   - In the meantime, we can just pytest_runtest_setup over and over and leave a mess of unused names as we go


# TODO Simplify usage:
#   - Invocation:
#       - Current usage: pytest_env(locals(), 'test_complist_happy_path_actives_above_and_below')
#       - Don't require user to pass locals()
#       - Don't require user to name the function they're in
#   - run_code ideas:
#       - Current happy path:
#           - Add '##' between function header and body (assuming a docstring to make function header standalone)
#           - Add `pytest_env(locals(), 'test_foo')` at top of body
#           - Add '##' below body
#           - Run all cells above, once
#           - Run current cell, repeatedly
#       - Add run-code-foo for function body / indent level / ..., instead of having to manually block it out with '##'
#       - Add run-code-for-current-pytest-function so user doesn't have to manually pytest_env(...)
def pytest_env(env: dict, test_fun_name: str):

    # Disable ipy %autoreload to avoid weird ipython reload bugs around failing to properly reload the current file
    #   - TODO Figure wth is going on here so we can re-enable `%autoreload 2`
    #   - I tried `%aimport -path.to.current_module`, but it didn't Just Work
    #   - To repro: follow dev loop above and modify/save the current test file somewhere in between two run-code calls
    print('Disabling %autoreload', file=sys.stderr)
    get_ipython().run_cell('%autoreload 0')

    # Collect all pytest test_* functions
    #   - From https://stackoverflow.com/a/46496720/397334
    cli_args = ['-s']
    config = _pytest.config._prepareconfig(cli_args)
    session = _pytest.main.Session(config)
    _pytest.tmpdir.pytest_configure(config)
    _pytest.fixtures.pytest_sessionstart(session)
    _pytest.runner.pytest_sessionstart(session)
    test_funs = session.perform_collect()

    # Get the requested test_* function, by name
    [test_fun] = [f for f in test_funs if f.name == test_fun_name]
    test_fun.funcargs

    # Run a pytest test_* function
    #   - Ref: https://github.com/pytest-dev/pytest/blob/71a7b3c/_pytest/hookspec.py
    #   - Example [https://github.com/pytest-dev/pytest/blob/71a7b3c/testing/python/collect.py#L612-L613]:
    #       test_fun.ihook.pytest_runtest_setup(item=test_fun)
    #       test_fun.ihook.pytest_pyfunc_call(pyfuncitem=test_fun)
    #   - Example:
    #       test_fun.ihook.pytest_runtest_setup(item=test_fun)
    #       test_fun.ihook.pytest_runtest_call(item=test_fun)
    #       test_fun.ihook.pytest_runtest_teardown(item=test_fun, nextitem=None)
    test_fun.ihook.pytest_runtest_setup(item=test_fun)

    # Load module into env
    #   - But don't overwrite stuff, e.g. __builtins__
    env.update({
        k: v for k, v in test_fun.parent.module.__dict__.items()
        if k not in env
    })

    # Resolve test_fun args from fixtures and load into env
    #   - Do go ahead and overwrite, for simplicity
    env.update(test_fun.funcargs)
