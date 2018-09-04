def ensure_python_bin_dir_in_path():
    """
    Ensure the bin/ dir containing the current python executable is in $PATH
    - This normally isn't necessary, but e.g. in a conda env started as an ipykernel, which means it's not running
      within the unix env created by the typical `activate <env>` idiom, if it has local bin/* commands installed, then
      they won't be found in the global $PATH -- or worse, some global variant will be found and silently used instead.
    """
    import os, sys
    python_bin_dir = os.path.dirname(sys.executable)
    if python_bin_dir not in os.environ['PATH'].split(':'):
        os.environ['PATH'] = '%s:%s' % (python_bin_dir, os.environ['PATH'])


def install_sigusr_hooks():
    """
    Install debugging hooks for SIGUSR1/SIGUSR2
    - Trigger manually, e.g. `kill -USR1 <pid>`
    - https://docs.python.org/3/library/faulthandler.html
    """
    import signal, faulthandler, pdb
    signal.signal(signal.SIGUSR1, lambda *args: faulthandler.dump_traceback())
    signal.signal(signal.SIGUSR2, lambda *args: pdb.set_trace())
