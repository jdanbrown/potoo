def install_sigusr_hooks():
    """
    Install debugging hooks for SIGUSR1/SIGUSR2
    - Trigger manually, e.g. `kill -USR1 <pid>`
    - https://docs.python.org/3/library/faulthandler.html
    """
    import signal, faulthandler, pdb
    signal.signal(signal.SIGUSR1, lambda *args: faulthandler.dump_traceback())
    signal.signal(signal.SIGUSR2, lambda *args: pdb.set_trace())
