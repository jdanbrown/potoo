import secrets

from IPython import get_ipython
from IPython.core.magic import Magics, cell_magic, magics_class


@magics_class
class LPRunCellMagic(Magics):

    @cell_magic
    def lprun(self,
        line,  # The '-f func1 -f func2' part of %lprun
        cell,  # The code part of %lprun, except it's allowed to be multiple lines (i.e. a cell body)
    ):
        """
        A cell version of %lprun.

        Usage:
            %%lprun -f func1 -f func2
            <statement>
            ...

        See %lprun? for more info.
        """

        # Create a function that runs the cell body
        def f():
            exec(cell, self.shell.user_ns)

        # Call it in %lprun by injecting it into the user's env
        f_name = '__lprun_cell_magic_%s' % secrets.token_hex(4)  # Use a fresh name, to avoid cloberring anything
        self.shell.user_ns.update({f_name: f})
        try:
            get_ipython().run_line_magic('lprun', '%s %s()' % (line, f_name))
        finally:
            del self.shell.user_ns[f_name]  # Don't leave it in the user's env


def load_ipython_extension(ipy):
    ipy.register_magics(LPRunCellMagic)
