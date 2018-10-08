from contextlib import contextmanager
from typing import *

from IPython import get_ipython
from IPython.core.magic import Magics, cell_magic, line_cell_magic, line_magic, magics_class

from potoo import debug_print


def load_ipython_extension(ipy):
    DefaultMagicMagic.create(ipy)
    default_magic_magic = DefaultMagicMagic.get()
    ipy.register_magics(default_magic_magic)
    default_magic_magic.register()


@magics_class
class DefaultMagicMagic(Magics):
    """
    Magics:
    - %default_magic
    - %%py/%py
    """

    # WARNING This implementation was redone in the ipython 6.x -> 7.x bump when ipython overhauled the api for input
    # transformation, so there might still be some buggy or unnecessary bits from the 6.x impl

    @classmethod
    def create(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = cls(*args, **kwargs)

    @classmethod
    def get(cls):
        return cls._instance

    def __init__(self, shell):
        super().__init__(shell)
        self._default_magic = 'py'
        self._is_in_transform = False
        self._disabled = False

    def register(self):
        self._register_unregister(register=True)

    def unregister(self):
        self._register_unregister(register=False)

    def _register_unregister(self, register: bool):
        # debug_print(register=register)
        # Docs: http://ipython.readthedocs.io/en/stable/config/inputtransforms.html
        f = self.transform
        fs = get_ipython().input_transformers_cleanup  # Yep: this is the step before ipy transforms magics (and we want to inject one)
        # fs = get_ipython().input_transformers_post   # Nope: this is the step after ipy transforms magics
        if register:
            fs.append(f)
        else:
            while f in fs:
                fs.remove(f)

    def transform(self, lines: Iterable[str]):
        # If %default_magic is set, add it to each cell unless the cell already has a cell magic
        #   - Don't condition on the presence of line magics: some cell magics support line magics (e.g. %%time) and
        #     some don't (e.g. %%R), so stay out of the way instead of trying to guess what the user means
        #   - Wrap `%%foo` with `%%_in_transform foo` so that we can detect when we've already done our transform for
        #     a given cell to avoid bad outcomes like:
        #       - We transform 'code' to '%%foo\ncode' -> cellmagics transforms '%%foo\ncode' to
        #         ipy.run_cell_magic('foo', '', 'code') -> we get called again with just 'code' -> loop!
        #       - We transform '%line code' to '%%foo\n%line code' -> regardless of how we avoid looping on '%line code'
        #         we're still going to have to deal with ipy.run_line_magic('line', 'code') re-invoking us with 'code',
        #         at which point we should keep it as 'code', not transform it to '%%foo\ncode'
        # debug_print(lines)
        default_magic = self.get_default_magic()
        if lines and not self._is_in_transform:
            first_line = lines[0]
            has_user_cell_magic = first_line.startswith('%%')
            if has_user_cell_magic:
                lines = ['%%_in_transform\n', *lines]
            elif default_magic:
                lines = ['%%_in_transform\n', f'%%{default_magic}\n', *lines]
        return lines

    def get_default_magic(self):
        # debug_print(_disabled=self._disabled, _default_magic=self._default_magic)
        if self._disabled or self._default_magic == 'py':
            return None
        else:
            return self._default_magic

    @contextmanager
    def _with_in_transform(self):
        assert not self._is_in_transform, "Can't use %%_in_transform inside %%_in_transform"
        try:
            self._is_in_transform = True
            # debug_print('enter')
            yield
        finally:
            # debug_print('exit')
            self._is_in_transform = False

    @cell_magic
    def _in_transform(self, line, cell):
        # debug_print(line=line, cell=cell)
        assert not line
        with self._with_in_transform():
            get_ipython().run_cell(cell)

    @line_magic
    def default_magic(self, line):
        """
        Set a default magic to use when no line/cell magic is given, e.g.
            %default_magic time  # Run all lines/cells through %time/%%time (unless they have an explicit magic)
            %default_magic R     # Run all lines/cells through %R/%%R (unless they have an explicit magic)
            %default_magic py    # Run all lines/cells normally
            %default_magic       # Show the current default magic
        """
        if line:
            # debug_print(line=line)
            self._default_magic = line
        print('default_magic: %%%s' % self._default_magic)

    @line_cell_magic
    def py(self, line, cell=None):
        """
        Bypass %default_magic, as if it was unset

        This concept _could_ generalize to kernels other than ipykernel, but this implementation is tied to ipykernel since
        there's no generic "magics" mechanism across backend kernels -- kernels aren't all implemented in python, and e.g.
        IRkernel/ijavascript decided to not support parsing and executing magics at all.
        - https://irkernel.github.io/faq/ -> "We don't and won't support %%cell magic"
        - https://github.com/n-riesco/ijavascript/issues/71 -> "IJavascript avoid magics by design"
        - https://github.com/ipython/ipykernel/blob/dbe1730/ipykernel/zmqshell.py#L595 -> opts in to Magics/KernelMagics
        """
        # debug_print(line=line, cell=cell)
        if cell and line:
            raise ValueError(f'Line args[{line}] not supported in cell mode')
        with self._disable():
            get_ipython().run_cell(cell or line)

    @contextmanager
    def _disable(self):
        assert not self._disabled, "Can't use %py inside %%py"
        if not self._default_magic:
            # debug_print(passthru (no _default_magic)')
            yield
        else:
            try:
                # Disable so we don't trigger again (infinite recursion, etc.)
                self._disabled = True
                # debug_print('enter', _default_magic=self._default_magic)
                yield
            finally:
                # debug_print('exit', _default_magic=self._default_magic)
                self._disabled = False
