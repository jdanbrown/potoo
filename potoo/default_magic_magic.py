from contextlib import contextmanager
import json

from IPython import get_ipython
from IPython.core.magic import Magics, cell_magic, line_cell_magic, line_magic, magics_class
from IPython.core.inputtransformer import InputTransformer
from IPython.core.splitinput import LineInfo

from potoo.util import singleton


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

    @classmethod
    def create(cls, *args, **kwargs):
        assert not hasattr(cls, '_instance')
        cls._instance = cls(*args, **kwargs)

    @classmethod
    def get(cls):
        return cls._instance

    def __init__(self, shell):
        super().__init__(shell)
        self._default_magic = 'py'
        self._disabled = False
        self._in_has_cell_magic = False

    def register(self):
        self._register_unregister(register=True)

    def unregister(self):
        self._register_unregister(register=False)

    def _register_unregister(self, register: bool):
        _debug(f"DefaultMagicMagic.{'register' if register else 'unregister'}")
        # Docs: http://ipython.readthedocs.io/en/stable/config/inputtransforms.html
        ipy = get_ipython()
        for input_splitter in [
            # ipy.input_splitter,  # XXX Omit this entirely, else cli ipython line parsing gets busted
            ipy.input_transformer_manager,
        ]:
            if register:
                # Insert before the 'cellmagic' transformer, but after the other transformers
                #   - Before the 'cellmagic' transformer so that we can noop on cell magics
                #   - After the other transformers so we can assume that leading space and ipython prompts are stripped
                transformer = DefaultMagicTransformer(self)
                i_cellmagic = -1
                assert ipy.input_splitter.physical_line_transforms[i_cellmagic].coro.gi_code.co_name == 'cellmagic'
                input_splitter.physical_line_transforms.insert(i_cellmagic, transformer)
            else:
                input_splitter.physical_line_transforms = [
                    x for x in input_splitter.physical_line_transforms
                    if not isinstance(x, DefaultMagicTransformer)
                ]

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
            self._default_magic = line
            _debug('DefaultMagicMagic.default_magic', _default_magic=self._default_magic)
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
        _debug('DefaultMagicMagic.py', line=line, cell=cell)
        if cell and line:
            raise ValueError(f'Line args[{line}] not supported in cell mode')
        with self._disable():
            get_ipython().run_cell(cell or line)

    @contextmanager
    def _disable(self):
        assert not self._disabled, "Can't use %py inside %%py"
        if not self._default_magic:
            _debug('DefaultMagicMagic._disable: passthru (no _default_magic)')
            yield
        else:
            try:
                # Disable so we don't trigger again (infinite recursion, etc.)
                self._disabled = True
                _debug('DefaultMagicMagic._disable: enter', _default_magic=self._default_magic)
                yield
            finally:
                _debug('DefaultMagicMagic._disable: exit', _default_magic=self._default_magic)
                self._disabled = False

    def get_default_magic(self):
        # _debug('DefaultMagicMagic.get_default_magic', _disabled=self._disabled, _default_magic=self._default_magic)
        if self._disabled or self._default_magic == 'py':
            return None
        else:
            return self._default_magic

    @cell_magic
    def _has_cell_magic(self, line, cell):
        _debug('DefaultMagicMagic._has_cell_magic', line=line, cell=cell)
        assert not line
        with self._with_has_cell_magic():
            get_ipython().run_cell(cell)

    @contextmanager
    def _with_has_cell_magic(self):
        assert not self._in_has_cell_magic, "Can't use %%_has_cell_magic inside %%_has_cell_magic"
        try:
            self._in_has_cell_magic = True
            _debug('DefaultMagicMagic._with_has_cell_magic: enter')
            yield
        finally:
            _debug('DefaultMagicMagic._with_has_cell_magic: exit')
            self._in_has_cell_magic = False


class DefaultMagicTransformer(InputTransformer):

    def __init__(self, parent):
        self.parent = parent
        self._lines = []

    def push(self, line):
        # If %default_magic is set, add it to each cell unless the cell already has a cell magic
        #   - Don't condition on the presence of line magics: some cell magics support line magics (e.g. %%time) and
        #     some don't (e.g. %%R), so stay out of the way instead of trying to guess what the user means
        #   - Wrap `%%foo` with `%%_has_cell_magic foo` so that we can detect when we've already done our transform for
        #     a given cell to avoid bad outcomes like:
        #       - We transform 'code' to '%%foo\ncode' -> cellmagics transforms '%%foo\ncode' to
        #         ipy.run_cell_magic('foo', '', 'code') -> we get called again with just 'code' -> loop!
        #       - We transform '%line code' to '%%foo\n%line code' -> regardless of how we avoid looping on '%line code'
        #         we're still going to have to deal with ipy.run_line_magic('line', 'code') re-invoking us with 'code',
        #         at which point we should keep it as 'code', not transform it to '%%foo\ncode'
        is_first_line = not self._lines
        default_magic = self.parent.get_default_magic()
        has_cell_magic = line.startswith('%%')
        if is_first_line:
            if not self.parent._in_has_cell_magic:
                if has_cell_magic:
                    self._lines.append('%%_has_cell_magic')
                elif default_magic:
                    self._lines.append('%%_has_cell_magic')
                    self._lines.append(f'%%{default_magic}')
        self._lines.append(line)
        _debug('DefaultMagicTransformer.push', line=line, lines=self._lines)
        return None

    def reset(self):
        if self._lines:
            _debug('DefaultMagicTransformer.reset', lines=self._lines)
        out = '\n'.join(self._lines)
        self._lines = []
        return out


def _debug(event, **kwargs):
    from IPython.display import display
    try:
        import crayons
    except:
        color = lambda x: x
    else:
        color = crayons.black
    # print(color(f'[{event}]{"" if not kwargs else " " + ipy_format(kwargs)}'))


def ipy_format(x: any) -> str:
    """Format like IPython.display.display"""
    formats, metadata = get_ipython().display_formatter.format(x)
    return formats['text/plain']
