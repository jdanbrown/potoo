from IPython.core.getipython import get_ipython
import numpy as np
import pandas as pd

from potoo.util import singleton


@singleton
class formats:

    def __init__(self):
        self._reset_stack = []

    def reset(self):
        for f in reversed(self._reset_stack):
            f()
        self._reset_stack.clear()

    # TODO Fix to respect display.max_rows (currently treats it as unlimited)
    # pandas: Render lists in df cells as multiple lines
    # - https://www.safaribooksonline.com/blog/2014/02/11/altering-display-existing-classes-ipython/
    def df_stack_arrays(self):
        """
        When displaying df's as html, display arrays within cells as vertically stacked values (like bq web UI)
        - http://ipython.readthedocs.io/en/stable/api/generated/IPython.core.formatters.html
        """
        ipy = get_ipython()
        if ipy:

            prev_html = ipy.display_formatter.formatters['text/html'].for_type(pd.DataFrame, lambda df:
                df.apply(axis=0, func=lambda col:
                    col.apply(lambda x:
                        (lambda y: y if self._has_number(x) else '<div class="not-number">%s</div>' % y)(
                            x if not isinstance(x, list)
                            else '' if len(x) == 0
                            else pd.Series(x).to_string(index=False).replace('\n', '<br/>')
                        )
                    )
                ).to_html(
                    escape=False,  # Allow html in cells
                )
            )
            self._reset_stack.append(lambda:
                ipy.display_formatter.formatters['text/html'].for_type(pd.DataFrame, prev_html) if prev_html
                else ipy.display_formatter.formatters['text/html'].pop(pd.DataFrame)
            )

            # TODO Have to self.foo(...) instead of returning...
            # prev_plain = ipy.display_formatter.formatters['text/plain'].for_type(pd.DataFrame, lambda df, self, cycle:
            #     df.apply(axis=0, func=lambda col:
            #         col.apply(lambda x:
            #             x if not isinstance(x, list)
            #             else '' if len(x) == 0
            #             else pd.Series(x).to_string(index=False)
            #         )
            #     ).to_string()
            # )
            # self._reset_stack.append(lambda:
            #     ipy.display_formatter.formatters['text/plain'].for_type(pd.DataFrame, prev_plain) if prev_plain
            #     else ipy.display_formatter.formatters['text/plain'].pop(pd.DataFrame)
            # )

    def _has_number(self, x):
        return (
            np.issubdtype(type(x), np.number) or
            isinstance(x, list) and any(self._has_number(y) for y in x)
        )
