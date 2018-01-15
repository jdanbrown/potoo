import argparse
import inspect
import time

import datalab.bigquery as bq
from IPython.core.magic import Magics, line_cell_magic, magics_class
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring
import pandas as pd
import sqlalchemy as sqla
from traitlets.config.configurable import Configurable
from traitlets import Bool, Int, Unicode


# TODO How to generically overlay %config defaults + %foo opts? We currently do it manually everywhere.


@magics_class
class SQLMagics(Magics, Configurable):

    # Traits (set via %config -- affects subclasses)
    quiet = Bool(False, help="Suppress output to stdout").tag(config=True)

    def __init__(self, shell):
        self.shell = shell
        self.shell.configurables.append(self)
        Configurable.__init__(self, config=shell.config)
        Magics.__init__(self, shell=shell)

    def _print(self, quiet, *args, **kwargs):
        if not (quiet or self.quiet):
            print(*args, **kwargs)


@magics_class
class SQLAMagics(SQLMagics):

    # %sqla traits (set via %config)
    #   - conn must be a string because trailets tries and fails to pickle a sqla Session object
    conn = Unicode(None, allow_none=True).tag(config=True)

    @line_cell_magic
    @magic_arguments()
    @argument('-o', '--out', help='Variable name to store the output in')
    @argument('-q', '--quiet', action='store_true', help='Suppress output to stdout')
    @argument('-c', '--conn', help='sqlalchemy connection/session to use')
    @argument('rest', nargs=argparse.REMAINDER)
    def sqla(self, line, cell=None) -> pd.DataFrame:

        # Parse input
        args = parse_argstring(self.sqla, line)

        # Parse code
        code = ' '.join(args.rest + [cell or ''])

        # Get db session
        db_session = self.shell.user_ns[args.conn or self.conn]
        db_desc = repr(db_session.session_factory.kw['bind'].url)  # repr masks the password, str doesn't

        # Run query
        self._print(args.quiet, 'Running query...')
        start_s = time.time()
        df = pd.read_sql(
            sql=sqla.text(code),
            con=db_session.bind,
            coerce_float=True,  # True is default -- is this sane?
        )
        self._print(args.quiet, '[%.0fs, %s]' % (time.time() - start_s, db_desc))

        # Store output
        if args.out:
            self.shell.user_ns[args.out] = df

        return df


@magics_class
class BQMagics(SQLMagics):

    # Traits (set via %config)
    #   - TODO Any way to avoid replicating arg names, arg types, and default values?
    #   - TODO Add project_id (via bq.Query(sql, context=Context(project_id, credentials))
    #   - bq.Query.execute
    table_name = Unicode(None, allow_none=True).tag(config=True)
    table_mode = Unicode('create').tag(config=True)
    use_cache = Bool(True).tag(config=True)
    priority = Unicode('interactive').tag(config=True)
    allow_large_results = Bool(False).tag(config=True)
    dialect = Unicode('standard').tag(config=True)  # Impose sane default (lib default is 'legacy')
    billing_tier = Int(None, allow_none=True).tag(config=True)  # None means use project default
    #   - bq.QueryResultsTable.to_dataframe
    start_row = Int(0).tag(config=True)
    max_rows = Int(None, allow_none=True).tag(config=True)

    @line_cell_magic
    @magic_arguments()
    @argument('-o', '--out', help='Variable name to store the output in')
    @argument('-q', '--quiet', action='store_true', help='Suppress output to stdout')
    # TODO Any way to avoid replicating arg names and arg types?
    #   - bq.Query.execute
    @argument('--table_name', type=str, default=argparse.SUPPRESS)
    @argument('--table_mode', type=str, default=argparse.SUPPRESS)
    @argument('--use_cache', type=bool, default=argparse.SUPPRESS)
    @argument('--priority', type=str, default=argparse.SUPPRESS)
    @argument('--allow_large_results', type=bool, default=argparse.SUPPRESS)
    @argument('--dialect', type=str, default=argparse.SUPPRESS)
    @argument('--billing_tier', type=int, default=argparse.SUPPRESS)
    #   - bq.QueryResultsTable.to_dataframe
    @argument('--start_row', type=int, default=argparse.SUPPRESS)
    @argument('--max_rows', type=int, default=argparse.SUPPRESS)
    @argument('rest', nargs=argparse.REMAINDER)
    def bq(self, line, cell=None) -> pd.DataFrame:

        # Parse args
        args = parse_argstring(self.bq, line)
        args_dict = dict(args._get_kwargs())
        execute_kwargs = {
            k: args_dict.get(k, getattr(self, k))
            for k in inspect.signature(bq.Query.execute).parameters.keys()
            if k in args_dict or hasattr(self, k)
        }
        to_dataframe_kwargs = {
            k: args_dict.get(k, getattr(self, k))
            for k in inspect.signature(bq.QueryResultsTable.to_dataframe).parameters.keys()
            if k in args_dict or hasattr(self, k)
        }

        # Parse code
        code = ' '.join(args.rest + [cell or ''])
        code = code.replace('$', '$$')  # Make '$' safe by assuming no variable references (see bq.Query? for details)

        # Run query
        self._print(args.quiet, 'Running query...')
        start_s = time.time()
        query = bq.Query(code).execute(**execute_kwargs)
        self._print(args.quiet, '[%.0fs, %s]' % (time.time() - start_s, bq_url_for_query(query)))

        # Fetch results
        df = query.results.to_dataframe(**to_dataframe_kwargs)

        # Store output
        if args.out:
            self.shell.user_ns[args.out] = df

        return df


def bq_url_for_query(query: bq.QueryJob) -> str:
    return 'https://bigquery.cloud.google.com/results/%s:%s' % (query.results.name.project_id, query.results.job_id)


def load_ipython_extension(ipy):
    ipy.register_magics(SQLMagics)
    ipy.register_magics(SQLAMagics)
    ipy.register_magics(BQMagics)
