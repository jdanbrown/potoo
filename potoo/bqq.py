# TODO Throw away potoo.bq and rename this to replace it (potoo.bqq -> potoo.bq)

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
import inspect
import logging
import random
import re
import textwrap
import time
from typing import Callable, List, Mapping, Optional, Tuple, Union

import datalab
import datalab.bigquery as bq
import datalab.storage as gs
from datalab.bigquery._utils import TableName
import gcsfs  # XXX Replace with pd.read_json in pandas 0.24.x [https://stackoverflow.com/a/50201179/397334]
import humanize
from more_itertools import flatten, intersperse
import pandas as pd

from potoo import humanize
from potoo.dataclasses import *
from potoo.util import puts

log = logging.getLogger(__name__)


# TODO Re-home?
def lines(*xs):
    return '\n'.join(x for x in xs if x is not None)
def indent_lines(*xs):
    return indent(lines(*xs))
def indent(s, width=4):
    return textwrap.indent(s, prefix=' ' * width)
def strip_margin(s):
    return textwrap.dedent(s).strip()
def fold_whitespace(s):
    return re.sub(r'\s+', ' ', s)
def strip_trailing_comma(s):
    # FIXME Breaks if trailing comment is inside a comment
    #   - Any good way to fix this without understanding the sql grammar?
    #   - e.g. trying to detect (or strip) comments will fail if we don't understand string quoting
    return re.sub(r',(\s*)$', r'\1', s)
def drop_lines(n, s):
    return '\n'.join(s.split('\n')[n:])


def bq_url_for_query(query: bq.QueryJob) -> str:
    return 'https://console.cloud.google.com/bigquery?project=%s&j=bq:US:%s&page=queryresults' % (query.results.name.project_id, query.results.job_id)


@dataclass
class BQQ(DataclassUtil, DataclassAsDict):

    # Global fields
    defs:         Optional[str] = field(default=None, metadata={'bqq.global': True})
    fresh_prefix: str           = field(default='_',  metadata={'bqq.global': True})
    fresh_len:    int           = field(default=4,    metadata={'bqq.global': True})

    # Local fields
    clauses: Mapping[str, any] = field(default_factory=lambda: {})
    _name:   Optional[str]     = None

    @property
    def no_defs(self):
        return self.replace(defs=None)

    @property
    def no_locals(self):
        cls = type(self)
        return cls(**self._globals())

    @property
    def name(self):
        if self._name is None:
            self._name = self._fresh_name()
        return self._name

    def ensure_name(self):
        self.name
        return self

    def named(self, name):
        if self._name is not None and self._name != name:
            raise ValueError(f"Can't overwrite existing name[{self._name}] (got new name[{name}])")
        return self.replace(_name=name)

    def __repr__(self):
        return lines(
            f"{type(self).__name__}(",
            indent_lines(
                f"fresh_prefix={self.fresh_prefix!r},",
                f"fresh_len={self.fresh_len!r},",
                f"defs={self.defs!r}," if not self.defs else lines(
                    f"defs='''",
                    indent_lines(strip_margin(self.defs)) or None,
                    "''',",
                ),
                f"_name={self._name!r},",
                f"sql={self.sql!r}," if not self.sql else lines(
                    f"sql='''",
                    indent_lines(strip_margin(self.sql)) or None,
                    "''',",
                ),
            ),
            ')',
        )

    # Simple api

    def __call__(self, query: str, **kwargs) -> pd.DataFrame:
        """Run given query (ignoring .sql), returning a pandas df"""
        return _bqq(query, **{
            'defs': None if self.defs is None else strip_margin(self.defs),
            **kwargs,
        })

    # Complex api
    #   - https://cloud.google.com/bigquery/docs/reference/standard-sql/query-syntax

    Q = Union['BQQ', str]

    def df(self, **kwargs) -> pd.DataFrame:
        """Run query from .sql, returning a pandas df"""
        if self.sql is None:
            raise ValueError('self.sql is None')
        return self(self.sql, **kwargs)

    @property
    def sql(self) -> str:
        sql = lines(
            self._if_clause('with', lambda qs:
                strip_trailing_comma(
                    lines('with', '', *[
                        indent_lines(f'{q.name} as (', indent(self._sql(q)), '),', '')
                        for q in qs
                    ]),
                ),
            ),
            self._if_clause('subquery',           lambda q:  lines('(', indent(self._sql(q)), ')')),
            self._if_clause('union_all',          lambda qs: lines(*intersperse('union all',          [self._sql(q) for q in qs]))),
            self._if_clause('union_distinct',     lambda qs: lines(*intersperse('union distinct',     [self._sql(q) for q in qs]))),
            self._if_clause('intersect_distinct', lambda qs: lines(*intersperse('intersect distinct', [self._sql(q) for q in qs]))),
            self._if_clause('except_distinct',    lambda qs: lines(*intersperse('except distinct',    [self._sql(q) for q in qs]))),
            self._get_clause('select', lambda s:
                strip_trailing_comma(
                    # Allow no `select` -> use `select *`
                    'select *' if s is None else
                    # Ergonomics for one-line select (semantically equiv)
                    f'select {strip_margin(s)}' if '\n' not in s else
                    # Ergonomics for `select distinct` (semantically equiv)
                    lines('select distinct', indent(strip_margin(drop_lines(1, s)))) if s.startswith('distinct\n') else
                    # Else normal select
                    lines('select', indent(strip_margin(s)))
                )
            ),
            self._if_clause('from_',              lambda s:  f'from {strip_margin(s)}'),
            self._if_clause('where',              lambda s:
                # Ergonomics for `where true\n... and ...` (semantically equiv)
                lines('where true', indent(strip_margin(drop_lines(1, s)))) if s.startswith('true\n') else
                # Ergonomics for `where false\n... or ...` (semantically equiv)
                lines('where false', indent(strip_margin(drop_lines(1, s)))) if s.startswith('false\n') else
                # Else normal where
                f'where {strip_margin(s)}'
            ),
            self._if_clause('group_by',           lambda s:  f'group by {strip_margin(s)}'),
            self._if_clause('having',             lambda s:  f'having {strip_margin(s)}'),
            self._if_clause('window',             lambda s:  f'window {strip_margin(s)}'),
            self._if_clause('order_by',           lambda s:  f'order by {strip_margin(s)}'),
            self._if_clause('limit',              lambda x:  f'limit {strip_margin(str(x))}'),  # Allow int
        )
        return sql or None

    # Alternate api, which is maybe more ergonomic would avoid needing .nest() and maybe be simpler
    #   - Makes clear where the query boundaries are
    #   - Doesn't allow invalid repetitions of the fluent-style clauses
    #   - Avoids .nest()
    #   - TODO Kill the fluent api so that .query() is the only api
    #       - Think harder about .with_(), which _is_ valid to call multiple times...
    #       - Think harder about how .nest() leaves .from_() seeded with the next table name...
    #       - Think harder about fluent api like .limit(), which is useful e.g. .inspect(lambda q: q.limit(10).df)
    #       - Clean up .name handling -- very stateful and error prone
    def query(self, name=None, **clauses):
        if set(self.clauses.keys()) - {'with'}:
            self = self.nest()
        if name:
            self = self.named(name)
        for clause, x in clauses.items():
            self = self._add_clause(clause, x)
        return self

    def nest(self):
        # Do a little work to keep the `with` clauses flat i/o O(n) nested
        #   - Semantically equivalent to:
        #       return (self
        #           .no_locals
        #           ._from(self, name=self.name)
        #       )
        self.ensure_name()  # Else we clone before .name, and two different names happen
        return (self
            .no_locals
            .with_(
                *(self.clauses.get('with') or []),
                *[self._map_clause('with', lambda qs: None)],
            )
            ._from(self.name)
        )

    def _from(self, q: Q, name=None):
        """Add BQQ's as a with_, add str's directly"""
        cls = type(self)
        if isinstance(q, str):
            if name:
                raise ValueError(f"Can't supply name[{name}] with str q[{q}]")
            return self._add_clause('from_', q)
        else:
            if name:
                q = q.named(name)
            return (self
                .with_(q)
                ._from(q.name)
            )

    # with_ is special
    def with_(self,
        *qs: List[Union[Q, Callable[['BQQ'], Q]]],
        **named_qs: Mapping[str, Union[Q, Callable[['BQQ'], Q]]],
    ):
        """Allow multiple .with_ calls on the same query (instead of requiring a nested query)"""
        qs = [
            # Resolve args
            #   - TODO Think harder about this api -- which of the many variants should we standardize on?
            *[
                q(self) if inspect.isfunction(q) else q  # Can't use callable() b/c BQQ.__call__
                for q in qs
            ],
            *[
                (q(self) if inspect.isfunction(q) else q)  # Can't use callable() b/c BQQ.__call__
                    .named(name)
                for name, q in named_qs.items()
            ],
        ]
        return self._map_clause('with', lambda x: [
            *(x or []),
            *qs,
        ])

    def inspect(self, f, display=None):
        if display is None:
            from IPython.display import display
        display(f(self))
        return self

    # Fluent api
    #   - Useful e.g. for .inspect(lambda q: q.limit(10).df())
    #   - NOTE .limit() = .query(limit=...) is bad b/c the table name gets pinned as tmp which breaks if you reuse it in a .with_() later...
    def subquery           (self, x): return self._add_clause('subquery',           x)
    def union_all          (self, x): return self._add_clause('union_all',          x)
    def union_distinct     (self, x): return self._add_clause('union_distinct',     x)
    def intersect_distinct (self, x): return self._add_clause('intersect_distinct', x)
    def except_distinct    (self, x): return self._add_clause('except_distinct',    x)
    def select             (self, x): return self._add_clause('select',             x)
    # def from_              (self, x): return self._add_clause('from_',              x)  # Not meaningful
    def where              (self, x): return self._add_clause('where',              x)
    def group_by           (self, x): return self._add_clause('group_by',           x)
    def having             (self, x): return self._add_clause('having',             x)
    def window             (self, x): return self._add_clause('window',             x)
    def order_by           (self, x): return self._add_clause('order_by',           x)
    def limit              (self, x): return self._add_clause('limit',              x)

    def _get_clause(self, clause, f=lambda x: x):
        return f(self.clauses.get(clause))

    def _if_clause(self, clause, f=lambda x: x):
        x = self.clauses.get(clause)
        return None if x is None else f(x)

    def _add_clause(self, clause, x):
        if clause in self.clauses:
            raise ValueError(f'Clause[{clause}] already set with value[{self.clauses[clause]}], in self[{self}]')
        else:
            return self.replace(clauses={
                **self.clauses,
                clause: x,
            })

    def _map_clause(self, clause, f):
        return self.replace(clauses={
            **self.clauses,
            clause: f(self.clauses.get(clause)),
        })

    def _fresh_name(self):
        return f'{self.fresh_prefix}%0{self.fresh_len}x' % random.getrandbits(self.fresh_len * 4)

    def _sql(self, q: Q):
        """q.sql if BQQ, else q if str"""
        if isinstance(q, str):
            return strip_margin(q)
        else:
            if q._globals() != self._globals():
                raise ValueError(f'Globals must match: q[{q._globals()}], self[{self._globals()}]')
            else:
                return q.sql

    def _globals(self):
        cls = type(self)
        return {
            f.name: self[f.name]
            for f in cls.fields()
            if f.metadata.get('bqq.global', False)
        }

    def _locals(self):
        cls = type(self)
        return {
            f.name: self[f.name]
            for f in cls.fields()
            if not f.metadata.get('bqq.global', False)
        }


bqq = BQQ()


# TODO Unify with potoo.sql_magics.BQMagics.bq (%bq)
def _bqq(
    query: str,
    max_rows=1000,
    defs=None,
    show_query=False,
    via_extract='infer',  # Use .extract (good for big results) i/o .to_dataframe (good for small results)
    extract_infer_schema=True,  # HACK Very brittle but usually what you want
    extract_infer_threshold_mb=100,
    # TODO Take this as one gs:// uri instead of two separate str args
    extract_bucket='potoo-bqq',
    extract_dir='extract/v0',
    **kwargs,
) -> pd.DataFrame:
    """
    e.g. _bqq('select 42')
    """

    kwargs.setdefault('billing_tier', 3)
    query = query.replace('$', '$$')  # Make '$' safe by assuming no variable references (see bq.Query? for details)
    if defs:
        query = lines(strip_margin(defs), '', strip_margin(query))
    else:
        query = strip_margin(query)

    if not show_query:
        log.info('Running query...')
    else:
        log.info(lines('Running query... [', indent(strip_margin(query)), ']'))
    start_s = time.time()
    query = bq.Query(query).execute(dialect='standard', **kwargs)
    job = query.results.job
    metadata = query.results.metadata
    log.info('[%.0fs] cost[%s] rows[%s] size[%s] url[%s]' % (
        time.time() - start_s,  # Also job.total_time, but prefer user wall clock
        'cached' if job.cache_hit else '$%.4f, %s' % (
            job.bytes_processed / 1024**4 * 5,  # Cost estimate: $5/TB
            humanize.naturalsize(job.bytes_processed),
        ),
        job.total_rows,
        humanize.naturalsize(metadata.size, binary=True),
        bq_url_for_query(query),
    ))

    # Use .extract
    if via_extract == 'infer':
        if metadata.size >= extract_infer_threshold_mb * 1024**2:
            log.info('Using .extract i/o .to_dataframe, because size[%s] ≥ extract_infer_threshold_mb[%s]' % (
                humanize.naturalsize(metadata.size, binary=True),
                extract_infer_threshold_mb,
            ))
            via_extract = True
        else:
            via_extract = False

    start_s = time.time()
    if not via_extract:
        log.info('Fetching results...')
        df = query.results.to_dataframe(max_rows=max_rows)
    else:
        # We use format='csv' with extract() b/c it works better than format='json' or format='avro' (surprisingly!)
        #   - json: some cols go missing (wat), col order not preserved
        #   - avro: injects its own tz datatype into pd.Timestamp, which I didn't try to undo
        #       - Requires additional reqs: fastavro==0.22.3 pandavro==1.4.0
        ext = 'csv.gz'
        basename = '%s' % re.sub(r'[^0-9T]', '-', datetime.utcnow().isoformat())
        path = '%s/%s.%s' % (extract_dir, basename, ext)
        gs_uri = 'gs://%s/%s' % (extract_bucket, path)
        # TODO Report time for extract (like we already do for query and fetch)
        log.info('Extracting results to uri[%s]...' % gs_uri)
        gs.Bucket(extract_bucket).create()
        extract_job = query.results.extract(destination=gs_uri,
            format='csv', compress=True,
        )
        # Raise if extract failed by triggering .result()
        #   - e.g. results >1gb fail [FIXME Requires wildcards: https://cloud.google.com/bigquery/docs/exporting-data]
        #   - extract_job (=None) isn't informative when extract succeeds
        extract_job.result()
        if extract_infer_schema:
            log.info('Peeking .to_dataframe(max_rows=1000) to infer schema for .extract() (disable with extract_infer_schema=False)...')
            _df = query.results.to_dataframe(max_rows=1000)
        log.info('Fetching results from uri[%s]...' % gs_uri)
        fs = gcsfs.GCSFileSystem()
        fs.invalidate_cache()  # HACK Work around https://github.com/dask/gcsfs/issues/5 (spurious FileNotFoundError's)
        with fs.open(gs_uri, 'rb') as f:
            df = (
                pd.read_csv(f, compression='gzip')
                .pipe(lambda df: df if not extract_infer_schema else df
                    .astype(_df.dtypes.to_dict())
                )
            )
    log.info('[%.0fs]' % (time.time() - start_s))

    return df


def bqq_from_url(
    bq_url: str,
    context: datalab.context.Context = None,
    **kwargs,
) -> pd.DataFrame:
    """
    e.g. bqq_from_url('https://bigquery.cloud.google.com/results/dwh-v2:bquijob_41b598ad_1614361c9a6')
    """
    (project_id, job_id) = re.match('^https://bigquery.cloud.google.com/results/(.*?):(.*)$', bq_url).groups()
    return bqq_from_job_id(job_id, context, **kwargs)


def bqq_from_job_id(
    job_id: str,
    context: datalab.context.Context = None,
    **kwargs,
) -> pd.DataFrame:
    """
    e.g. bqq_from_job_id('bquijob_41b598ad_1614361c9a6')
    """
    context = context or datalab.context.Context.default()
    job = bq.Job(job_id, context)
    job_data = job._api.jobs_get(job.id)
    query = job_data['configuration']['query']['query']
    log.info(f'Running job_id[{job_id}] with query[\n{textwrap.indent(query, prefix="  ")}\n]...')
    # Have to re-run query since tmp tables that hold job results aren't shared to other users
    #   - e.g. if you open someone else's bq job url in the web ui, you see no results and have to click "Run Query"
    return bqq(query, **kwargs)


def df_to_bq(
    df: pd.DataFrame,
    table: Union[str, Tuple[str, str, str]],  # 'dataset.table' | 'project:dataset.table' | (project, dataset, table)
    schema: Mapping[str, str] = None,  # https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#TableFieldSchema
    table_kwargs=None,
    overwrite=False,
    create_kwargs=None,
    load_bucket='potoo-bqq',
    load_dir='load/v0',
    load_kwargs=None,
):
    start_s = time.time()

    # Infer schema, if not given
    if schema is None:
        dtypes = dict(df.dtypes)
        schema = {
            k: {
                'object': 'string',
                # TODO Add mappings for more types other than string
            }.get(v.name, 'string')
            for k, v in dtypes.items()
        }
        log.info('Inferred schema[%(schema)s] from dtypes[%(dtypes)s]' % locals())

    # Create table
    log.info('Creating table[%(table)s] with schema[%(schema)s]...' % locals())
    _table = bq.Table(**{
        'name': table,
        **(table_kwargs or {}),
    })
    _table.create(**{
        'schema': [dict(name=k, type=v) for k, v in schema.items()],
        'overwrite': overwrite,
        **(create_kwargs or {}),
    })

    # Upload df -> gs
    basename = re.sub(r'[^0-9T]', '-', datetime.utcnow().isoformat())
    gs_uri = 'gs://%(load_bucket)s/%(load_dir)s/%(basename)s.csv' % locals()
    log.info('Uploading df to gs[%(gs_uri)s]...' % locals())
    gs.Bucket(load_bucket).create()
    fs = gcsfs.GCSFileSystem()
    fs.invalidate_cache()  # HACK Work around https://github.com/dask/gcsfs/issues/5 (spurious FileNotFoundError's)
    with fs.open(gs_uri, 'wt') as f:
        df.to_csv(f,
            index=False,
            # TODO Think about date formats, etc.
            #   - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html
            # FIXME Ignored when f is a file i/o a path
            #   - https://github.com/pandas-dev/pandas/issues/22555
            # compression='gzip',
        )

    # Load table <- gs
    log.info('Loading table[%(table)s] from gs[%(gs_uri)s]...' % locals())
    job = _table.load(**{
        'source': gs_uri,
        'mode': 'append',  # Assume table is empty, since we just created it above
        'source_format': 'csv',
        'csv_options': bq.CSVOptions(
            skip_leading_rows=1,  # Ignore header row
        ),
        **(load_kwargs or {}),
    })
    job.result()  # Throw if failed

    # Print stats
    log.info('[%.0fs] rows[%s] size[%s]' % (
        time.time() - start_s,
        _table.metadata.rows,
        humanize.naturalsize(_table.metadata.size, binary=True),
    ))


def bqi():
    next_query = ''
    last_df = None
    for line in sys.stdin:
        if line.strip():
            next_query += line
        elif next_query.strip():
            try:
                last_df = bqq(next_query)
                print(last_df)
            except Exception as e:
                print(e)
            next_query = ''
            print()
        else:
            print(last_df)
            print()
