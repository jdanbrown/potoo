# TODO Throw away potoo.bq and rename this to replace it (potoo.bqq -> potoo.bq)

import re
import textwrap
import time

import datalab
import datalab.bigquery as bq
import datalab.storage as gs
from datalab.bigquery._utils import TableName
import humanize
import pandas as pd

from potoo import humanize


def bq_url_for_query(query: bq.QueryJob) -> str:
    return 'https://console.cloud.google.com/bigquery?project=%s&j=bq:US:%s&page=queryresults' % (query.results.name.project_id, query.results.job_id)


# TODO Unify with potoo.sql_magics.BQMagics.bq (%bq)
def bqq(sql: str, max_rows=1000, defs=None, show_query=False, **kwargs) -> pd.DataFrame:
    """
    e.g. bqq('select 42')
    """

    kwargs.setdefault('billing_tier', 3)
    sql = sql.replace('$', '$$')  # Make '$' safe by assuming no variable references (see bq.Query? for details)
    if defs: sql = defs + sql  # Prepend defs, if given

    if not show_query:
        print('Running query...')
    else:
        print('Running query[%s]...' % sql)
    start_s = time.time()
    query = bq.Query(sql).execute(dialect='standard', **kwargs)
    job = query.results.job
    metadata = query.results.metadata
    print('[%.0fs] cost[%s] rows[%s] size[%s] url[%s]' % (
        time.time() - start_s,  # Also job.total_time, but prefer user wall clock
        'cached' if job.cache_hit else '$%.4f, %s' % (
            job.bytes_processed / 1024**4 * 5,  # Cost estimate: $5/TB
            humanize.naturalsize(job.bytes_processed),
        ),
        job.total_rows,
        humanize.naturalsize(metadata.size),
        bq_url_for_query(query),
    ))

    print('Fetching results...')
    start_s = time.time()
    df = query.results.to_dataframe(max_rows=max_rows)
    print('[%.0fs]' % (time.time() - start_s))

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
    sql = job_data['configuration']['query']['query']
    print(f'Running job_id[{job_id}] with sql[\n{textwrap.indent(sql, prefix="  ")}\n]...')
    # Have to re-run query since tmp tables that hold job results aren't shared to other users
    #   - e.g. if you open someone else's bq job url in the web ui, you see no results and have to click "Run Query"
    return bqq(sql, **kwargs)


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
