# TODO Throw away potoo.bq and rename this to replace it (potoo.bqq -> potoo.bq)

from datetime import datetime
import re
import textwrap
import time

import datalab
import datalab.bigquery as bq
import datalab.storage as gs
from datalab.bigquery._utils import TableName
import gcsfs  # XXX Replace with pd.read_json in pandas 0.24.x [https://stackoverflow.com/a/50201179/397334]
import humanize
import pandas as pd
import pandavro

from potoo import humanize


def bq_url_for_query(query: bq.QueryJob) -> str:
    return 'https://console.cloud.google.com/bigquery?project=%s&j=bq:US:%s&page=queryresults' % (query.results.name.project_id, query.results.job_id)


# TODO Unify with potoo.sql_magics.BQMagics.bq (%bq)
def bqq(
    sql: str,
    max_rows=1000,
    defs=None,
    show_query=False,
    via_extract='infer',  # Use .extract (good for big results) i/o .to_dataframe (good for small results)
    extract_infer_schema=True,  # HACK Very brittle but usually what you want
    extract_infer_threshold_mb=100,
    # TODO Take this as one gs:// uri instead of two separate str args
    extract_bucket='potoo-bqq-extract',
    extract_dir='v0',
    **kwargs,
) -> pd.DataFrame:
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
        humanize.naturalsize(metadata.size, binary=True),
        bq_url_for_query(query),
    ))

    # Use .extract
    if via_extract == 'infer':
        if metadata.size >= extract_infer_threshold_mb * 1024**2:
            print('Using .extract i/o .to_dataframe, because size[%s] ≥ extract_infer_threshold_mb[%s]' % (
                humanize.naturalsize(metadata.size, binary=True),
                extract_infer_threshold_mb,
            ))
            via_extract = True
        else:
            via_extract = False

    start_s = time.time()
    if not via_extract:
        print('Fetching results...')
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
        print('Extracting results to uri[%s]...' % gs_uri)
        gs.Bucket(extract_bucket).create()
        extract_job = query.results.extract(destination=gs_uri,
            format='csv', compress=True,
        )
        # Raise if extract failed by triggering .result()
        #   - e.g. results >1gb fail [FIXME Requires wildcards: https://cloud.google.com/bigquery/docs/exporting-data]
        #   - extract_job (=None) isn't informative when extract succeeds
        extract_job.result()
        if extract_infer_schema:
            print('Peeking .to_dataframe(max_rows=1000) to infer schema for .extract() (disable with extract_infer_schema=False)...')
            _df = query.results.to_dataframe(max_rows=1000)
        print('Fetching results from uri[%s]...' % gs_uri)
        fs = gcsfs.GCSFileSystem()
        fs.invalidate_cache()  # HACK Work around https://github.com/dask/gcsfs/issues/5 (spurious FileNotFoundError's)
        with fs.open(gs_uri) as f:
            df = (
                pd.read_csv(f, compression='gzip')
                .pipe(lambda df: df if not extract_infer_schema else df
                    .astype(_df.dtypes.to_dict())
                )
            )
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
