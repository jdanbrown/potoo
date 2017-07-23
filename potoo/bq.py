# TODO Throw this away and use https://github.com/googledatalab/pydatalab instead
#   - Supports full API, e.g. use_cached_results, max_billing_tier
#   - See examples: https://github.com/googledatalab/notebooks
#   - But verify it works with arrays and structs...

from collections import OrderedDict
import pandas as pd

from potoo.pandas import pd_read_bq, bq_default_project
import potoo.pandas_io_gbq_par_io as gbq


# TODO Use https://googlecloudplatform.github.io/google-cloud-python/stable/bigquery-usage.html
#   - TODO Needs a little extra setup for auth
class BQ:

    def __init__(self, project_id):
        self.project_id = project_id
        self.service = gbq.GbqConnector(project_id=project_id).service

    def pd_read(self, *args, **kw):
        # Reuses self.project_id but not self.services (makes its own)
        return pd_read_bq(*args, **kw, project_id=self.project_id)

    # TODO Stop doing stuff like this; use `bq ls` and/or google-cloud-python instead
    def tables_list(self, dataset_id):
        res = bq.service.tables().list(projectId=bq.project_id, datasetId='rt_v1_s2').execute()
        return pd.DataFrame(res['tables']).apply(axis=1, func=lambda x: pd.Series(OrderedDict([
            ('type', x['type']),
            ('project_id', x['tableReference']['projectId']),
            ('dataset_id', x['tableReference']['datasetId']),
            ('table_id', x['tableReference']['tableId']),
        ])))

    def table_get(self, dataset_id, table_id):
        return self.service.tables().get(projectId=self.project_id, datasetId=dataset_id, tableId=table_id).execute()

    def table_schema(self, dataset_id, table_id):
        return self.table_get(dataset_id, table_id)['schema']['fields']

    def table_head(self, dataset_id, table_id, limit=10):
        return self.pd_read('select * from `%(dataset_id)s.%(table_id)s` limit %(limit)s' % locals())

    # bq.table_summary
    #   - cf. https://www.postgresql.org/docs/current/static/view-pg-stats.html
    #   - TODO histogram (graph?)
    #   - TODO Add perc to approx_top_count [{'count':100, 'value':'a'}, {'count':50, 'value':'b'}, ...]
    def table_summary(
        self,
        dataset_id,
        table_id,
        fields_f=lambda xs: xs,  # e.g. to subset when bq barfs on too many things
    ):
        return self.pd_read('''
            select * from unnest((
                select %(array_agg_expr)s
                from %(dataset_id)s.%(table_id)s
            ))
        ''' % dict(
                array_agg_expr = '[\n%s\n]' % ',\n'.join(
                    '''
                        struct(
                            '%(name)s' as name,
                            count(*) as count,
                            approx_count_distinct(`%(name)s`) as `distinct`,
                            sum(case when `%(name)s` is null or `%(name)s` = '' then 1 else 0 end) / count(*) as null_frac,
                            approx_top_count(`%(name)s`, 5) as top_by_count,
                            approx_quantiles(`%(name)s`, 4) as quantiles
                        )
                    ''' % field
                    for field in fields_f(bq.table_schema(dataset_id, table_id))
                ),
                dataset_id = dataset_id,
                table_id = table_id,
            )
        )


# Connect bq
bq = BQ(bq_default_project())

# XXX Testing
# importlib.reload(gbq)
# bq.table_head('rt_v1_s1', 'outgoing_Neighborhood_Data_OpenDoor_AttomDataNeighborhoodAirportNoise_zip_OpenDoor_AttomDataNeighborhoodAirportNoise')
# bq.table_summary('rt_v1_s1', 'outgoing_Neighborhood_Data_OpenDoor_AttomDataNeighborhoodAirportNoise_zip_OpenDoor_AttomDataNeighborhoodAirportNoise')
# bq.pd_read('''
#     select
#         address_token,
#         4,
#         'foo',
#         true,
#         struct('a',3,true),
#         struct('b' as x, 4 as y, false as z),
#         [1,2,3],
#         'foo'
#     from (select address_token from dwellings_v0.addresses limit 2)
# ''')
