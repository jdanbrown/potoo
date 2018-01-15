import sqlalchemy as sqla
import sqlalchemy.orm as sqlo


def sqla_session(db_url):
    """
    Do a pile of sane defaults to get a sqla session. Example usage:

        db = sqla_session(...)
        df = pd.read_sql(sql=..., con=db.bind)
    """
    if '/' not in db_url:
        db_url = f'postgres://localhost/{db_url}'
    return sqlo.scoped_session(
        sqlo.sessionmaker(
            autocommit=True,
            bind=sqla.create_engine(
                db_url,
                convert_unicode=True,
            ),
        ),
    )
