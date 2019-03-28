from typing import Union

import sqlalchemy as sqla
import sqlalchemy.orm as sqlo


def sqla_session(x: Union['db_url', 'engine']):
    """
    Do a pile of sane defaults to get a sqla session. Example usage:

        db = sqla_session(...)
        df = pd.read_sql(sql=..., con=db.bind)
    """

    # Resolve args
    if isinstance(x, str):
        db_url = x
        if '/' not in db_url:
            db_url = f'postgres://localhost/{db_url}'
        eng = sqla.create_engine(
            db_url,
            convert_unicode=True,
        )
    elif isinstance(x, sqla.engine.base.Engine):
        eng = x
    else:
        raise ValueError(f"Expected db_url or engine, got: {x!r}")

    # Make session
    return sqlo.scoped_session(
        sqlo.sessionmaker(
            autocommit=True,
            bind=eng,
        ),
    )
