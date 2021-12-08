from redis import Redis


class Dev():
    ENV = 'development'
    SECRET_KEY='123456'

    SESSION_TYPE = 'redis'
    SESSION_REDIS = Redis(host='127.0.0.1', db=1)

    STATIC_FOLDER = '/resource'
    STATIC_URL_PATH = '/static'

    # SESSION_FILE_DIR = 'path'

    # STATIC_URL_PARH = '/static'
    # STATIC_FOLDER = '/resource'

    # sqlalchemy database orm
    SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://cai:123456@127.0.0.1:3306/company'
    SQLALCHEMY_TRACK_MODIFICATIONS = True      # coexist with previous version
    SQLALCHEMY_COMMIT_ON_TEARDOWN = True  # commit transaction when destroy resources
    #SQLALCHEMY_ECHO = True       # display debug sql message

    #CACHE_TYPE = 'redis'
    #CACHE_REDIS_HOST = '127.0.0.1'
    #CACHE_REDIS_PORT = 6379
    #CACHE_REDIS_DB = 5

