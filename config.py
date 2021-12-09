bind = '127.0.0.1:8000'
backlog = 2048

workers = 4
worker_class = 'gevent'
worker_connections = 1000
timeout = 30
keepalive = 20


daemon = False

pidfile = 'logs/gunicorn.pid'

graceful_timeout = 10      # restart timeout
forwarded_allow_ips = '*'  # allowed visitor ip

#capture_output = True
#loglevel = 'info'
#errorlog = 'logs/error.log'
#debug = True

errorlog = '-'
loglevel = 'info'
accesslog = '-'
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'


