import os

class Config(object):
    #signature key 
    SECRET_KY = os.environ.get('SECRET_KEY') or "secret_string"
    