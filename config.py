import os

# 获取当前文件所在的目录
basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'a-hard-to-guess-string' # 生产环境建议使用环境变量
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'instance', 'app.db') # 默认使用 SQLite
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # 可以添加其他应用配置
    NEWS_API_KEY = os.environ.get('NEWS_API_KEY') # 示例：新闻 API 密钥
    MARKET_API_KEY = os.environ.get('MARKET_API_KEY') # 示例：行情 API 密钥 