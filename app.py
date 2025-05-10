from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from config import Config
import os

db = SQLAlchemy()

def create_app(config_class=Config):
    """应用工厂函数"""
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(config_class)

    # 确保 instance 文件夹存在
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # 初始化数据库
    db.init_app(app)

    # 注册 Blueprints
    from views.main import bp as main_bp
    app.register_blueprint(main_bp)

    from views.visualization import bp as viz_bp
    app.register_blueprint(viz_bp, url_prefix='/viz')

    from views.news import bp as news_bp
    app.register_blueprint(news_bp, url_prefix='/news')

    from views.forum import bp as forum_bp
    app.register_blueprint(forum_bp, url_prefix='/forum')

    # 注册认证蓝图
    from views.auth import bp as auth_bp
    app.register_blueprint(auth_bp, url_prefix='/auth')


    # 配置日志
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    return app

# 在脚本主入口创建应用实例 (方便直接运行 python app.py 启动开发服务器)
if __name__ == '__main__':
    import argparse

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='启动豆粕期货价格预测系统')
    parser.add_argument('--port', type=int, default=5000, help='服务器端口号')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='服务器主机地址')

    # 解析命令行参数
    args = parser.parse_args()

    app = create_app()
    app.run(debug=True, host=args.host, port=args.port)