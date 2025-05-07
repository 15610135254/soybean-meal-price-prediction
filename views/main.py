from flask import Blueprint, render_template

# 创建一个名为 'main' 的 Blueprint
bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    """首页路由"""
    return render_template('index.html')

# 可以在这里添加其他通用页面的路由，例如关于页面
# @bp.route('/about')
# def about():
#    return render_template('about.html') 