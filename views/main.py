from flask import Blueprint, render_template, redirect, url_for, session

# 创建一个名为 'main' 的 Blueprint
bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    """首页路由，检查用户是否已登录，未登录则重定向到登录页面"""
    # 检查用户是否已登录
    if 'user_id' not in session:
        # 未登录，重定向到登录页面
        return redirect(url_for('auth.login'))

    # 已登录，显示首页
    return render_template('index.html')

# 可以在这里添加其他通用页面的路由，例如关于页面
# @bp.route('/about')
# def about():
#    return render_template('about.html')